from __future__ import annotations

from ...ast_nodes import (
    Block,
    LetStmt,
    ExprStmt,
    Identifier,
    ScanExpr,
    WhileExpr,
    FoldExpr,
    _ScanGrad,
    _WhileGrad,
    _FoldGrad,
)
from ...types import MaomiType, ScalarType, ArrayType, StructType, StructArrayType
from ...errors import MaomiError
from .utils import _mlir_type, _MLIR_ETYPE, _block_references_var


class LoopCodegenMixin:

    def _slice_struct_array_element(self, seq_ssa: str, idx_ssa: str,
                                       sa_type: StructArrayType) -> str:
        """Slice a single element from a StructArrayType at a dynamic index.
        Extracts each field, slices axis 0, reconstructs the struct."""
        elem_struct = sa_type.struct_type
        field_results = []

        for i, (fname, ftype) in enumerate(elem_struct.fields):
            # Compute batched field type (field with leading dims)
            if isinstance(ftype, ScalarType):
                batched = ArrayType(ftype.base, sa_type.dims)
            elif isinstance(ftype, ArrayType):
                batched = ArrayType(ftype.base, sa_type.dims + ftype.dims)
            else:
                raise MaomiError("codegen: unsupported field type in struct array", "<codegen>", 0, 0)

            # Extract field from SoA tuple
            field_ssa = self._fresh()
            self._emit(
                f"{field_ssa} = stablehlo.get_tuple_element {seq_ssa}[{i}] "
                f": ({_mlir_type(sa_type)}) -> {_mlir_type(batched)}"
            )

            # Slice this field using existing _slice_element
            sliced = self._slice_element(field_ssa, idx_ssa, batched, ftype, iter_axis=0)
            field_results.append(sliced)

        # Reconstruct struct from sliced fields
        vals_str = ", ".join(field_results)
        result_mlir = _mlir_type(elem_struct)
        var = self._fresh()
        self._emit(f"{var} = stablehlo.tuple {vals_str} : {result_mlir}")
        return var

    def _slice_element(self, seq_ssa: str, idx_ssa: str, seq_type: ArrayType, elem_type: MaomiType, iter_axis: int = 0) -> str:
        """Slice a single element from a 1D+ array at a dynamic index along iter_axis."""
        mlir_seq = _mlir_type(seq_type)

        # Build slice sizes: all dims kept except iter_axis which becomes 1
        slice_sizes = list(seq_type.dims)
        slice_sizes[iter_axis] = 1
        sizes_str = ", ".join(str(s) for s in slice_sizes)
        sliced_type = ArrayType(seq_type.base, tuple(slice_sizes))
        mlir_sliced = _mlir_type(sliced_type)

        # Build start indices: 0 for all dims except iter_axis which is idx_ssa
        start_ssas = []
        for i in range(len(seq_type.dims)):
            if i == iter_axis:
                start_ssas.append(idx_ssa)
            else:
                z = self._fresh()
                self._emit(f"{z} = stablehlo.constant dense<0> : tensor<i32>")
                start_ssas.append(z)
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        sliced = self._fresh()
        self._emit(
            f"{sliced} = stablehlo.dynamic_slice {seq_ssa}, {starts_str}, "
            f"sizes = [{sizes_str}] : ({mlir_seq}, {start_types}) -> {mlir_sliced}"
        )
        # Reshape to remove the iter_axis dim (size 1)
        result = self._fresh()
        mlir_elem = _mlir_type(elem_type)
        self._emit(f"{result} = stablehlo.reshape {sliced} : ({mlir_sliced}) -> {mlir_elem}")
        return result

    def _update_element(self, arr_ssa: str, elem_ssa: str, idx_ssa: str,
                         arr_type: ArrayType, elem_type: MaomiType, iter_axis: int = 0) -> str:
        """Update a single element in an array at a dynamic index along iter_axis."""
        mlir_arr = _mlir_type(arr_type)
        # Reshape elem to insert size-1 dim at iter_axis
        if isinstance(elem_type, ScalarType):
            update_dims = (1,)
        elif isinstance(elem_type, ArrayType):
            update_dims = elem_type.dims[:iter_axis] + (1,) + elem_type.dims[iter_axis:]
        else:
            update_dims = (1,)
        update_type = ArrayType(arr_type.base, update_dims)
        mlir_update = _mlir_type(update_type)

        reshaped = self._fresh()
        self._emit(f"{reshaped} = stablehlo.reshape {elem_ssa} : ({_mlir_type(elem_type)}) -> {mlir_update}")

        # Build start indices: 0 for all dims except iter_axis which is idx_ssa
        start_ssas = []
        for i in range(len(arr_type.dims)):
            if i == iter_axis:
                start_ssas.append(idx_ssa)
            else:
                z = self._fresh()
                self._emit(f"{z} = stablehlo.constant dense<0> : tensor<i32>")
                start_ssas.append(z)
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.dynamic_update_slice {arr_ssa}, {reshaped}, {starts_str} "
            f": ({mlir_arr}, {mlir_update}, {start_types}) -> {mlir_arr}"
        )
        return result

    # -- While loop codegen --

    def _gen_while(self, expr: WhileExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        state_type = self._type_of(expr.init)
        mlir_state = _mlir_type(state_type)

        uid = self._counter
        state_name = f"whileS{uid}"

        while_result = self._fresh()
        header = f"%{state_name} = {init_val}"
        self._emit(f"{while_result}:1 = stablehlo.while({header}) : {mlir_state}")

        # Cond region
        self._emit("cond {")
        self._indent += 1
        cond_env = dict(env)
        cond_env[expr.state_var] = f"%{state_name}"
        cond_val = self._gen_block(expr.cond, cond_env)
        self._emit(f"stablehlo.return {cond_val} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1
        body_env = dict(env)
        body_env[expr.state_var] = f"%{state_name}"
        new_state = self._gen_block(expr.body, body_env)
        self._emit(f"stablehlo.return {new_state} : {mlir_state}")
        self._indent -= 1
        self._emit("}")

        return f"{while_result}#0"

    def _gen_while_grad(self, expr: _WhileGrad, env: dict[str, str]) -> str:
        """Emit augmented forward + reverse while for bounded while backward pass."""
        # Generate the forward result (the original while loop)
        fwd_val = self._gen_expr(expr.forward_result, env)

        state_type = self._type_of(expr.init)
        mlir_state = _mlir_type(state_type)
        mlir_i32 = "tensor<i32>"
        mlir_bool = "tensor<i1>"
        max_iters = expr.max_iters

        # Build trajectory type: tensor<max_iters x ...state_shape>
        if isinstance(state_type, ScalarType):
            traj_type = ArrayType(state_type.base, (max_iters,))
        elif isinstance(state_type, ArrayType):
            traj_type = ArrayType(state_type.base, (max_iters,) + state_type.dims)
        else:
            raise MaomiError("codegen: while grad only supports scalar/array state", "<codegen>", 0, 0)
        mlir_traj = _mlir_type(traj_type)

        # ── Forward augmented while: save trajectory ──
        init_val = self._gen_expr(expr.init, env)

        # Pre-allocate trajectory buffer (zeros)
        traj_init = self._fresh()
        self._emit(f"{traj_init} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        # Step counter = 0
        step_init = self._fresh()
        self._emit(f"{step_init} = stablehlo.constant dense<0> : {mlir_i32}")

        # running = true
        run_init = self._fresh()
        self._emit(f"{run_init} = stablehlo.constant dense<true> : {mlir_bool}")

        uid = self._counter
        s_name = f"fwdS{uid}"
        step_name = f"fwdStep{uid}"
        traj_name = f"fwdTraj{uid}"
        run_name = f"fwdRun{uid}"

        fwd_result = self._fresh()
        header_parts = ", ".join([
            f"%{s_name} = {init_val}",
            f"%{step_name} = {step_init}",
            f"%{traj_name} = {traj_init}",
            f"%{run_name} = {run_init}",
        ])
        types_str = f"{mlir_state}, {mlir_i32}, {mlir_traj}, {mlir_bool}"
        self._emit(f"{fwd_result}:4 = stablehlo.while({header_parts}) : {types_str}")

        # Cond: running && step < max_iters
        self._emit("cond {")
        self._indent += 1
        max_v = self._fresh()
        self._emit(f"{max_v} = stablehlo.constant dense<{max_iters}> : {mlir_i32}")
        lt_v = self._fresh()
        self._emit(f"{lt_v} = stablehlo.compare LT, %{step_name}, {max_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> {mlir_bool}")
        and_v = self._fresh()
        self._emit(f"{and_v} = stablehlo.and %{run_name}, {lt_v} : {mlir_bool}")
        self._emit(f"stablehlo.return {and_v} : {mlir_bool}")
        self._indent -= 1

        # Body: save state to trajectory, run body, check cond on new state
        self._emit("} do {")
        self._indent += 1

        # Save current state to trajectory at position step
        if isinstance(state_type, ScalarType):
            reshaped = self._fresh()
            self._emit(f"{reshaped} = stablehlo.reshape %{s_name} : ({mlir_state}) -> tensor<1x{_MLIR_ETYPE[state_type.base]}>")
            new_traj = self._fresh()
            self._emit(f"{new_traj} = stablehlo.dynamic_update_slice %{traj_name}, {reshaped}, %{step_name} : ({mlir_traj}, tensor<1x{_MLIR_ETYPE[state_type.base]}>, {mlir_i32}) -> {mlir_traj}")
        elif isinstance(state_type, ArrayType):
            reshaped = self._fresh()
            inner_shape = "x".join(str(d) for d in state_type.dims)
            etype = _MLIR_ETYPE[state_type.base]
            self._emit(f"{reshaped} = stablehlo.reshape %{s_name} : ({mlir_state}) -> tensor<1x{inner_shape}x{etype}>")
            # Start indices: step for dim 0, then zeros for remaining dims
            zeros = [self._fresh() for _ in state_type.dims]
            for z in zeros:
                self._emit(f"{z} = stablehlo.constant dense<0> : {mlir_i32}")
            start_indices = f"%{step_name}, " + ", ".join(zeros)
            slice_shape = f"1x{inner_shape}x{etype}"
            new_traj = self._fresh()
            self._emit(f"{new_traj} = stablehlo.dynamic_update_slice %{traj_name}, {reshaped}, {start_indices} : ({mlir_traj}, tensor<{slice_shape}>, {', '.join([mlir_i32] * (1 + len(state_type.dims)))}) -> {mlir_traj}")

        # Run body
        body_env = dict(env)
        body_env[expr.state_var] = f"%{s_name}"
        new_state = self._gen_block(expr.body, body_env)

        # Check condition on new state
        cond_env = dict(env)
        cond_env[expr.state_var] = new_state
        still_running = self._gen_block(expr.cond, cond_env)

        # Increment step
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_step = self._fresh()
        self._emit(f"{new_step} = stablehlo.add %{step_name}, {one_v} : {mlir_i32}")

        vals_str = f"{new_state}, {new_step}, {new_traj}, {still_running}"
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract: num_iters = fwd_result#1, trajectory = fwd_result#2
        fwd_num_iters = f"{fwd_result}#1"
        fwd_trajectory = f"{fwd_result}#2"

        # ── Backward while: reverse through trajectory ──
        adj_val = self._gen_expr(expr.adj, env)

        # Start step = num_iters - 1
        one_bwd = self._fresh()
        self._emit(f"{one_bwd} = stablehlo.constant dense<1> : {mlir_i32}")
        start_step = self._fresh()
        self._emit(f"{start_step} = stablehlo.subtract {fwd_num_iters}, {one_bwd} : {mlir_i32}")

        uid2 = self._counter
        badj_name = f"bwdAdj{uid2}"
        bstep_name = f"bwdStep{uid2}"

        bwd_result = self._fresh()
        bwd_header = f"%{badj_name} = {adj_val}, %{bstep_name} = {start_step}"
        bwd_types = f"{mlir_state}, {mlir_i32}"
        self._emit(f"{bwd_result}:2 = stablehlo.while({bwd_header}) : {bwd_types}")

        # Cond: step >= 0
        self._emit("cond {")
        self._indent += 1
        zero_v = self._fresh()
        self._emit(f"{zero_v} = stablehlo.constant dense<0> : {mlir_i32}")
        ge_v = self._fresh()
        self._emit(f"{ge_v} = stablehlo.compare GE, %{bstep_name}, {zero_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> {mlir_bool}")
        self._emit(f"stablehlo.return {ge_v} : {mlir_bool}")
        self._indent -= 1

        # Body: read saved state, evaluate derivative, multiply
        self._emit("} do {")
        self._indent += 1

        # Read saved state from trajectory at step
        if isinstance(state_type, ScalarType):
            etype = _MLIR_ETYPE[state_type.base]
            sliced = self._fresh()
            self._emit(f"{sliced} = stablehlo.dynamic_slice {fwd_trajectory}, %{bstep_name}, sizes = [1] : ({mlir_traj}, {mlir_i32}) -> tensor<1x{etype}>")
            saved_state = self._fresh()
            self._emit(f"{saved_state} = stablehlo.reshape {sliced} : (tensor<1x{etype}>) -> {mlir_state}")
        elif isinstance(state_type, ArrayType):
            inner_shape = "x".join(str(d) for d in state_type.dims)
            etype = _MLIR_ETYPE[state_type.base]
            zeros_bwd = [self._fresh() for _ in state_type.dims]
            for z in zeros_bwd:
                self._emit(f"{z} = stablehlo.constant dense<0> : {mlir_i32}")
            start_indices_bwd = f"%{bstep_name}, " + ", ".join(zeros_bwd)
            n_idx = 1 + len(state_type.dims)
            slice_sizes = f"1x{inner_shape}x{etype}"
            sliced = self._fresh()
            self._emit(f"{sliced} = stablehlo.dynamic_slice {fwd_trajectory}, {start_indices_bwd}, sizes = [1, {', '.join(str(d) for d in state_type.dims)}] : ({mlir_traj}, {', '.join([mlir_i32] * n_idx)}) -> tensor<{slice_sizes}>")
            saved_state = self._fresh()
            self._emit(f"{saved_state} = stablehlo.reshape {sliced} : (tensor<{slice_sizes}>) -> {mlir_state}")

        # Evaluate d_body_d_state with state_var = saved_state
        deriv_env = dict(env)
        deriv_env[expr.state_var] = saved_state
        d_val = self._gen_expr(expr.d_body_d_state, deriv_env)

        # new_adj = adj * d_val
        new_adj = self._fresh()
        self._emit(f"{new_adj} = stablehlo.multiply %{badj_name}, {d_val} : {mlir_state}")

        # Decrement step
        one_bwd2 = self._fresh()
        self._emit(f"{one_bwd2} = stablehlo.constant dense<1> : {mlir_i32}")
        new_bstep = self._fresh()
        self._emit(f"{new_bstep} = stablehlo.subtract %{bstep_name}, {one_bwd2} : {mlir_i32}")

        self._emit(f"stablehlo.return {new_adj}, {new_bstep} : {bwd_types}")
        self._indent -= 1
        self._emit("}")

        # Result is the accumulated adjoint (index 0)
        return f"{bwd_result}#0"

    def _gen_scan(self, expr: ScanExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]
        result_type = self._type_of(expr)

        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: scan sequence must be array", "<codegen>", 0, 0)

        bd = self._batch_depth
        iter_axis = bd  # batch dims occupy 0..bd-1, iteration is along dim bd

        seq_len = seq_types[0].dims[iter_axis]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: scan requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        # Element types: strip dim at iter_axis (keep batch dims before it)
        elem_types: list[MaomiType] = []
        for st in seq_types:
            remaining = st.dims[:iter_axis] + st.dims[iter_axis + 1:]
            if len(remaining) == 0:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, remaining))

        mlir_init = _mlir_type(init_type)
        mlir_result = _mlir_type(result_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]
        mlir_i32 = "tensor<i32>"

        # Create initial values outside the while
        counter_var = self._fresh()
        if expr.reverse:
            self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")
        else:
            self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        output_var = self._fresh()
        self._emit(f"{output_var} = stablehlo.constant dense<0.000000e+00> : {mlir_result}")

        # Build while arg names and types
        n_seqs = len(seq_vals)
        uid = self._counter
        ctr_name = f"iterC{uid}"
        carry_name = f"iterK{uid}"
        out_name = f"iterO{uid}"
        seq_names = [f"iterS{uid}_{i}" for i in range(n_seqs)]

        arg_names = [ctr_name, carry_name, out_name] + seq_names
        init_vals_list = [counter_var, init_val, output_var] + seq_vals
        arg_types = [mlir_i32, mlir_init, mlir_result] + mlir_seqs
        n_args = len(arg_names)

        # Emit custom while format: stablehlo.while(%name = %init, ...) : types
        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals_list[i]}" for i in range(n_args)
        )
        types_str = ", ".join(arg_types)
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond region
        self._emit("cond {")
        self._indent += 1
        limit_v = self._fresh()
        if expr.reverse:
            self._emit(f"{limit_v} = stablehlo.constant dense<0> : {mlir_i32}")
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare GE, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        else:
            self._emit(f"{limit_v} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
            c_cmp = self._fresh()
            self._emit(f"{c_cmp} = stablehlo.compare LT, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        # Slice elements from each sequence (skip if elem_var unused in body)
        body_env = dict(env)
        body_env[expr.carry_var] = f"%{carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                b_elem = self._slice_element(f"%{seq_names[i]}", f"%{ctr_name}", st, et, iter_axis=iter_axis)
                body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, body_env)

        # Update output
        new_output = self._update_element(f"%{out_name}", new_carry, f"%{ctr_name}", result_type, init_type, iter_axis=iter_axis)

        # Update counter
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_counter = self._fresh()
        if expr.reverse:
            self._emit(f"{new_counter} = stablehlo.subtract %{ctr_name}, {one_v} : {mlir_i32}")
        else:
            self._emit(f"{new_counter} = stablehlo.add %{ctr_name}, {one_v} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_carry, new_output] + [f"%{sn}" for sn in seq_names]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Result is the output buffer (index 2 in while results)
        return f"{while_result}#2"

    # -- Fold codegen --

    def _gen_fold(self, expr: FoldExpr, env: dict[str, str]) -> str:
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]

        for st in seq_types:
            if not isinstance(st, (ArrayType, StructArrayType)):
                raise MaomiError("codegen: fold sequence must be array", "<codegen>", 0, 0)

        bd = self._batch_depth
        iter_axis = bd

        # Get sequence length from first sequence
        first_st = seq_types[0]
        if isinstance(first_st, StructArrayType):
            seq_len = first_st.dims[0]
        else:
            seq_len = first_st.dims[iter_axis]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: fold requires concrete sequence length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            if isinstance(st, StructArrayType):
                if len(st.dims) == 1:
                    elem_types.append(st.struct_type)
                else:
                    elem_types.append(StructArrayType(st.struct_type, st.dims[1:]))
            else:
                remaining = st.dims[:iter_axis] + st.dims[iter_axis + 1:]
                if len(remaining) == 0:
                    elem_types.append(ScalarType(st.base))
                else:
                    elem_types.append(ArrayType(st.base, remaining))

        mlir_i32 = "tensor<i32>"

        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<0> : {mlir_i32}")

        uid = self._counter
        ctr_name = f"foldC{uid}"

        # --- Flatten carry if StructType (XLA while doesn't accept tuples) ---
        carry_is_struct = isinstance(init_type, StructType)
        if carry_is_struct:
            carry_field_names: list[str] = []
            carry_field_init_vals: list[str] = []
            carry_field_mlir_types: list[str] = []
            carry_field_types: list[MaomiType] = []
            for fi, (fname, ftype) in enumerate(init_type.fields):
                flat_name = f"foldK{uid}f{fi}"
                field_ssa = self._fresh()
                self._emit(
                    f"{field_ssa} = stablehlo.get_tuple_element {init_val}[{fi}] "
                    f": ({_mlir_type(init_type)}) -> {_mlir_type(ftype)}"
                )
                carry_field_names.append(flat_name)
                carry_field_init_vals.append(field_ssa)
                carry_field_mlir_types.append(_mlir_type(ftype))
                carry_field_types.append(ftype)
        else:
            carry_field_names = [f"foldK{uid}"]
            carry_field_init_vals = [init_val]
            carry_field_mlir_types = [_mlir_type(init_type)]
            carry_field_types = [init_type]

        # --- Flatten StructArrayType sequences ---
        flat_seq_names: list[str] = []
        flat_seq_init_vals: list[str] = []
        flat_seq_mlir_types: list[str] = []
        seq_field_map: list[list[tuple[str, MaomiType]]] = []

        for i, (sv, st) in enumerate(zip(seq_vals, seq_types)):
            if isinstance(st, StructArrayType):
                field_info = []
                for fi, (fname, ftype) in enumerate(st.struct_type.fields):
                    if isinstance(ftype, ScalarType):
                        batched = ArrayType(ftype.base, st.dims)
                    elif isinstance(ftype, ArrayType):
                        batched = ArrayType(ftype.base, st.dims + ftype.dims)
                    else:
                        raise MaomiError("codegen: unsupported struct array field type", "<codegen>", 0, 0)
                    flat_name = f"foldS{uid}_{i}f{fi}"
                    field_ssa = self._fresh()
                    self._emit(
                        f"{field_ssa} = stablehlo.get_tuple_element {sv}[{fi}] "
                        f": ({_mlir_type(st)}) -> {_mlir_type(batched)}"
                    )
                    flat_seq_names.append(flat_name)
                    flat_seq_init_vals.append(field_ssa)
                    flat_seq_mlir_types.append(_mlir_type(batched))
                    field_info.append((flat_name, batched))
                seq_field_map.append(field_info)
            else:
                flat_name = f"foldS{uid}_{i}"
                flat_seq_names.append(flat_name)
                flat_seq_init_vals.append(sv)
                flat_seq_mlir_types.append(_mlir_type(st))
                seq_field_map.append([(flat_name, st)])

        # --- Build while loop ---
        arg_names = [ctr_name] + carry_field_names + flat_seq_names
        init_vals_list = [counter_var] + carry_field_init_vals + flat_seq_init_vals
        arg_types = [mlir_i32] + carry_field_mlir_types + flat_seq_mlir_types
        n_args = len(arg_names)

        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals_list[i]}" for i in range(n_args)
        )
        types_str = ", ".join(arg_types)
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond region
        self._emit("cond {")
        self._indent += 1
        limit_v = self._fresh()
        self._emit(f"{limit_v} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare LT, %{ctr_name}, {limit_v}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        body_env = dict(env)

        # Reconstruct carry tuple if struct
        if carry_is_struct:
            carry_field_ssas = [f"%{n}" for n in carry_field_names]
            carry_types_str = ", ".join(carry_field_mlir_types)
            carry_tup = self._fresh()
            self._emit(f"{carry_tup} = stablehlo.tuple {', '.join(carry_field_ssas)} : tuple<{carry_types_str}>")
            body_env[expr.carry_var] = carry_tup
        else:
            body_env[expr.carry_var] = f"%{carry_field_names[0]}"

        # Slice sequence elements
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                if isinstance(st, StructArrayType):
                    field_info = seq_field_map[i]
                    field_ssas = [f"%{fn}" for fn, _ in field_info]
                    field_types_str = ", ".join(_mlir_type(bt) for _, bt in field_info)
                    tup = self._fresh()
                    self._emit(f"{tup} = stablehlo.tuple {', '.join(field_ssas)} : tuple<{field_types_str}>")
                    b_elem = self._slice_struct_array_element(tup, f"%{ctr_name}", st)
                else:
                    b_elem = self._slice_element(f"%{seq_field_map[i][0][0]}", f"%{ctr_name}", st, et, iter_axis=iter_axis)
                body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, body_env)

        # Destructure new carry if struct
        if carry_is_struct:
            new_carry_fields = []
            for fi, (fname, ftype) in enumerate(init_type.fields):
                field_ssa = self._fresh()
                self._emit(
                    f"{field_ssa} = stablehlo.get_tuple_element {new_carry}[{fi}] "
                    f": ({_mlir_type(init_type)}) -> {_mlir_type(ftype)}"
                )
                new_carry_fields.append(field_ssa)
        else:
            new_carry_fields = [new_carry]

        # Update counter
        one_v = self._fresh()
        self._emit(f"{one_v} = stablehlo.constant dense<1> : {mlir_i32}")
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.add %{ctr_name}, {one_v} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter] + new_carry_fields + [f"%{sn}" for sn in flat_seq_names]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Reconstruct carry from while result
        if carry_is_struct:
            result_fields = [f"{while_result}#{1 + fi}" for fi in range(len(init_type.fields))]
            fields_str = ", ".join(result_fields)
            carry_types_str = ", ".join(carry_field_mlir_types)
            result_tup = self._fresh()
            self._emit(f"{result_tup} = stablehlo.tuple {fields_str} : tuple<{carry_types_str}>")
            return result_tup
        else:
            return f"{while_result}#1"

    # -- Fold gradient codegen --

    def _gen_fold_grad(self, expr: _FoldGrad, env: dict[str, str]) -> str:
        """Emit augmented forward + reverse while loop for fold backward pass.

        Phase A: Run fold body forward, stacking carries into a trajectory buffer.
        Phase B: Build padded adjoint array (zeros with adj at last position).
        Phase C: Reverse while loop identical to _gen_scan_grad.
        """
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        adj_val = self._gen_expr(expr.adj, env)

        init_type = self._type_of(expr.init)
        seq_types = [self._type_of(s) for s in expr.sequences]
        adj_type = self._type_of(expr.adj)

        for st in seq_types:
            if not isinstance(st, (ArrayType, StructArrayType)):
                raise MaomiError("codegen: _FoldGrad sequence must be array", "<codegen>", 0, 0)

        first_st = seq_types[0]
        seq_len = first_st.dims[0] if isinstance(first_st, StructArrayType) else first_st.dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: _FoldGrad requires concrete length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            if isinstance(st, StructArrayType):
                if len(st.dims) == 1:
                    elem_types.append(st.struct_type)
                else:
                    elem_types.append(StructArrayType(st.struct_type, st.dims[1:]))
            elif len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))

        # Trajectory type: stack init_type along a new leading dim of seq_len
        if isinstance(init_type, ScalarType):
            traj_type = ArrayType(init_type.base, (seq_len,))
        elif isinstance(init_type, ArrayType):
            traj_type = ArrayType(init_type.base, (seq_len,) + init_type.dims)
        else:
            raise MaomiError("codegen: _FoldGrad init must be scalar or array", "<codegen>", 0, 0)

        n_seqs = len(expr.sequences)
        mlir_i32 = "tensor<i32>"
        mlir_init = _mlir_type(init_type)
        mlir_traj = _mlir_type(traj_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]

        # ---- Phase A: Augmented forward (build trajectory) ----

        fwd_counter = self._fresh()
        self._emit(f"{fwd_counter} = stablehlo.constant dense<0> : {mlir_i32}")

        fwd_traj_init = self._fresh()
        self._emit(f"{fwd_traj_init} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        uid_fwd = self._counter
        fwd_ctr_name = f"fgFC{uid_fwd}"
        fwd_carry_name = f"fgFK{uid_fwd}"
        fwd_traj_name = f"fgFT{uid_fwd}"
        fwd_seq_names = [f"fgFS{uid_fwd}_{i}" for i in range(n_seqs)]

        fwd_arg_names = [fwd_ctr_name, fwd_carry_name, fwd_traj_name] + fwd_seq_names
        fwd_init_vals = [fwd_counter, init_val, fwd_traj_init] + seq_vals
        fwd_arg_types = [mlir_i32, mlir_init, mlir_traj] + mlir_seqs
        fwd_n_args = len(fwd_arg_names)

        fwd_while = self._fresh()
        fwd_header = ", ".join(
            f"%{fwd_arg_names[i]} = {fwd_init_vals[i]}" for i in range(fwd_n_args)
        )
        fwd_types_str = ", ".join(fwd_arg_types)
        self._emit(f"{fwd_while}:{fwd_n_args} = stablehlo.while({fwd_header}) : {fwd_types_str}")

        # Cond: counter < seq_len
        self._emit("cond {")
        self._indent += 1
        fwd_limit = self._fresh()
        self._emit(f"{fwd_limit} = stablehlo.constant dense<{seq_len}> : {mlir_i32}")
        fwd_cmp = self._fresh()
        self._emit(f"{fwd_cmp} = stablehlo.compare LT, %{fwd_ctr_name}, {fwd_limit}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {fwd_cmp} : tensor<i1>")
        self._indent -= 1

        # Body: compute new_carry from fold body, stack into trajectory
        self._emit("} do {")
        self._indent += 1

        fwd_body_env = dict(env)
        fwd_body_env[expr.carry_var] = f"%{fwd_carry_name}"
        for i, (ev, st, et) in enumerate(zip(expr.elem_vars, seq_types, elem_types)):
            if _block_references_var(expr.body, ev):
                b_elem = self._slice_element(f"%{fwd_seq_names[i]}", f"%{fwd_ctr_name}", st, et)
                fwd_body_env[ev] = b_elem

        new_carry = self._gen_block(expr.body, fwd_body_env)

        # Stack new_carry into trajectory at counter position
        new_traj = self._update_element(f"%{fwd_traj_name}", new_carry, f"%{fwd_ctr_name}", traj_type, init_type)

        # Increment counter
        fwd_one = self._fresh()
        self._emit(f"{fwd_one} = stablehlo.constant dense<1> : {mlir_i32}")
        fwd_new_ctr = self._fresh()
        self._emit(f"{fwd_new_ctr} = stablehlo.add %{fwd_ctr_name}, {fwd_one} : {mlir_i32}")

        fwd_new_vals = [fwd_new_ctr, new_carry, new_traj] + [f"%{sn}" for sn in fwd_seq_names]
        fwd_vals_str = ", ".join(fwd_new_vals)
        self._emit(f"stablehlo.return {fwd_vals_str} : {fwd_types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract trajectory from augmented forward result (index 2)
        fwd_trajectory = f"{fwd_while}#2"

        # ---- Phase B: Padded adjoint array ----
        # Build adj_array = zeros(traj_type) with adj placed at position seq_len - 1

        adj_zeros = self._fresh()
        self._emit(f"{adj_zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_traj}")

        last_idx = self._fresh()
        self._emit(f"{last_idx} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        # Broadcast scalar adj to init_type if needed
        if isinstance(adj_type, ScalarType) and isinstance(init_type, ArrayType):
            adj_val = self._maybe_broadcast(adj_val, adj_type, init_type)

        adj_array = self._update_element(adj_zeros, adj_val, last_idx, traj_type, init_type)

        # ---- Phase C: Reverse while loop (same structure as _gen_scan_grad) ----

        rev_counter = self._fresh()
        self._emit(f"{rev_counter} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        adj_carry_init = self._fresh()
        self._emit(f"{adj_carry_init} = stablehlo.constant dense<0.000000e+00> : {mlir_init}")

        adj_seq_inits = []
        for ms in mlir_seqs:
            v = self._fresh()
            self._emit(f"{v} = stablehlo.constant dense<0.000000e+00> : {ms}")
            adj_seq_inits.append(v)

        uid_rev = self._counter
        gc_name = f"fgRC{uid_rev}"
        gac_name = f"fgRAC{uid_rev}"
        gas_names = [f"fgRAS{uid_rev}_{i}" for i in range(n_seqs)]
        gfwd_name = f"fgRFwd{uid_rev}"
        gos_names = [f"fgROS{uid_rev}_{i}" for i in range(n_seqs)]
        gadj_name = f"fgRAdj{uid_rev}"
        ginit_name = f"fgRInit{uid_rev}"

        arg_names = [gc_name, gac_name]
        init_vals = [rev_counter, adj_carry_init]
        arg_types_list = [mlir_i32, mlir_init]

        for i in range(n_seqs):
            arg_names.append(gas_names[i])
            init_vals.append(adj_seq_inits[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gfwd_name)
        init_vals.append(fwd_trajectory)
        arg_types_list.append(mlir_traj)

        for i in range(n_seqs):
            arg_names.append(gos_names[i])
            init_vals.append(seq_vals[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gadj_name)
        init_vals.append(adj_array)
        arg_types_list.append(mlir_traj)

        arg_names.append(ginit_name)
        init_vals.append(init_val)
        arg_types_list.append(mlir_init)

        n_args = len(arg_names)
        types_str = ", ".join(arg_types_list)

        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals[i]}" for i in range(n_args)
        )
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond: counter >= 0
        self._emit("cond {")
        self._indent += 1
        zero_c = self._fresh()
        self._emit(f"{zero_c} = stablehlo.constant dense<0> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare GE, %{gc_name}, {zero_c}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body
        self._emit("} do {")
        self._indent += 1

        # Slice adj_t from adj_array
        adj_t = self._slice_element(f"%{gadj_name}", f"%{gc_name}", traj_type, init_type)

        # adj_total = adj_carry + adj_t
        adj_total = self._fresh()
        self._emit(f"{adj_total} = stablehlo.add %{gac_name}, {adj_t} : {mlir_init}")

        # prev_carry = select(counter > 0, trajectory[counter-1], init)
        one_b = self._fresh()
        self._emit(f"{one_b} = stablehlo.constant dense<1> : {mlir_i32}")
        zero_b = self._fresh()
        self._emit(f"{zero_b} = stablehlo.constant dense<0> : {mlir_i32}")

        prev_idx = self._fresh()
        self._emit(f"{prev_idx} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")
        clamped_idx = self._fresh()
        self._emit(f"{clamped_idx} = stablehlo.clamp {zero_b}, {prev_idx}, %{gc_name} : ({mlir_i32}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}")

        fwd_prev = self._slice_element(f"%{gfwd_name}", clamped_idx, traj_type, init_type)

        gt_zero = self._fresh()
        self._emit(f"{gt_zero} = stablehlo.compare GT, %{gc_name}, {zero_b}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")

        prev_carry = self._fresh()
        self._emit(f"{prev_carry} = stablehlo.select {gt_zero}, {fwd_prev}, %{ginit_name} : (tensor<i1>, {mlir_init}, {mlir_init}) -> {mlir_init}")

        # Slice elements from original sequences
        b_elems = []
        for i in range(n_seqs):
            elem = self._slice_element(f"%{gos_names[i]}", f"%{gc_name}", seq_types[i], elem_types[i])
            b_elems.append(elem)

        # Evaluate derivative expressions
        deriv_env = dict(env)
        deriv_env[expr.carry_var] = prev_carry
        for ev, elem_ssa in zip(expr.elem_vars, b_elems):
            deriv_env[ev] = elem_ssa

        d_carry_val = self._gen_expr(expr.d_body_d_carry, deriv_env)

        # new_adj_carry = adj_total * d_carry_val
        new_adj_carry = self._fresh()
        self._emit(f"{new_adj_carry} = stablehlo.multiply {adj_total}, {d_carry_val} : {mlir_init}")

        # For each sequence, compute adj_elem and accumulate
        new_adj_seqs = []
        for i in range(n_seqs):
            d_elem_val = self._gen_expr(expr.d_body_d_elems[i], deriv_env)
            adj_elem = self._fresh()
            mlir_et = _mlir_type(elem_types[i])
            self._emit(f"{adj_elem} = stablehlo.multiply {adj_total}, {d_elem_val} : {mlir_et}")
            new_adj_seq = self._update_element(f"%{gas_names[i]}", adj_elem, f"%{gc_name}", seq_types[i], elem_types[i])
            new_adj_seqs.append(new_adj_seq)

        # Decrement counter
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_adj_carry] + new_adj_seqs
        new_vals += [f"%{gfwd_name}"] + [f"%{gos_names[i]}" for i in range(n_seqs)]
        new_vals += [f"%{gadj_name}", f"%{ginit_name}"]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract result based on wrt
        if expr.wrt == "__init__":
            return f"{while_result}#1"  # adj_carry
        else:
            for i, seq_expr in enumerate(expr.sequences):
                if isinstance(seq_expr, Identifier) and seq_expr.name == expr.wrt:
                    return f"{while_result}#{2 + i}"  # adj_seq_i
            return f"{while_result}#2"  # fallback: first adj_seq

    # -- Scan gradient codegen --

    def _gen_scan_grad(self, expr: _ScanGrad, env: dict[str, str]) -> str:
        """Emit a reverse while loop for the backward pass of scan."""
        fwd_result = self._gen_expr(expr.forward_result, env)
        init_val = self._gen_expr(expr.init, env)
        seq_vals = [self._gen_expr(s, env) for s in expr.sequences]
        adj_val = self._gen_expr(expr.adj, env)

        init_type = self._type_of(expr.init)
        fwd_type = self._type_of(expr.forward_result)
        adj_type = self._type_of(expr.adj)
        seq_types = [self._type_of(s) for s in expr.sequences]

        if not isinstance(fwd_type, ArrayType):
            raise MaomiError("codegen: _ScanGrad forward_result must be array", "<codegen>", 0, 0)

        # Broadcast scalar adj to array if needed
        if isinstance(adj_type, ScalarType) and isinstance(fwd_type, ArrayType):
            adj_val = self._maybe_broadcast(adj_val, adj_type, fwd_type)
            adj_type = fwd_type

        seq_len = fwd_type.dims[0]
        if isinstance(seq_len, str):
            raise MaomiError(f"codegen: _ScanGrad requires concrete length, got '{seq_len}'", "<codegen>", 0, 0)

        elem_types: list[MaomiType] = []
        for st in seq_types:
            if not isinstance(st, ArrayType):
                raise MaomiError("codegen: _ScanGrad sequence must be array", "<codegen>", 0, 0)
            if len(st.dims) == 1:
                elem_types.append(ScalarType(st.base))
            else:
                elem_types.append(ArrayType(st.base, st.dims[1:]))

        n_seqs = len(expr.sequences)
        mlir_i32 = "tensor<i32>"
        mlir_init = _mlir_type(init_type)
        mlir_fwd = _mlir_type(fwd_type)
        mlir_adj = _mlir_type(adj_type)
        mlir_seqs = [_mlir_type(st) for st in seq_types]

        # Initialize outside the while
        counter_var = self._fresh()
        self._emit(f"{counter_var} = stablehlo.constant dense<{seq_len - 1}> : {mlir_i32}")

        adj_carry_init = self._fresh()
        self._emit(f"{adj_carry_init} = stablehlo.constant dense<0.000000e+00> : {mlir_init}")

        adj_seq_inits = []
        for ms in mlir_seqs:
            v = self._fresh()
            self._emit(f"{v} = stablehlo.constant dense<0.000000e+00> : {ms}")
            adj_seq_inits.append(v)

        # Build while arg names and types
        # State: counter, adj_carry, adj_seq_0..N-1, fwd_carries, seq_0..N-1, adj_array, init_val
        uid = self._counter
        gc_name = f"gC{uid}"
        gac_name = f"gAC{uid}"
        gas_names = [f"gAS{uid}_{i}" for i in range(n_seqs)]
        gfwd_name = f"gFwd{uid}"
        gos_names = [f"gOS{uid}_{i}" for i in range(n_seqs)]
        gadj_name = f"gAdj{uid}"
        ginit_name = f"gInit{uid}"

        arg_names = [gc_name, gac_name]
        init_vals = [counter_var, adj_carry_init]
        arg_types_list = [mlir_i32, mlir_init]

        for i in range(n_seqs):
            arg_names.append(gas_names[i])
            init_vals.append(adj_seq_inits[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gfwd_name)
        init_vals.append(fwd_result)
        arg_types_list.append(mlir_fwd)

        for i in range(n_seqs):
            arg_names.append(gos_names[i])
            init_vals.append(seq_vals[i])
            arg_types_list.append(mlir_seqs[i])

        arg_names.append(gadj_name)
        init_vals.append(adj_val)
        arg_types_list.append(mlir_adj)

        arg_names.append(ginit_name)
        init_vals.append(init_val)
        arg_types_list.append(mlir_init)

        n_args = len(arg_names)
        types_str = ", ".join(arg_types_list)

        # Emit custom while format
        while_result = self._fresh()
        header_parts = ", ".join(
            f"%{arg_names[i]} = {init_vals[i]}" for i in range(n_args)
        )
        self._emit(f"{while_result}:{n_args} = stablehlo.while({header_parts}) : {types_str}")

        # Cond region: counter >= 0
        self._emit("cond {")
        self._indent += 1
        zero_c = self._fresh()
        self._emit(f"{zero_c} = stablehlo.constant dense<0> : {mlir_i32}")
        c_cmp = self._fresh()
        self._emit(f"{c_cmp} = stablehlo.compare GE, %{gc_name}, {zero_c}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")
        self._emit(f"stablehlo.return {c_cmp} : tensor<i1>")
        self._indent -= 1

        # Body region
        self._emit("} do {")
        self._indent += 1

        # Slice adj_t from adj_array
        adj_t = self._slice_element(f"%{gadj_name}", f"%{gc_name}", adj_type, init_type)

        # adj_total = adj_carry + adj_t
        adj_total = self._fresh()
        self._emit(f"{adj_total} = stablehlo.add %{gac_name}, {adj_t} : {mlir_init}")

        # Compute prev_carry = select(counter > 0, fwd_carries[counter-1], init)
        one_b = self._fresh()
        self._emit(f"{one_b} = stablehlo.constant dense<1> : {mlir_i32}")
        zero_b = self._fresh()
        self._emit(f"{zero_b} = stablehlo.constant dense<0> : {mlir_i32}")

        prev_idx = self._fresh()
        self._emit(f"{prev_idx} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")
        clamped_idx = self._fresh()
        self._emit(f"{clamped_idx} = stablehlo.clamp {zero_b}, {prev_idx}, %{gc_name} : ({mlir_i32}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}")

        fwd_prev = self._slice_element(f"%{gfwd_name}", clamped_idx, fwd_type, init_type)

        gt_zero = self._fresh()
        self._emit(f"{gt_zero} = stablehlo.compare GT, %{gc_name}, {zero_b}, SIGNED : ({mlir_i32}, {mlir_i32}) -> tensor<i1>")

        prev_carry = self._fresh()
        self._emit(f"{prev_carry} = stablehlo.select {gt_zero}, {fwd_prev}, %{ginit_name} : (tensor<i1>, {mlir_init}, {mlir_init}) -> {mlir_init}")

        # Slice elements from original sequences
        b_elems = []
        for i in range(n_seqs):
            elem = self._slice_element(f"%{gos_names[i]}", f"%{gc_name}", seq_types[i], elem_types[i])
            b_elems.append(elem)

        # Evaluate derivative expressions with substituted values
        deriv_env = dict(env)
        deriv_env[expr.carry_var] = prev_carry
        for ev, elem_ssa in zip(expr.elem_vars, b_elems):
            deriv_env[ev] = elem_ssa

        d_carry_val = self._gen_expr(expr.d_body_d_carry, deriv_env)

        # new_adj_carry = adj_total * d_carry_val
        new_adj_carry = self._fresh()
        self._emit(f"{new_adj_carry} = stablehlo.multiply {adj_total}, {d_carry_val} : {mlir_init}")

        # For each sequence, compute adj_elem and accumulate
        new_adj_seqs = []
        for i in range(n_seqs):
            d_elem_val = self._gen_expr(expr.d_body_d_elems[i], deriv_env)
            adj_elem = self._fresh()
            mlir_et = _mlir_type(elem_types[i])
            self._emit(f"{adj_elem} = stablehlo.multiply {adj_total}, {d_elem_val} : {mlir_et}")
            new_adj_seq = self._update_element(f"%{gas_names[i]}", adj_elem, f"%{gc_name}", seq_types[i], elem_types[i])
            new_adj_seqs.append(new_adj_seq)

        # Decrement counter
        new_counter = self._fresh()
        self._emit(f"{new_counter} = stablehlo.subtract %{gc_name}, {one_b} : {mlir_i32}")

        # Return new values
        new_vals = [new_counter, new_adj_carry] + new_adj_seqs
        new_vals += [f"%{gfwd_name}"] + [f"%{gos_names[i]}" for i in range(n_seqs)]
        new_vals += [f"%{gadj_name}", f"%{ginit_name}"]
        vals_str = ", ".join(new_vals)
        self._emit(f"stablehlo.return {vals_str} : {types_str}")
        self._indent -= 1
        self._emit("}")

        # Extract result based on wrt
        if expr.wrt == "__init__":
            return f"{while_result}#1"  # adj_carry
        else:
            for i, seq_expr in enumerate(expr.sequences):
                if isinstance(seq_expr, Identifier) and seq_expr.name == expr.wrt:
                    return f"{while_result}#{2 + i}"  # adj_seq_i
            return f"{while_result}#2"  # fallback: first adj_seq
