from __future__ import annotations

from ...ast_nodes import (
    IntLiteral,
    IndexExpr,
    _IndexGrad,
    _GatherGrad,
)
from ...types import MaomiType, ScalarType, ArrayType
from ...errors import MaomiError
from .utils import _mlir_type, _MLIR_ETYPE


class IndexingCodegenMixin:

    def _normalize_index(self, idx_ssa: str, dim: int) -> str:
        """Emit runtime normalization for potentially negative index: select(i < 0, i + dim, i)."""
        mlir_i32 = "tensor<i32>"
        mlir_i1 = "tensor<i1>"
        zero = self._gen_literal(0, ScalarType("i32"))
        dim_ssa = self._gen_literal(dim, ScalarType("i32"))
        is_neg = self._fresh()
        self._emit(
            f"{is_neg} = stablehlo.compare LT, {idx_ssa}, {zero}, SIGNED "
            f": ({mlir_i32}, {mlir_i32}) -> {mlir_i1}"
        )
        wrapped = self._fresh()
        self._emit(f"{wrapped} = stablehlo.add {idx_ssa}, {dim_ssa} : {mlir_i32}")
        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.select {is_neg}, {wrapped}, {idx_ssa} "
            f": ({mlir_i1}, {mlir_i32}, {mlir_i32}) -> {mlir_i32}"
        )
        return result

    def _gen_index(self, expr: IndexExpr, env: dict[str, str]) -> str:
        base_type = self._type_of(expr.base)

        if not isinstance(base_type, ArrayType):
            raise MaomiError("codegen: indexing non-array", "<codegen>", expr.span.line_start, expr.span.col_start)

        # Check for array-based indexing (gather) — dispatch before scalar path
        for ic in expr.indices:
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    return self._gen_gather(expr, env)

        base_ssa = self._gen_expr(expr.base, env)
        result_type = self._type_of(expr)

        # Build per-dimension start indices and slice sizes
        start_ssas: list[str] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []  # axes to remove (single-indexed)
        all_static = True

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_ssa = self._gen_expr(ic.value, env)
                # Runtime normalization for dynamic indices (may be negative)
                if not isinstance(ic.value, IntLiteral):
                    idx_ssa = self._normalize_index(idx_ssa, dim)
                start_ssas.append(idx_ssa)
                slice_sizes.append(1)
                squeezed_axes.append(i)
                if not isinstance(ic.value, IntLiteral):
                    all_static = False
            elif ic.kind == "slice":
                if isinstance(ic.start, IntLiteral):
                    s = self._gen_literal(ic.start.value, ScalarType("i32"))
                else:
                    s = self._gen_expr(ic.start, env)
                    s = self._normalize_index(s, dim)
                    all_static = False
                start_ssas.append(s)
                slice_sizes.append(ic.static_size)
            elif ic.kind == "full":
                s = self._gen_literal(0, ScalarType("i32"))
                start_ssas.append(s)
                if isinstance(dim, int):
                    slice_sizes.append(dim)
                else:
                    raise MaomiError(f"codegen: symbolic dim '{dim}' in index", "<codegen>", 0, 0)

        # Trailing unindexed axes
        for i in range(len(expr.indices), len(base_type.dims)):
            dim = base_type.dims[i]
            s = self._gen_literal(0, ScalarType("i32"))
            start_ssas.append(s)
            if isinstance(dim, int):
                slice_sizes.append(dim)
            else:
                raise MaomiError(f"codegen: symbolic dim '{dim}' in index", "<codegen>", 0, 0)

        # Intermediate type: same rank as base, but with size-1 for single-indexed dims
        inter_dims = tuple(slice_sizes)
        inter_type = ArrayType(base_type.base, inter_dims)
        mlir_base = _mlir_type(base_type)
        mlir_inter = _mlir_type(inter_type)

        # Check if all starts are static IntLiterals → use stablehlo.slice
        if all_static and all(isinstance(ic.value, IntLiteral) for ic in expr.indices if ic.kind == "single"):
            # Build static slice: stablehlo.slice %x [start:start+size:1, ...]
            starts = []
            limits = []
            for i in range(len(base_type.dims)):
                if i < len(expr.indices):
                    ic = expr.indices[i]
                    if ic.kind == "single":
                        starts.append(ic.value.value)
                        limits.append(ic.value.value + 1)
                    elif ic.kind == "slice":
                        starts.append(ic.start.value)
                        limits.append(ic.start.value + ic.static_size)
                    else:  # full
                        starts.append(0)
                        limits.append(base_type.dims[i])
                else:
                    starts.append(0)
                    limits.append(base_type.dims[i])

            strides = [1] * len(base_type.dims)
            # Custom assembly: [start:limit:stride, ...] or [start:limit, ...] when stride=1
            parts = []
            for s, l, st in zip(starts, limits, strides):
                if st == 1:
                    parts.append(f"{s}:{l}")
                else:
                    parts.append(f"{s}:{l}:{st}")
            slice_spec = ", ".join(parts)

            sliced = self._fresh()
            self._emit(
                f"{sliced} = stablehlo.slice {base_ssa} "
                f"[{slice_spec}] "
                f": ({mlir_base}) -> {mlir_inter}"
            )
        else:
            # Dynamic path: stablehlo.dynamic_slice
            sizes_str = ", ".join(str(s) for s in slice_sizes)
            starts_str = ", ".join(start_ssas)
            start_types = ", ".join("tensor<i32>" for _ in start_ssas)

            sliced = self._fresh()
            self._emit(
                f"{sliced} = stablehlo.dynamic_slice {base_ssa}, {starts_str}, "
                f"sizes = [{sizes_str}] "
                f": ({mlir_base}, {start_types}) -> {mlir_inter}"
            )

        # Reshape to remove squeezed dimensions
        if squeezed_axes:
            result = self._fresh()
            self._emit(f"{result} = stablehlo.reshape {sliced} : ({mlir_inter}) -> {_mlir_type(result_type)}")
            return result
        return sliced

    def _gen_index_grad(self, expr: _IndexGrad, env: dict[str, str]) -> str:
        """Emit backward pass for indexing: zeros + dynamic_update_slice."""
        base_type = self._type_of(expr.base_expr)
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)

        if not isinstance(base_type, ArrayType):
            raise MaomiError("codegen: _IndexGrad base must be array", "<codegen>", 0, 0)

        mlir_base = _mlir_type(base_type)

        # Create zero tensor of base shape
        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_base}")

        # Build start indices (same logic as forward)
        start_ssas: list[str] = []
        slice_sizes: list[int] = []
        squeezed_axes: list[int] = []

        for i, ic in enumerate(expr.indices):
            dim = base_type.dims[i]
            if ic.kind == "single":
                idx_ssa = self._gen_expr(ic.value, env)
                # Runtime normalization for dynamic indices (may be negative)
                if not isinstance(ic.value, IntLiteral):
                    idx_ssa = self._normalize_index(idx_ssa, dim)
                start_ssas.append(idx_ssa)
                slice_sizes.append(1)
                squeezed_axes.append(i)
            elif ic.kind == "slice":
                if isinstance(ic.start, IntLiteral):
                    s = self._gen_literal(ic.start.value, ScalarType("i32"))
                else:
                    s = self._gen_expr(ic.start, env)
                    s = self._normalize_index(s, dim)
                start_ssas.append(s)
                slice_sizes.append(ic.static_size)
            elif ic.kind == "full":
                s = self._gen_literal(0, ScalarType("i32"))
                start_ssas.append(s)
                slice_sizes.append(dim)

        for i in range(len(expr.indices), len(base_type.dims)):
            s = self._gen_literal(0, ScalarType("i32"))
            start_ssas.append(s)
            slice_sizes.append(base_type.dims[i])

        # Reshape adj to re-insert squeezed dims
        update_dims = tuple(slice_sizes)
        update_type = ArrayType(base_type.base, update_dims)
        mlir_update = _mlir_type(update_type)

        if squeezed_axes:
            reshaped = self._fresh()
            self._emit(f"{reshaped} = stablehlo.reshape {adj_ssa} : ({_mlir_type(adj_type)}) -> {mlir_update}")
            adj_ssa = reshaped

        # Emit dynamic_update_slice
        starts_str = ", ".join(start_ssas)
        start_types = ", ".join("tensor<i32>" for _ in start_ssas)

        result = self._fresh()
        self._emit(
            f"{result} = stablehlo.dynamic_update_slice {zeros}, {adj_ssa}, {starts_str} "
            f": ({mlir_base}, {mlir_update}, {start_types}) -> {mlir_base}"
        )
        return result

    # -- Gather / Scatter codegen --

    def _gen_gather(self, expr: IndexExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.gather for array-based indexing (e.g., table[ids])."""
        base_ssa = self._gen_expr(expr.base, env)
        base_type = self._type_of(expr.base)
        result_type = self._type_of(expr)
        rank = len(base_type.dims)

        # Find the array-indexed axis
        gather_axis = 0
        indices_expr = None
        for i, ic in enumerate(expr.indices):
            if ic.kind == "single":
                idx_type = self._type_of(ic.value)
                if isinstance(idx_type, ArrayType):
                    gather_axis = i
                    indices_expr = ic.value
                    break

        indices_ssa = self._gen_expr(indices_expr, env)
        indices_type = self._type_of(indices_expr)
        B = indices_type.dims[0]

        # Reshape indices: i32[B] → i32[B, 1] (add index_vector_dim)
        reshaped_type = ArrayType(indices_type.base, (B, 1))
        reshaped_indices = self._fresh()
        self._emit(
            f"{reshaped_indices} = stablehlo.reshape {indices_ssa} "
            f": ({_mlir_type(indices_type)}) -> {_mlir_type(reshaped_type)}"
        )

        # Dimension numbers
        offset_dims = list(range(0, gather_axis)) + list(range(gather_axis + 1, rank))
        collapsed_slice_dims = [gather_axis]
        start_index_map = [gather_axis]

        # Slice sizes: 1 for gathered axis, full for others
        slice_sizes = [d for d in base_type.dims]
        slice_sizes[gather_axis] = 1

        # Format dimension number lists
        od_str = ", ".join(str(d) for d in offset_dims)
        cd_str = ", ".join(str(d) for d in collapsed_slice_dims)
        sim_str = ", ".join(str(d) for d in start_index_map)
        ss_str = ", ".join(str(s) for s in slice_sizes)

        # Build gather dimension_numbers attribute — omit offset_dims if empty
        dnums_parts = []
        if offset_dims:
            dnums_parts.append(f"offset_dims = [{od_str}]")
        dnums_parts.append(f"collapsed_slice_dims = [{cd_str}]")
        dnums_parts.append(f"start_index_map = [{sim_str}]")
        dnums_parts.append("index_vector_dim = 1")
        dnums_str = ", ".join(dnums_parts)

        result = self._fresh()
        self._emit(
            f'{result} = "stablehlo.gather"({base_ssa}, {reshaped_indices}) '
            f'<{{dimension_numbers = #stablehlo.gather<{dnums_str}>, '
            f'indices_are_sorted = false, '
            f'slice_sizes = array<i64: {ss_str}>}}> '
            f': ({_mlir_type(base_type)}, {_mlir_type(reshaped_type)}) -> {_mlir_type(result_type)}'
        )
        return result

    def _gen_gather_grad(self, expr: _GatherGrad, env: dict[str, str]) -> str:
        """Emit stablehlo.scatter (add-combiner) for gather gradient."""
        base_type = self._type_of(expr.base_expr)
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        indices_ssa = self._gen_expr(expr.indices, env)
        indices_type = self._type_of(expr.indices)
        k = expr.gather_axis
        rank = len(base_type.dims)
        B = indices_type.dims[0]

        # Expected updates shape = gather output shape (base with dims[k] → B)
        updates_dims = list(base_type.dims)
        updates_dims[k] = B
        updates_type = ArrayType(base_type.base, tuple(updates_dims))

        # Broadcast adjoint if it's scalar or lower rank than expected
        if adj_type != updates_type:
            broadcast_dims: list[int] = []
            if isinstance(adj_type, ArrayType):
                # Map existing dims to output dims
                broadcast_dims = list(range(len(adj_type.dims)))
            broadcast_str = ", ".join(str(d) for d in broadcast_dims)
            broadcasted = self._fresh()
            self._emit(
                f"{broadcasted} = stablehlo.broadcast_in_dim {adj_ssa}, dims = [{broadcast_str}] "
                f": ({_mlir_type(adj_type)}) -> {_mlir_type(updates_type)}"
            )
            adj_ssa = broadcasted
            adj_type = updates_type

        mlir_base = _mlir_type(base_type)

        # Create zero tensor of base shape
        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0.000000e+00> : {mlir_base}")

        # Reshape indices: i32[B] → i32[B, 1]
        B = indices_type.dims[0]
        reshaped_idx_type = ArrayType(indices_type.base, (B, 1))
        reshaped_indices = self._fresh()
        self._emit(
            f"{reshaped_indices} = stablehlo.reshape {indices_ssa} "
            f": ({_mlir_type(indices_type)}) -> {_mlir_type(reshaped_idx_type)}"
        )

        # Scatter dimension numbers (mirror of gather)
        update_window_dims = list(range(0, k)) + list(range(k + 1, rank))
        inserted_window_dims = [k]
        scatter_dims_to_operand_dims = [k]

        uwd_str = ", ".join(str(d) for d in update_window_dims)
        iwd_str = ", ".join(str(d) for d in inserted_window_dims)
        sdtod_str = ", ".join(str(d) for d in scatter_dims_to_operand_dims)

        # Build scatter dimension_numbers attribute — omit update_window_dims if empty
        sdnums_parts = []
        if update_window_dims:
            sdnums_parts.append(f"update_window_dims = [{uwd_str}]")
        sdnums_parts.append(f"inserted_window_dims = [{iwd_str}]")
        sdnums_parts.append(f"scatter_dims_to_operand_dims = [{sdtod_str}]")
        sdnums_parts.append("index_vector_dim = 1")
        sdnums_str = ", ".join(sdnums_parts)

        # Element type for combiner region — use fresh names to avoid SSA conflicts
        etype = _MLIR_ETYPE[base_type.base]
        lhs = self._fresh()
        rhs = self._fresh()
        add_var = self._fresh()

        result = self._fresh()
        self._emit(
            f'{result} = "stablehlo.scatter"({zeros}, {reshaped_indices}, {adj_ssa}) '
            f'<{{scatter_dimension_numbers = #stablehlo.scatter<{sdnums_str}>, '
            f'indices_are_sorted = false, unique_indices = false}}> '
            f'({{')
        self._emit(f'  ^bb0({lhs}: tensor<{etype}>, {rhs}: tensor<{etype}>):')
        self._emit(f'    {add_var} = stablehlo.add {lhs}, {rhs} : tensor<{etype}>')
        self._emit(f'    stablehlo.return {add_var} : tensor<{etype}>')
        self._emit(
            f'}}) : ({mlir_base}, {_mlir_type(reshaped_idx_type)}, {_mlir_type(adj_type)}) -> {mlir_base}'
        )
        return result
