from __future__ import annotations

from ..ast_nodes import (
    IntLiteral,
    BoolLiteral,
    BinOp,
    CallExpr,
    CastExpr,
    IndexExpr,
    IndexComponent,
    _BroadcastExpr,
    Expr,
)
from ..types import MaomiType, ScalarType, ArrayType
from .constants import (
    _DUMMY_SPAN,
    _EW_REGISTRY,
)


class SimpleGradRulesMixin:
    def _backprop_elementwise(self, callee: str, args: list[Expr], adj: Expr,
                               adjoints: dict[str, Expr], var_map: dict[int, str],
                               node: Expr):
        descriptor = _EW_REGISTRY[callee]
        arg = args[0]
        arg_name = var_map[id(arg)]
        arg_ref = self._make_ref(arg_name, self._type_of(arg))
        grad_expr = descriptor.grad_rule(self, arg_ref, adj)
        self._accumulate(adjoints, arg_name, grad_expr)

    def _backprop_reduction(self, callee: str, args: list[Expr], adj: Expr,
                             adjoints: dict[str, Expr], var_map: dict[int, str],
                             node: Expr):
        arg = args[0]
        arg_name = var_map[id(arg)]
        arg_type = self._type_of(arg)

        has_axis = len(args) >= 2
        axis = args[1].value if has_axis else None
        keepdims = (len(args) == 3
                    and isinstance(args[2], BoolLiteral)
                    and args[2].value)

        if callee == "mean":
            if isinstance(arg_type, ArrayType):
                if has_axis:
                    # Axis-specific mean: adj / axis_size, broadcast back
                    axis_size = arg_type.dims[axis]
                    if isinstance(axis_size, str):
                        from ..errors import MaomiError
                        raise MaomiError(f"grad: cannot differentiate mean with symbolic dim '{axis_size}'", "<ad>", 0, 0)
                    scaled = self._make_binop("/", adj, self._make_float(float(axis_size)))
                    if keepdims:
                        # adj already has same rank with size-1 dim; size-1 broadcasting handles it
                        broadcast = _BroadcastExpr(scaled, tuple(arg_type.dims), _DUMMY_SPAN)
                    else:
                        ndim = len(arg_type.dims)
                        broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                        broadcast = _BroadcastExpr(scaled, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
                else:
                    # All-dims mean: broadcast(dz / numel)
                    numel = 1
                    for d in arg_type.dims:
                        if isinstance(d, int):
                            numel *= d
                        else:
                            from ..errors import MaomiError
                            raise MaomiError(f"grad: cannot differentiate mean with symbolic dim '{d}'", "<ad>", 0, 0)
                    scaled = self._make_binop("/", adj, self._make_float(float(numel)))
                    broadcast = _BroadcastExpr(scaled, tuple(arg_type.dims), _DUMMY_SPAN)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
            else:
                self._accumulate(adjoints, arg_name, adj)

        elif callee == "sum":
            if isinstance(arg_type, ArrayType):
                if has_axis:
                    if keepdims:
                        # adj already has same rank with size-1 dim; size-1 broadcasting handles it
                        broadcast = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                    else:
                        # Axis-specific sum: broadcast adj back by inserting reduced dim
                        ndim = len(arg_type.dims)
                        broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                        broadcast = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
                else:
                    # All-dims sum: broadcast scalar adj to input shape
                    broadcast = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                    self.type_map[id(broadcast)] = arg_type
                    self._accumulate(adjoints, arg_name, broadcast)
            else:
                self._accumulate(adjoints, arg_name, adj)

        elif callee in ("max", "min"):
            if not isinstance(arg_type, ArrayType):
                self._accumulate(adjoints, arg_name, adj)
                return

            # JAX indicator rule: grad = adj_bc * indicators / counts_bc
            # indicators = cast(operand == broadcast(result), f32)
            # counts = sum(indicators, axes)
            tape_name = var_map.get(id(node))
            node_ref = self._make_ref(tape_name, self._type_of(node)) if tape_name else node
            arg_ref = self._make_ref(arg_name, arg_type)
            ndim = len(arg_type.dims)

            # Broadcast result back to input shape
            if has_axis:
                if keepdims:
                    result_bc = _BroadcastExpr(node_ref, tuple(arg_type.dims), _DUMMY_SPAN)
                else:
                    broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                    result_bc = _BroadcastExpr(node_ref, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
            else:
                result_bc = _BroadcastExpr(node_ref, tuple(arg_type.dims), _DUMMY_SPAN)
            self.type_map[id(result_bc)] = arg_type

            # indicators = cast(arg == result_bc, arg_type.base)
            eq = BinOp("==", arg_ref, result_bc, _DUMMY_SPAN)
            bool_type = ArrayType("bool", arg_type.dims)
            self.type_map[id(eq)] = bool_type
            indicators = CastExpr(eq, arg_type.base, _DUMMY_SPAN)
            self.type_map[id(indicators)] = arg_type

            # counts = sum(indicators) or sum(indicators, axis)
            if has_axis:
                axis_lit = IntLiteral(axis, _DUMMY_SPAN)
                self.type_map[id(axis_lit)] = ScalarType("i32")
                counts = CallExpr("sum", [indicators, axis_lit], _DUMMY_SPAN)
                # count_type: without keepdims, same as reduced result
                reduced_dims = tuple(d for i, d in enumerate(arg_type.dims) if i != axis)
                count_type = ArrayType(arg_type.base, reduced_dims) if reduced_dims else ScalarType(arg_type.base)
            else:
                counts = CallExpr("sum", [indicators], _DUMMY_SPAN)
                count_type = ScalarType(arg_type.base)
            self.type_map[id(counts)] = count_type

            # Broadcast adj and counts back to input shape
            if has_axis:
                if keepdims:
                    adj_bc = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                    counts_bc = _BroadcastExpr(counts, tuple(arg_type.dims), _DUMMY_SPAN,
                                               broadcast_dims=tuple(i for i in range(ndim) if i != axis))
                else:
                    broadcast_dims = tuple(i for i in range(ndim) if i != axis)
                    adj_bc = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
                    counts_bc = _BroadcastExpr(counts, tuple(arg_type.dims), _DUMMY_SPAN, broadcast_dims=broadcast_dims)
            else:
                adj_bc = _BroadcastExpr(adj, tuple(arg_type.dims), _DUMMY_SPAN)
                counts_bc = _BroadcastExpr(counts, tuple(arg_type.dims), _DUMMY_SPAN)
            self.type_map[id(adj_bc)] = arg_type
            self.type_map[id(counts_bc)] = arg_type

            # grad = adj_bc * indicators / counts_bc
            grad = BinOp("*", adj_bc, indicators, _DUMMY_SPAN)
            self.type_map[id(grad)] = arg_type
            grad = BinOp("/", grad, counts_bc, _DUMMY_SPAN)
            self.type_map[id(grad)] = arg_type
            self._accumulate(adjoints, arg_name, grad)

    def _backprop_where(self, args: list[Expr], adj: Expr,
                         adjoints: dict[str, Expr], var_map: dict[int, str],
                         node: Expr):
        """Backprop through where(cond, x, y): adj_x = where(cond, adj, 0), adj_y = where(cond, 0, adj)."""
        cond = args[0]
        x_arg = args[1]
        y_arg = args[2]

        # cond has no gradient (boolean)
        # For x: gradient flows where cond is true
        if id(x_arg) in var_map:
            x_name = var_map[id(x_arg)]
            x_type = self._type_of(x_arg)
            zero = self._make_float(0.0)
            if isinstance(x_type, ArrayType):
                zero_broadcast = _BroadcastExpr(zero, tuple(x_type.dims), _DUMMY_SPAN)
                self.type_map[id(zero_broadcast)] = x_type
                adj_x = self._make_call("where", [cond, adj, zero_broadcast])
            else:
                adj_x = self._make_call("where", [cond, adj, zero])
            self.type_map[id(adj_x)] = self._type_of(x_arg)
            self._accumulate(adjoints, x_name, adj_x)

        # For y: gradient flows where cond is false
        if id(y_arg) in var_map:
            y_name = var_map[id(y_arg)]
            y_type = self._type_of(y_arg)
            zero = self._make_float(0.0)
            if isinstance(y_type, ArrayType):
                zero_broadcast = _BroadcastExpr(zero, tuple(y_type.dims), _DUMMY_SPAN)
                self.type_map[id(zero_broadcast)] = y_type
                adj_y = self._make_call("where", [cond, zero_broadcast, adj])
            else:
                adj_y = self._make_call("where", [cond, zero, adj])
            self.type_map[id(adj_y)] = self._type_of(y_arg)
            self._accumulate(adjoints, y_name, adj_y)

    def _backprop_reshape(self, args: list[Expr], adj: Expr,
                           adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through reshape: reshape adjoint back to original shape."""
        arg = args[0]
        if id(arg) not in var_map:
            return
        arg_name = var_map[id(arg)]
        arg_type = self._type_of(arg)
        if not isinstance(arg_type, ArrayType):
            self._accumulate(adjoints, arg_name, adj)
            return

        # Build reshape(adj, *original_dims)
        dim_literals = []
        for d in arg_type.dims:
            lit = IntLiteral(d, _DUMMY_SPAN)
            self.type_map[id(lit)] = ScalarType("i32")
            dim_literals.append(lit)
        reshape_call = CallExpr("reshape", [adj] + dim_literals, _DUMMY_SPAN)
        self.type_map[id(reshape_call)] = arg_type
        self._accumulate(adjoints, arg_name, reshape_call)

    def _backprop_concat(self, args: list[Expr], adj: Expr,
                          adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through concat: slice adjoint into pieces for each input."""
        # Detect axis
        if (isinstance(args[-1], IntLiteral)
                and isinstance(self.type_map.get(id(args[-1])), ScalarType)):
            axis = args[-1].value
            array_args = args[:-1]
        else:
            axis = 0
            array_args = args

        adj_type = self._type_of(adj)

        # If adj is scalar (e.g. from sum backprop), broadcast handles it —
        # just accumulate the scalar adj to each input (codegen broadcasts).
        if not isinstance(adj_type, ArrayType):
            for arg in array_args:
                if id(arg) in var_map:
                    self._accumulate(adjoints, var_map[id(arg)], adj)
            return

        rank = len(adj_type.dims)

        offset = 0
        for arg in array_args:
            arg_type = self._type_of(arg)
            if not isinstance(arg_type, ArrayType):
                continue
            size = arg_type.dims[axis]
            if not isinstance(size, int):
                continue

            if id(arg) in var_map:
                arg_name = var_map[id(arg)]

                # Build IndexExpr: adj sliced along concat axis
                components = []
                for d in range(rank):
                    if d == axis:
                        start = IntLiteral(offset, _DUMMY_SPAN)
                        end = IntLiteral(offset + size, _DUMMY_SPAN)
                        self.type_map[id(start)] = ScalarType("i32")
                        self.type_map[id(end)] = ScalarType("i32")
                        ic = IndexComponent("slice", None, start, end, _DUMMY_SPAN)
                        ic.static_size = size
                        components.append(ic)
                    else:
                        components.append(IndexComponent("full", None, None, None, _DUMMY_SPAN))

                slice_expr = IndexExpr(adj, components, _DUMMY_SPAN)
                self.type_map[id(slice_expr)] = arg_type
                self._accumulate(adjoints, arg_name, slice_expr)

            offset += size

    def _backprop_stack(self, args: list[Expr], adj: Expr,
                         adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through stack: slice adj along the stack axis, reshape to remove the extra dim."""
        axis = args[-1].value
        array_args = args[:-1]

        adj_type = self._type_of(adj)

        # If adj is scalar, just accumulate to each input
        if not isinstance(adj_type, ArrayType):
            for arg in array_args:
                if id(arg) in var_map:
                    self._accumulate(adjoints, var_map[id(arg)], adj)
            return

        rank = len(adj_type.dims)

        for i, arg in enumerate(array_args):
            arg_type = self._type_of(arg)
            if not isinstance(arg_type, ArrayType):
                continue
            if id(arg) not in var_map:
                continue
            arg_name = var_map[id(arg)]

            # Slice adj along stack axis at position i, size 1
            components = []
            for d in range(rank):
                if d == axis:
                    start = IntLiteral(i, _DUMMY_SPAN)
                    end = IntLiteral(i + 1, _DUMMY_SPAN)
                    self.type_map[id(start)] = ScalarType("i32")
                    self.type_map[id(end)] = ScalarType("i32")
                    ic = IndexComponent("slice", None, start, end, _DUMMY_SPAN)
                    ic.static_size = 1
                    components.append(ic)
                else:
                    components.append(IndexComponent("full", None, None, None, _DUMMY_SPAN))

            # Slice type: same as adj but with size 1 at axis
            slice_dims = list(adj_type.dims)
            slice_dims[axis] = 1
            slice_type = ArrayType(adj_type.base, tuple(slice_dims))
            slice_expr = IndexExpr(adj, components, _DUMMY_SPAN)
            self.type_map[id(slice_expr)] = slice_type

            # Reshape to remove the size-1 dim (back to original shape)
            dim_literals = []
            for d in arg_type.dims:
                lit = IntLiteral(d, _DUMMY_SPAN)
                self.type_map[id(lit)] = ScalarType("i32")
                dim_literals.append(lit)
            reshape_call = CallExpr("reshape", [slice_expr] + dim_literals, _DUMMY_SPAN)
            self.type_map[id(reshape_call)] = arg_type
            self._accumulate(adjoints, arg_name, reshape_call)

    def _backprop_pad(self, args: list[Expr], adj: Expr,
                       adjoints: dict[str, Expr], var_map: dict[int, str]):
        """Backprop through pad: slice adj to remove padding (extract the original portion)."""
        x_arg = args[0]
        if id(x_arg) not in var_map:
            return
        x_name = var_map[id(x_arg)]
        x_type = self._type_of(x_arg)
        if not isinstance(x_type, ArrayType):
            self._accumulate(adjoints, x_name, adj)
            return

        adj_type = self._type_of(adj)
        if not isinstance(adj_type, ArrayType):
            self._accumulate(adjoints, x_name, adj)
            return

        pad_lo = args[2].value

        rank = len(x_type.dims)
        # Slice adj[pad_lo : pad_lo + orig_dim] along each axis
        components = []
        for d in range(rank):
            start = IntLiteral(pad_lo, _DUMMY_SPAN)
            end = IntLiteral(pad_lo + x_type.dims[d], _DUMMY_SPAN)
            self.type_map[id(start)] = ScalarType("i32")
            self.type_map[id(end)] = ScalarType("i32")
            ic = IndexComponent("slice", None, start, end, _DUMMY_SPAN)
            ic.static_size = x_type.dims[d]
            components.append(ic)

        slice_expr = IndexExpr(adj, components, _DUMMY_SPAN)
        self.type_map[id(slice_expr)] = x_type
        self._accumulate(adjoints, x_name, slice_expr)
