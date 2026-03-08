from __future__ import annotations

from ..ast_nodes import (
    Identifier,
    IntLiteral,
    FloatLiteral,
    Block,
    IfExpr,
    ScanExpr,
    MapExpr,
    _ScanGrad,
    _WhileGrad,
    _Conv2dGrad,
    _MaxPoolGrad,
    _AvgPoolGrad,
    _FoldGrad,
    _ReduceSum,
    Expr,
)
from ..types import MaomiType, ScalarType, ArrayType
from ..errors import MaomiError
from .constants import (
    _DUMMY_SPAN,
    _collect_free_vars,
)


class ComplexGradRulesMixin:
    def _backprop_conv_pool(self, callee: str, args: list[Expr], adj: Expr,
                             adjoints: dict[str, Expr], var_map: dict[int, str],
                             node: Expr):
        """Backprop through conv2d, max_pool, avg_pool."""
        if callee == "conv2d":
            # conv2d(input, kernel, ...) — bilinear
            input_expr = args[0]
            kernel_expr = args[1]
            input_type = self._type_of(input_expr)
            kernel_type = self._type_of(kernel_expr)
            assert isinstance(input_type, ArrayType) and isinstance(kernel_type, ArrayType)

            # Extract stride/pad from literal args
            nargs = len(args)
            if nargs == 2:
                sh, sw, ph, pw = 1, 1, 0, 0
            elif nargs == 4:
                sh = sw = args[2].value
                ph = pw = args[3].value
            else:
                sh, sw = args[2].value, args[3].value
                ph, pw = args[4].value, args[5].value

            strides = (sh, sw)
            padding = (ph, pw)

            # grad w.r.t. input
            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _Conv2dGrad(
                    input_expr, kernel_expr, adj, "lhs", strides, padding, _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

            # grad w.r.t. kernel
            if id(kernel_expr) in var_map:
                kernel_name = var_map[id(kernel_expr)]
                grad_node = _Conv2dGrad(
                    input_expr, kernel_expr, adj, "rhs", strides, padding, _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = kernel_type
                self._accumulate(adjoints, kernel_name, grad_node)

        elif callee == "max_pool":
            # max_pool(input, wh, ww, sh, sw)
            input_expr = args[0]
            input_type = self._type_of(input_expr)
            wh, ww = args[1].value, args[2].value
            sh, sw = args[3].value, args[4].value

            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _MaxPoolGrad(
                    input_expr, adj, (wh, ww), (sh, sw), _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

        elif callee == "avg_pool":
            # avg_pool(input, wh, ww, sh, sw)
            input_expr = args[0]
            input_type = self._type_of(input_expr)
            wh, ww = args[1].value, args[2].value
            sh, sw = args[3].value, args[4].value

            if id(input_expr) in var_map:
                input_name = var_map[id(input_expr)]
                grad_node = _AvgPoolGrad(
                    input_expr, adj, (wh, ww), (sh, sw), _DUMMY_SPAN,
                )
                self.type_map[id(grad_node)] = input_type
                self._accumulate(adjoints, input_name, grad_node)

    def _backprop_if(self, cond: Expr, then_b: Block, else_b: Block,
                      adj: Expr, adjoints: dict[str, Expr],
                      var_map: dict[int, str], node: Expr):
        """Backprop through if/else: differentiate both branches, select with condition.

        Two strategies depending on whether branch expressions are already on the tape:

        1. If a branch expr is on the tape (e.g. after function inlining), it's a direct
           reference — propagate adj masked by the condition. This avoids differentiating
           through compound expressions (like matmul) that the outer tape will handle.

        2. Otherwise, use element-wise symbolic differentiation for each free variable
           (the original approach, correct for element-wise branch expressions).
        """
        then_expr = then_b.expr
        else_expr = else_b.expr
        if then_expr is None or else_expr is None:
            return

        # Check if branch expressions are directly on the tape
        then_tape = var_map.get(id(then_expr))
        else_tape = var_map.get(id(else_expr))

        # Strategy 1: Direct tape references — propagate masked adjoint
        if then_tape is not None or else_tape is not None:
            self._backprop_if_direct(cond, adj, then_tape, else_tape, adjoints, node)
            return

        # Strategy 2: Element-wise differentiation (original approach)
        self._backprop_if_elementwise(cond, then_expr, else_expr, adj, adjoints, var_map)

    def _backprop_if_direct(self, cond: Expr, adj: Expr,
                             then_tape: str | None, else_tape: str | None,
                             adjoints: dict[str, Expr], node: Expr):
        """Backprop when branch exprs are tape variables: mask adjoint with condition."""
        adj_type = self.type_map.get(id(adj))
        zero = self._make_zero(adj_type) if adj_type is not None else self._make_float(0.0)

        if then_tape is not None and else_tape is not None:
            if then_tape == else_tape:
                # Both branches reference the same variable — full adjoint
                self._accumulate(adjoints, then_tape, adj)
            else:
                # Different variables in each branch
                then_block = Block([], adj, _DUMMY_SPAN)
                else_block = Block([], zero, _DUMMY_SPAN)
                then_if = IfExpr(cond, then_block, else_block, _DUMMY_SPAN)
                if adj_type is not None:
                    self.type_map[id(then_if)] = adj_type
                self._accumulate(adjoints, then_tape, then_if)

                zero2 = self._make_zero(adj_type) if adj_type is not None else self._make_float(0.0)
                then_block2 = Block([], zero2, _DUMMY_SPAN)
                else_block2 = Block([], adj, _DUMMY_SPAN)
                else_if = IfExpr(cond, then_block2, else_block2, _DUMMY_SPAN)
                if adj_type is not None:
                    self.type_map[id(else_if)] = adj_type
                self._accumulate(adjoints, else_tape, else_if)
        elif then_tape is not None:
            # Only then-branch is a tape var (else is constant like 0.0)
            then_block = Block([], adj, _DUMMY_SPAN)
            else_block = Block([], zero, _DUMMY_SPAN)
            masked = IfExpr(cond, then_block, else_block, _DUMMY_SPAN)
            if adj_type is not None:
                self.type_map[id(masked)] = adj_type
            self._accumulate(adjoints, then_tape, masked)
        elif else_tape is not None:
            # Only else-branch is a tape var
            then_block = Block([], zero, _DUMMY_SPAN)
            else_block = Block([], adj, _DUMMY_SPAN)
            masked = IfExpr(cond, then_block, else_block, _DUMMY_SPAN)
            if adj_type is not None:
                self.type_map[id(masked)] = adj_type
            self._accumulate(adjoints, else_tape, masked)

    def _backprop_if_elementwise(self, cond: Expr, then_expr: Expr, else_expr: Expr,
                                  adj: Expr, adjoints: dict[str, Expr],
                                  var_map: dict[int, str]):
        """Backprop for element-wise branches: compute d(branch)/d(var) symbolically."""
        free_vars = _collect_free_vars(then_expr) | _collect_free_vars(else_expr)
        tape_vars = set(var_map.values())

        for v_name in free_vars:
            if v_name not in tape_vars:
                continue

            then_grad = self._differentiate_branch(then_expr, v_name)
            else_grad = self._differentiate_branch(else_expr, v_name)

            then_block = Block([], then_grad, _DUMMY_SPAN)
            else_block = Block([], else_grad, _DUMMY_SPAN)
            grad_if = IfExpr(cond, then_block, else_block, _DUMMY_SPAN)

            gt = self.type_map.get(id(then_grad))
            if gt is not None:
                self.type_map[id(grad_if)] = gt

            contribution = self._make_binop("*", adj, grad_if)
            self._accumulate(adjoints, v_name, contribution)

    def _differentiate_branch(self, expr: Expr, wrt: str) -> Expr:
        """Compute d(expr)/d(wrt) for a branch expression using a fresh tape."""
        saved_tape_exprs = self._tape_exprs

        tape: list[tuple[str, Expr]] = []
        inner_var_map: dict[int, str] = {}
        self._linearize(expr, tape, inner_var_map, {})

        if id(expr) not in inner_var_map:
            self._tape_exprs = saved_tape_exprs
            return self._make_float(0.0)

        self._tape_exprs = {name: node for name, node in tape}

        output_name = inner_var_map[id(expr)]
        inner_adjoints: dict[str, Expr] = {output_name: self._make_float(1.0)}

        for name, node in reversed(tape):
            if name not in inner_adjoints:
                continue
            adj = inner_adjoints[name]
            self._backprop(name, node, adj, inner_adjoints, inner_var_map)

        self._tape_exprs = saved_tape_exprs

        if wrt in inner_adjoints:
            return inner_adjoints[wrt]
        return self._make_float(0.0)

    def _backprop_map(self, elem_var: str, seq: Expr, body: Block,
                       adj: Expr, adjoints: dict[str, Expr],
                       var_map: dict[int, str], node: Expr):
        """Backprop through map: adj_seq = map elem in seq { d(body)/d(elem) * adj_elem }."""
        seq_name = var_map[id(seq)]
        body_expr = body.expr
        if body_expr is None:
            return

        # Differentiate the body w.r.t. the element variable
        body_grad = self._differentiate_branch(body_expr, elem_var)

        # Build: map elem_var in seq { body_grad * adj_elem }
        # Since adj is the adjoint of the whole map output (an array),
        # and body_grad gives per-element derivative, we want:
        #   adj_seq = adj * map elem_var in seq { body_grad }
        # This works because both are same-shape arrays and * is elementwise.
        grad_map = MapExpr(elem_var, self._make_ref(seq_name, self._type_of(seq)),
                          Block([], body_grad, _DUMMY_SPAN), _DUMMY_SPAN)
        # Type the grad map — same type as the original map
        map_type = self._type_of(node)
        if map_type is not None:
            self.type_map[id(grad_map)] = map_type

        contribution = self._make_binop("*", adj, grad_map)
        self._accumulate(adjoints, seq_name, contribution)

        # Propagate to free variables in the body (other than elem_var)
        # For free var w: d(map x in xs { f(x, w) })/dw = sum_i(adj[i] * d(f(x_i, w))/dw)
        free_vars = _collect_free_vars(body_expr) - {elem_var}
        seq_type = self._type_of(seq)
        for v_name in free_vars:

            # Per-element gradient d(body)/d(w)
            grad_w = self._differentiate_branch(body_expr, v_name)

            # Wrap in map: map elem_var in seq { d(body)/d(w) }
            seq_ref = self._make_ref(seq_name, seq_type)
            grad_map_w = MapExpr(elem_var, seq_ref,
                                 Block([], grad_w, _DUMMY_SPAN), _DUMMY_SPAN)

            # Type the grad map: seq_len prepended to w's type
            w_type = self._type_of(grad_w)
            if isinstance(seq_type, ArrayType) and w_type is not None:
                seq_len = seq_type.dims[0]
                if isinstance(w_type, ScalarType):
                    grad_map_type = ArrayType(w_type.base, (seq_len,))
                elif isinstance(w_type, ArrayType):
                    grad_map_type = ArrayType(w_type.base, (seq_len,) + w_type.dims)
                else:
                    grad_map_type = w_type
                self.type_map[id(grad_map_w)] = grad_map_type
            else:
                grad_map_type = map_type

            # Scale by adjoint: adj * grad_map_w (broadcasts adj over non-batch dims)
            scaled = self._make_binop("*", adj, grad_map_w)

            # Reduce over dimension 0 (the map/batch dimension) to get w-shaped gradient
            reduced = _ReduceSum(expr=scaled, axes=(0,), span=_DUMMY_SPAN)
            if w_type is not None:
                self.type_map[id(reduced)] = w_type

            self._accumulate(adjoints, v_name, reduced)

    def _backprop_while(self, state_var: str, init: Expr,
                         max_iters: int | None, cond: Block, body: Block,
                         adj: Expr, adjoints: dict[str, Expr],
                         var_map: dict[int, str], node: Expr):
        """Backprop through while: emit _WhileGrad for bounded, error for unbounded."""
        if max_iters is None:
            raise MaomiError(
                "reverse-mode AD is not supported through while loops without a limit. "
                "Use 'while state in init limit N { cond } do { body }' for differentiable while loops, "
                "or use scan for fixed-iteration differentiable loops.",
                "<ad>", node.span.line_start, node.span.col_start,
            )

        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivative of body w.r.t. state
        d_body_d_state = self._differentiate_branch(body_expr, state_var)

        fwd_ref = node

        if id(init) in var_map:
            init_name = var_map[id(init)]
            grad_node = _WhileGrad(
                d_body_d_state=d_body_d_state,
                state_var=state_var,
                init=init,
                max_iters=max_iters,
                cond=cond,
                body=body,
                forward_result=fwd_ref,
                adj=adj,
                span=node.span,
            )
            state_type = self._type_of(init)
            self.type_map[id(grad_node)] = state_type
            self._accumulate(adjoints, init_name, grad_node)

    def _backprop_scan(self, carry_var: str, elem_vars: list[str],
                        init: Expr, sequences: list[Expr], body: Block,
                        adj: Expr, adjoints: dict[str, Expr],
                        var_map: dict[int, str], node: Expr):
        """Backprop through scan: emit reverse ScanExpr or _ScanGrad nodes."""
        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivatives of body w.r.t. carry and each elem
        d_body_d_carry = self._differentiate_branch(body_expr, carry_var)
        d_body_d_elems = [self._differentiate_branch(body_expr, ev) for ev in elem_vars]

        # The forward result is the scan node itself — codegen will generate it
        fwd_ref = node

        # Check if derivatives are constant (don't reference carry/elem vars).
        # When constant, we can emit standard ScanExpr nodes (JAX-style)
        # that support grad-of-grad. Non-constant falls back to _ScanGrad.
        all_deriv_vars = _collect_free_vars(d_body_d_carry)
        for de in d_body_d_elems:
            all_deriv_vars |= _collect_free_vars(de)
        constant_derivs = not (all_deriv_vars & {carry_var, *elem_vars})

        # Propagate to init
        if id(init) in var_map:
            init_name = var_map[id(init)]
            init_type = self._type_of(init)
            if constant_derivs:
                # Build reverse scan then extract final carry via indexing.
                # The reverse scan accumulates: carry * d_carry + adj_elem
                # The final carry (last stacked output) is the init gradient.
                rev_scan = self._build_reverse_scan_grad(
                    d_body_d_carry, self._make_float(1.0), adj, sequences[0], node)
                # Get sequence length from type to index the last element
                seq_type = self._type_of(sequences[0])
                if isinstance(seq_type, ArrayType):
                    seq_len = seq_type.dims[0]
                    assert isinstance(seq_len, int)
                    last_idx = IntLiteral(seq_len - 1, _DUMMY_SPAN)
                    self.type_map[id(last_idx)] = ScalarType("i32")
                    from ..ast_nodes import IndexComponent, IndexExpr
                    ic = IndexComponent("single", last_idx, None, None, _DUMMY_SPAN)
                    grad_init = IndexExpr(rev_scan, [ic], _DUMMY_SPAN)
                    if init_type is not None:
                        self.type_map[id(grad_init)] = init_type
                    rev_scan_type = seq_type  # reverse scan stacks carries
                    self.type_map[id(rev_scan)] = rev_scan_type
                else:
                    grad_init = rev_scan
                    if init_type is not None:
                        self.type_map[id(grad_init)] = init_type
            else:
                # Fallback: _ScanGrad for non-constant derivatives
                grad_init = _ScanGrad(
                    d_body_d_carry=d_body_d_carry,
                    d_body_d_elems=d_body_d_elems,
                    carry_var=carry_var,
                    elem_vars=elem_vars,
                    init=init,
                    sequences=sequences,
                    forward_result=fwd_ref,
                    adj=adj,
                    wrt="__init__",
                    span=_DUMMY_SPAN,
                )
                if init_type is not None:
                    self.type_map[id(grad_init)] = init_type
            self._accumulate(adjoints, init_name, grad_init)

        # Propagate to each sequence
        for i, seq in enumerate(sequences):
            if id(seq) not in var_map:
                continue
            seq_name = var_map[id(seq)]

            if constant_derivs:
                # JAX-style: backward pass is just a reverse scan
                grad_seq = self._build_reverse_scan_grad(
                    d_body_d_carry, d_body_d_elems[i], adj, seq, node)
            else:
                # Fallback: _ScanGrad for non-constant derivatives
                wrt_name = seq.name if isinstance(seq, Identifier) else f"__seq_{i}__"
                grad_seq = _ScanGrad(
                    d_body_d_carry=d_body_d_carry,
                    d_body_d_elems=d_body_d_elems,
                    carry_var=carry_var,
                    elem_vars=elem_vars,
                    init=init,
                    sequences=sequences,
                    forward_result=fwd_ref,
                    adj=adj,
                    wrt=wrt_name,
                    span=_DUMMY_SPAN,
                )

            seq_type = self._type_of(seq)
            if seq_type is not None:
                self.type_map[id(grad_seq)] = seq_type
            self._accumulate(adjoints, seq_name, grad_seq)

    def _build_reverse_scan_grad(self, d_carry: Expr, d_elem: Expr,
                                  adj: Expr, seq: Expr, scan_node: Expr) -> ScanExpr:
        """Build a reverse ScanExpr for the backward pass (constant derivative case).

        The backward scan accumulates: new_carry = carry * d_carry + adj_elem * d_elem
        and stacks the carries as the gradient array.
        """
        fresh_carry = self._fresh_name("_adj_c")
        fresh_elem = self._fresh_name("_adj_e")

        # Carry type = element type of scan result (what each step produces)
        scan_type = self._type_of(scan_node)
        if isinstance(scan_type, ArrayType):
            if len(scan_type.dims) == 1:
                carry_type = ScalarType(scan_type.base)
            else:
                carry_type = ArrayType(scan_type.base, scan_type.dims[1:])
        else:
            carry_type = scan_type

        # Create typed carry reference
        carry_ref = Identifier(fresh_carry, _DUMMY_SPAN)
        self.type_map[id(carry_ref)] = carry_type

        # Determine adj element and backward sequence
        adj_type = self._type_of(adj)
        if isinstance(adj_type, ArrayType):
            # adj is an array — use it as the sequence, elem_var gets sliced adj
            back_seq = adj
            adj_element = Identifier(fresh_elem, _DUMMY_SPAN)
            if len(adj_type.dims) == 1:
                self.type_map[id(adj_element)] = ScalarType(adj_type.base)
            else:
                self.type_map[id(adj_element)] = ArrayType(adj_type.base, adj_type.dims[1:])
        else:
            # adj is scalar — use original seq for iteration count, adj directly in body
            back_seq = seq
            adj_element = adj

        # Build body: carry * d_carry + adj_element * d_elem
        # Simplify: skip * 1.0 when derivative is FloatLiteral(1.0)
        def is_one(e):
            return isinstance(e, FloatLiteral) and e.value == 1.0

        term1 = carry_ref if is_one(d_carry) else self._make_binop("*", carry_ref, d_carry)
        term2 = adj_element if is_one(d_elem) else self._make_binop("*", adj_element, d_elem)
        body_expr = self._make_binop("+", term1, term2)
        body_block = Block([], body_expr, _DUMMY_SPAN)

        # Init: zero with carry type
        back_init = self._make_zero(carry_type)

        return ScanExpr(
            carry_var=fresh_carry,
            elem_vars=[fresh_elem],
            init=back_init,
            sequences=[back_seq],
            body=body_block,
            span=_DUMMY_SPAN,
            reverse=True,
        )

    def _backprop_fold(self, carry_var: str, elem_vars: list[str],
                        init: Expr, sequences: list[Expr], body: Block,
                        adj: Expr, adjoints: dict[str, Expr],
                        var_map: dict[int, str], node: Expr):
        """Backprop through fold: emit _FoldGrad nodes."""
        body_expr = body.expr
        if body_expr is None:
            return

        # Compute symbolic derivatives of body w.r.t. carry and each elem
        d_body_d_carry = self._differentiate_branch(body_expr, carry_var)
        d_body_d_elems = [self._differentiate_branch(body_expr, ev) for ev in elem_vars]

        # Propagate to init
        if id(init) in var_map:
            init_name = var_map[id(init)]
            grad_init = _FoldGrad(
                d_body_d_carry=d_body_d_carry,
                d_body_d_elems=d_body_d_elems,
                carry_var=carry_var,
                elem_vars=elem_vars,
                init=init,
                sequences=sequences,
                body=body,
                adj=adj,
                wrt="__init__",
                span=_DUMMY_SPAN,
            )
            init_type = self._type_of(init)
            if init_type is not None:
                self.type_map[id(grad_init)] = init_type
            self._accumulate(adjoints, init_name, grad_init)

        # Propagate to each sequence
        for i, seq in enumerate(sequences):
            if id(seq) not in var_map:
                continue
            seq_name = var_map[id(seq)]

            wrt_name = seq.name if isinstance(seq, Identifier) else f"__seq_{i}__"
            grad_seq = _FoldGrad(
                d_body_d_carry=d_body_d_carry,
                d_body_d_elems=d_body_d_elems,
                carry_var=carry_var,
                elem_vars=elem_vars,
                init=init,
                sequences=sequences,
                body=body,
                adj=adj,
                wrt=wrt_name,
                span=_DUMMY_SPAN,
            )

            seq_type = self._type_of(seq)
            if seq_type is not None:
                self.type_map[id(grad_seq)] = seq_type
            self._accumulate(adjoints, seq_name, grad_seq)
