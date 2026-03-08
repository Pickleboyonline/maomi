from __future__ import annotations

from ...ast_nodes import (
    CallExpr,
    _Conv2dGrad,
    _MaxPoolGrad,
    _AvgPoolGrad,
)
from ...types import MaomiType, ScalarType, ArrayType
from ...errors import MaomiError
from .utils import _mlir_type


class ConvCodegenMixin:

    def _extract_conv2d_params(self, expr: CallExpr) -> tuple[int, int, int, int]:
        """Extract (stride_h, stride_w, pad_h, pad_w) from conv2d args."""
        nargs = len(expr.args)
        if nargs == 2:
            return (1, 1, 0, 0)
        elif nargs == 4:
            return (expr.args[2].value, expr.args[2].value,
                    expr.args[3].value, expr.args[3].value)
        else:
            return (expr.args[2].value, expr.args[3].value,
                    expr.args[4].value, expr.args[5].value)

    def _gen_conv2d(self, expr: CallExpr, env: dict[str, str]) -> str:
        lhs = self._gen_expr(expr.args[0], env)
        rhs = self._gen_expr(expr.args[1], env)
        lhs_type = self._type_of(expr.args[0])
        rhs_type = self._type_of(expr.args[1])
        result_type = self._type_of(expr)

        sh, sw, ph, pw = self._extract_conv2d_params(expr)

        var = self._fresh()
        self._emit(
            f"{var} = stablehlo.convolution({lhs}, {rhs}) "
            f"dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], "
            f"window = {{stride = [{sh}, {sw}], pad = [[{ph}, {ph}], [{pw}, {pw}]], "
            f"lhs_dilate = [1, 1], rhs_dilate = [1, 1]}} "
            f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
            f": ({_mlir_type(lhs_type)}, {_mlir_type(rhs_type)}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_reduce_window(self, input_ssa: str, input_type: ArrayType,
                           result_type: MaomiType, reducer_op: str,
                           init_value: str, window_dims: list[int],
                           window_strides: list[int],
                           padding: list[tuple[int, int]]) -> str:
        """Emit stablehlo.reduce_window with the given reducer."""
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        ndims = len(window_dims)

        init_var = self._fresh()
        self._emit(f"{init_var} = stablehlo.constant dense<{init_value}> : {mlir_scalar}")

        dims_str = ", ".join(str(d) for d in window_dims)
        strides_str = ", ".join(str(s) for s in window_strides)
        base_dilations_str = ", ".join("1" for _ in window_dims)

        # Build padding as dense tensor attribute
        # Check if all padding is zero
        all_zero_pad = all(lo == 0 and hi == 0 for lo, hi in padding)
        if all_zero_pad:
            pad_attr = f"dense<0> : tensor<{ndims}x2xi64>"
        else:
            pad_rows = ", ".join(f"{lo}, {hi}" for lo, hi in padding)
            pad_attr = f"dense<[[{pad_rows}]]> : tensor<{ndims}x2xi64>"

        var = self._fresh()
        self._emit(
            f'{var} = "stablehlo.reduce_window"({input_ssa}, {init_var}) '
            f"<{{base_dilations = array<i64: {base_dilations_str}>, "
            f"padding = {pad_attr}, "
            f"window_dilations = array<i64: {base_dilations_str}>, "
            f"window_dimensions = array<i64: {dims_str}>, "
            f"window_strides = array<i64: {strides_str}>}}> ({{"
        )
        # Emit reducer body
        self._indent += 1
        a_var = self._fresh()
        b_var = self._fresh()
        self._emit(f"^bb0({a_var}: {mlir_scalar}, {b_var}: {mlir_scalar}):")
        self._indent += 1
        r_var = self._fresh()
        self._emit(f"{r_var} = {reducer_op} {a_var}, {b_var} : {mlir_scalar}")
        self._emit(f"stablehlo.return {r_var} : {mlir_scalar}")
        self._indent -= 1
        self._indent -= 1
        self._emit(
            f"}}) : ({_mlir_type(input_type)}, {mlir_scalar}) -> {_mlir_type(result_type)}"
        )
        return var

    def _gen_max_pool(self, expr: CallExpr, env: dict[str, str]) -> str:
        input_ssa = self._gen_expr(expr.args[0], env)
        input_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.args[1].value, expr.args[2].value
        sh, sw = expr.args[3].value, expr.args[4].value

        base = input_type.base
        if base in ("f32", "f64"):
            init = "0xFF800000" if base == "f32" else "0xFFF0000000000000"
        else:
            init = str(-(2**31)) if base == "i32" else str(-(2**63))

        return self._gen_reduce_window(
            input_ssa, input_type, result_type,
            "stablehlo.maximum", init,
            [1, 1, wh, ww], [1, 1, sh, sw],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )

    def _gen_avg_pool(self, expr: CallExpr, env: dict[str, str]) -> str:
        input_ssa = self._gen_expr(expr.args[0], env)
        input_type = self._type_of(expr.args[0])
        result_type = self._type_of(expr)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.args[1].value, expr.args[2].value
        sh, sw = expr.args[3].value, expr.args[4].value

        # Sum pool
        sum_var = self._gen_reduce_window(
            input_ssa, input_type, result_type,
            "stablehlo.add", "0.000000e+00",
            [1, 1, wh, ww], [1, 1, sh, sw],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )

        # Divide by window size
        count = float(wh * ww)
        count_var = self._fresh()
        mlir_result = _mlir_type(result_type)
        self._emit(f"{count_var} = stablehlo.constant dense<{count:e}> : {mlir_result}")

        var = self._fresh()
        self._emit(f"{var} = stablehlo.divide {sum_var}, {count_var} : {mlir_result}")
        return var

    @staticmethod
    def _dilate_dim(d: int, dilation: int) -> int:
        """max(0, 1 + dilation * (d - 1)) — same as JAX's core.dilate_dim."""
        return max(0, 1 + dilation * (d - 1))

    @staticmethod
    def _conv_vjp_lhs_padding(
        in_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...],
        strides: tuple[int, ...], out_spatial: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
        lhs_dilation: tuple[int, ...], rhs_dilation: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        """Compute VJP padding for input gradient (from JAX convolution.py:1007-1016)."""
        result = []
        for i in range(len(in_spatial)):
            ld = ConvCodegenMixin._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = ConvCodegenMixin._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = ConvCodegenMixin._dilate_dim(out_spatial[i], strides[i])
            pad_before = rd - padding[i][0] - 1
            pad_after = ld + rd - 1 - od - pad_before
            result.append((pad_before, pad_after))
        return result

    @staticmethod
    def _conv_vjp_rhs_padding(
        in_spatial: tuple[int, ...], kernel_spatial: tuple[int, ...],
        strides: tuple[int, ...], out_spatial: tuple[int, ...],
        padding: tuple[tuple[int, int], ...],
        lhs_dilation: tuple[int, ...], rhs_dilation: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        """Compute VJP padding for kernel gradient (from JAX convolution.py:1019-1032)."""
        result = []
        for i in range(len(in_spatial)):
            ld = ConvCodegenMixin._dilate_dim(in_spatial[i], lhs_dilation[i])
            rd = ConvCodegenMixin._dilate_dim(kernel_spatial[i], rhs_dilation[i])
            od = ConvCodegenMixin._dilate_dim(out_spatial[i], strides[i])
            pad_lo = padding[i][0]
            pads_from_lhs = od - ld
            pads_from_rhs = rd - pad_lo - 1
            pad_hi = pads_from_lhs + pads_from_rhs
            result.append((pad_lo, pad_hi))
        return result

    def _gen_conv2d_grad(self, expr: _Conv2dGrad, env: dict[str, str]) -> str:
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        input_type = self._type_of(expr.input_expr)
        kernel_type = self._type_of(expr.kernel_expr)
        assert isinstance(input_type, ArrayType) and isinstance(kernel_type, ArrayType)
        assert isinstance(adj_type, ArrayType)

        N, Ci, H, W = input_type.dims
        Co, Ki, Kh, Kw = kernel_type.dims
        _, _, OH, OW = adj_type.dims
        sh, sw = expr.strides
        ph, pw = expr.padding
        fwd_padding = ((ph, ph), (pw, pw))

        if expr.wrt == "lhs":
            # grad w.r.t. input: conv(adj, reverse(kernel)) with VJP padding
            # reverse kernel along spatial dims [2, 3]
            kernel_ssa = self._gen_expr(expr.kernel_expr, env)
            rev_kernel = self._fresh()
            self._emit(
                f"{rev_kernel} = stablehlo.reverse {kernel_ssa}, dims = [2, 3] "
                f": {_mlir_type(kernel_type)}"
            )

            vjp_pad = self._conv_vjp_lhs_padding(
                (H, W), (Kh, Kw), (sh, sw), (OH, OW),
                fwd_padding, (1, 1), (1, 1),
            )

            # Transposed dimension numbers: swap lhs_spec <-> out_spec, transpose rhs_spec
            # Forward: [b,f,0,1]x[o,i,0,1]->[b,f,0,1]
            # Backward (lhs): [b,f,0,1]x[i,o,0,1]->[b,f,0,1]
            # (rhs spec transposed: swap o<->i)
            result_type = input_type  # gradient has same shape as input
            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.convolution({adj_ssa}, {rev_kernel}) "
                f"dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], "
                f"window = {{stride = [1, 1], pad = [[{vjp_pad[0][0]}, {vjp_pad[0][1]}], "
                f"[{vjp_pad[1][0]}, {vjp_pad[1][1]}]], "
                f"lhs_dilate = [{sh}, {sw}], rhs_dilate = [1, 1]}} "
                f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
                f": ({_mlir_type(adj_type)}, {_mlir_type(kernel_type)}) -> {_mlir_type(result_type)}"
            )
            return var

        else:  # wrt == "rhs"
            # grad w.r.t. kernel: conv(input, adj) with transposed roles
            input_ssa = self._gen_expr(expr.input_expr, env)

            vjp_pad = self._conv_vjp_rhs_padding(
                (H, W), (Kh, Kw), (sh, sw), (OH, OW),
                fwd_padding, (1, 1), (1, 1),
            )

            # Transposed dim numbers for kernel gradient (from JAX):
            # lhs (input): [f, b, 0, 1]
            # rhs (adj):   [i, o, 0, 1]  (StableHLO rhs always uses i/o labels)
            # out (kernel grad): [f, b, 0, 1]
            result_type = kernel_type  # gradient has same shape as kernel
            var = self._fresh()
            self._emit(
                f"{var} = stablehlo.convolution({input_ssa}, {adj_ssa}) "
                f"dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[f, b, 0, 1], "
                f"window = {{stride = [1, 1], pad = [[{vjp_pad[0][0]}, {vjp_pad[0][1]}], "
                f"[{vjp_pad[1][0]}, {vjp_pad[1][1]}]], "
                f"lhs_dilate = [1, 1], rhs_dilate = [{sh}, {sw}]}} "
                f"{{batch_group_count = 1 : i64, feature_group_count = 1 : i64}} "
                f": ({_mlir_type(input_type)}, {_mlir_type(adj_type)}) -> {_mlir_type(result_type)}"
            )
            return var

    def _gen_max_pool_grad(self, expr: _MaxPoolGrad, env: dict[str, str]) -> str:
        """Emit select_and_scatter for max_pool backward pass."""
        input_ssa = self._gen_expr(expr.input_expr, env)
        adj_ssa = self._gen_expr(expr.adj, env)
        input_type = self._type_of(expr.input_expr)
        adj_type = self._type_of(expr.adj)
        assert isinstance(input_type, ArrayType)

        wh, ww = expr.window
        sh, sw = expr.strides
        result_type = input_type  # gradient has same shape as input
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)

        # Init value (0.0)
        init_var = self._fresh()
        self._emit(f"{init_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        var = self._fresh()
        self._emit(
            f"{var} = \"stablehlo.select_and_scatter\"({input_ssa}, {adj_ssa}, {init_var}) "
            f"({{")
        # Select region (ge comparator)
        self._indent += 1
        sa = self._fresh()
        sb = self._fresh()
        self._emit(f"^bb0({sa}: {mlir_scalar}, {sb}: {mlir_scalar}):")
        self._indent += 1
        cmp_var = self._fresh()
        self._emit(f"{cmp_var} = stablehlo.compare GE, {sa}, {sb} : ({mlir_scalar}, {mlir_scalar}) -> tensor<i1>")
        self._emit(f"stablehlo.return {cmp_var} : tensor<i1>")
        self._indent -= 1
        self._indent -= 1
        self._emit("}, {")
        # Scatter region (add)
        self._indent += 1
        sc = self._fresh()
        sd = self._fresh()
        self._emit(f"^bb0({sc}: {mlir_scalar}, {sd}: {mlir_scalar}):")
        self._indent += 1
        add_var = self._fresh()
        self._emit(f"{add_var} = stablehlo.add {sc}, {sd} : {mlir_scalar}")
        self._emit(f"stablehlo.return {add_var} : {mlir_scalar}")
        self._indent -= 1
        self._indent -= 1
        self._emit("}) {")
        self._indent += 1
        self._emit(f"window_dimensions = array<i64: 1, 1, {wh}, {ww}>,")
        self._emit(f"window_strides = array<i64: 1, 1, {sh}, {sw}>,")
        self._emit(f"padding = dense<0> : tensor<4x2xi64>")
        self._indent -= 1
        self._emit(
            f"}} : ({_mlir_type(input_type)}, {_mlir_type(adj_type)}, {mlir_scalar}) "
            f"-> {_mlir_type(result_type)}"
        )
        return var

    def _gen_avg_pool_grad(self, expr: _AvgPoolGrad, env: dict[str, str]) -> str:
        """Emit backward pass for avg_pool: scale, pad, reduce_window_sum."""
        adj_ssa = self._gen_expr(expr.adj, env)
        adj_type = self._type_of(expr.adj)
        input_type = self._type_of(expr.input_expr)
        assert isinstance(input_type, ArrayType) and isinstance(adj_type, ArrayType)

        wh, ww = expr.window
        sh, sw = expr.strides
        N, C, H, W = input_type.dims
        _, _, OH, OW = adj_type.dims

        # 1. Scale adjoint by 1/(wh*ww)
        count = float(wh * ww)
        scale_var = self._fresh()
        mlir_adj = _mlir_type(adj_type)
        self._emit(f"{scale_var} = stablehlo.constant dense<{1.0/count:e}> : {mlir_adj}")
        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {adj_ssa}, {scale_var} : {mlir_adj}")

        # 2. Compute VJP padding
        vjp_pad = self._conv_vjp_lhs_padding(
            (1, 1, H, W), (1, 1, wh, ww), (1, 1, sh, sw), (1, 1, OH, OW),
            ((0, 0), (0, 0), (0, 0), (0, 0)), (1, 1, 1, 1), (1, 1, 1, 1),
        )

        # 3. Pad the scaled adjoint (edge padding + interior padding of stride-1)
        scalar_type = ScalarType(input_type.base)
        mlir_scalar = _mlir_type(scalar_type)
        zero_var = self._fresh()
        self._emit(f"{zero_var} = stablehlo.constant dense<0.000000e+00> : {mlir_scalar}")

        # Build padding config: [low, high, interior] for each dim
        # Interior padding inserts stride-1 zeros between elements
        interior = [0, 0, sh - 1, sw - 1]
        # Compute padded type
        padded_dims = []
        for i, d in enumerate(adj_type.dims):
            assert isinstance(d, int)
            lo, hi = vjp_pad[i]
            padded_d = lo + d + (d - 1) * interior[i] + hi
            padded_dims.append(padded_d)
        padded_type = ArrayType(input_type.base, tuple(padded_dims))

        padded = self._fresh()
        low_str = ", ".join(str(vjp_pad[i][0]) for i in range(4))
        high_str = ", ".join(str(vjp_pad[i][1]) for i in range(4))
        interior_str = ", ".join(str(interior[i]) for i in range(4))
        self._emit(
            f"{padded} = stablehlo.pad {scaled}, {zero_var}, "
            f"low = [{low_str}], high = [{high_str}], interior = [{interior_str}] "
            f": ({mlir_adj}, {mlir_scalar}) -> {_mlir_type(padded_type)}"
        )

        # 4. reduce_window sum with window=original window, stride=1
        result_type = input_type
        return self._gen_reduce_window(
            padded, padded_type, result_type,
            "stablehlo.add", "0.000000e+00",
            [1, 1, wh, ww], [1, 1, 1, 1],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
        )
