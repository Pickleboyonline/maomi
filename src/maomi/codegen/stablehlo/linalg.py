from __future__ import annotations

from ...ast_nodes import CallExpr, BoolLiteral
from ...types import ArrayType
from .utils import _mlir_type


class LinalgCodegenMixin:

    def _gen_cholesky(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.cholesky with lower=true."""
        x = self._gen_expr(expr.args[0], env)
        result_type = self._type_of(expr)
        mlir_t = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f'{var} = "stablehlo.cholesky"({x}) '
            f'{{lower = true}} : ({mlir_t}) -> {mlir_t}'
        )
        return var

    def _gen_triangular_solve(self, expr: CallExpr, env: dict[str, str]) -> str:
        """Emit stablehlo.triangular_solve."""
        a = self._gen_expr(expr.args[0], env)
        b = self._gen_expr(expr.args[1], env)
        a_type = self._type_of(expr.args[0])
        b_type = self._type_of(expr.args[1])
        result_type = self._type_of(expr)

        assert isinstance(expr.args[2], BoolLiteral)
        assert isinstance(expr.args[3], BoolLiteral)
        lower = "true" if expr.args[2].value else "false"
        left_side = "true" if expr.args[3].value else "false"

        mlir_a = _mlir_type(a_type)
        mlir_b = _mlir_type(b_type)
        mlir_result = _mlir_type(result_type)
        var = self._fresh()
        self._emit(
            f'{var} = "stablehlo.triangular_solve"({a}, {b}) '
            f'{{left_side = {left_side}, lower = {lower}, '
            f'unit_diagonal = false, '
            f'transpose_a = #stablehlo<transpose NO_TRANSPOSE>}} '
            f': ({mlir_a}, {mlir_b}) -> {mlir_result}'
        )
        return var
