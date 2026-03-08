from __future__ import annotations

from ...ast_nodes import (
    CallExpr,
)
from ...types import ArrayType
from ...errors import MaomiError
from .utils import _mlir_type


class RNGCodegenMixin:

    def _gen_rng(self, expr: CallExpr, env: dict[str, str]) -> str:
        callee = expr.callee
        if callee == "random.key":
            return self._gen_rng_key(expr, env)
        elif callee == "random.split":
            return self._gen_rng_split(expr, env)
        elif callee == "random.uniform":
            return self._gen_rng_uniform(expr, env)
        elif callee == "random.normal":
            return self._gen_rng_normal(expr, env)
        raise MaomiError(f"codegen: unknown RNG builtin '{callee}'", "<codegen>", 0, 0)

    def _gen_key_to_u64(self, key_ssa: str) -> str:
        """Convert i32[4] key to ui64[2] for rng_bit_generator.

        i32[4] -> bitcast -> ui32[4] -> reshape -> 2x2xui32 -> bitcast -> 2xui64
        """
        key_u32 = self._fresh()
        self._emit(f"{key_u32} = stablehlo.bitcast_convert {key_ssa} : (tensor<4xi32>) -> tensor<4xui32>")
        key_2x2 = self._fresh()
        self._emit(f"{key_2x2} = stablehlo.reshape {key_u32} : (tensor<4xui32>) -> tensor<2x2xui32>")
        key_u64 = self._fresh()
        self._emit(f"{key_u64} = stablehlo.bitcast_convert {key_2x2} : (tensor<2x2xui32>) -> tensor<2xui64>")
        return key_u64

    def _gen_rng_bits(self, key_u64: str, shape: tuple[int, ...]) -> tuple[str, str]:
        """Call rng_bit_generator, return (out_key_u64, bits_ui32)."""
        shape_str = "x".join(str(d) for d in shape)
        out_key = self._fresh()
        bits = self._fresh()
        self._emit(
            f"{out_key}, {bits} = stablehlo.rng_bit_generator {key_u64}, algorithm = DEFAULT "
            f": (tensor<2xui64>) -> (tensor<2xui64>, tensor<{shape_str}xui32>)"
        )
        return out_key, bits

    def _gen_bits_to_uniform(self, bits_ssa: str, shape: tuple[int, ...]) -> str:
        """Convert ui32 random bits to f32 uniform [0, 1).

        bits >> 9 | 0x3F800000 -> bitcast to f32 -> subtract 1.0
        """
        shape_str = "x".join(str(d) for d in shape)
        ui32_type = f"tensor<{shape_str}xui32>"
        f32_type = f"tensor<{shape_str}xf32>"

        # Broadcast constants to shape
        nine_s = self._fresh()
        self._emit(f"{nine_s} = stablehlo.constant dense<9> : tensor<ui32>")
        nine = self._fresh()
        dims = "" if not shape else "dims = []"
        self._emit(f"{nine} = stablehlo.broadcast_in_dim {nine_s}, {dims} : (tensor<ui32>) -> {ui32_type}")

        shifted = self._fresh()
        self._emit(f"{shifted} = stablehlo.shift_right_logical {bits_ssa}, {nine} : {ui32_type}")

        # 0x3F800000 = 1065353216 = bit pattern of 1.0f
        one_bits_s = self._fresh()
        self._emit(f"{one_bits_s} = stablehlo.constant dense<1065353216> : tensor<ui32>")
        one_bits = self._fresh()
        self._emit(f"{one_bits} = stablehlo.broadcast_in_dim {one_bits_s}, {dims} : (tensor<ui32>) -> {ui32_type}")

        mantissa = self._fresh()
        self._emit(f"{mantissa} = stablehlo.or {shifted}, {one_bits} : {ui32_type}")

        as_float = self._fresh()
        self._emit(f"{as_float} = stablehlo.bitcast_convert {mantissa} : ({ui32_type}) -> {f32_type}")

        one_f_s = self._fresh()
        self._emit(f"{one_f_s} = stablehlo.constant dense<1.000000e+00> : tensor<f32>")
        one_f = self._fresh()
        self._emit(f"{one_f} = stablehlo.broadcast_in_dim {one_f_s}, {dims} : (tensor<f32>) -> {f32_type}")

        result = self._fresh()
        self._emit(f"{result} = stablehlo.subtract {as_float}, {one_f} : {f32_type}")
        return result

    def _gen_rng_key(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_key(seed) -> i32[4]: pack [0, 0, 0, seed]."""
        seed_ssa = self._gen_expr(expr.args[0], env)

        zeros = self._fresh()
        self._emit(f"{zeros} = stablehlo.constant dense<0> : tensor<3xi32>")
        seed_1d = self._fresh()
        self._emit(f"{seed_1d} = stablehlo.reshape {seed_ssa} : (tensor<i32>) -> tensor<1xi32>")
        key = self._fresh()
        self._emit(
            f"{key} = stablehlo.concatenate {zeros}, {seed_1d}, dim = 0 "
            f": (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>"
        )
        return key

    def _gen_rng_split(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_split(key, n) -> i32[n, 4]: generate n subkeys."""
        key_ssa = self._gen_expr(expr.args[0], env)
        n = expr.args[1].value  # IntLiteral, validated by type checker

        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, (n * 4,))

        # Reshape flat ui32[n*4] -> ui32[n, 4]
        reshaped_u = self._fresh()
        self._emit(f"{reshaped_u} = stablehlo.reshape {bits} : (tensor<{n * 4}xui32>) -> tensor<{n}x4xui32>")

        # Bitcast back to i32[n, 4]
        result = self._fresh()
        self._emit(f"{result} = stablehlo.bitcast_convert {reshaped_u} : (tensor<{n}x4xui32>) -> tensor<{n}x4xi32>")
        return result

    def _gen_rng_uniform(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_uniform(key, low, high, d1, ...) -> f32[d1, ...]."""
        key_ssa = self._gen_expr(expr.args[0], env)
        low_ssa = self._gen_expr(expr.args[1], env)
        high_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"

        # Generate uniform [0, 1)
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, shape)
        base_uniform = self._gen_bits_to_uniform(bits, shape)

        # Scale to [low, high): result = base_uniform * (high - low) + low
        dims = "dims = []"
        low_bc = self._fresh()
        self._emit(f"{low_bc} = stablehlo.broadcast_in_dim {low_ssa}, {dims} : (tensor<f32>) -> {f32_type}")
        high_bc = self._fresh()
        self._emit(f"{high_bc} = stablehlo.broadcast_in_dim {high_ssa}, {dims} : (tensor<f32>) -> {f32_type}")

        range_val = self._fresh()
        self._emit(f"{range_val} = stablehlo.subtract {high_bc}, {low_bc} : {f32_type}")
        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {base_uniform}, {range_val} : {f32_type}")
        result = self._fresh()
        self._emit(f"{result} = stablehlo.add {scaled}, {low_bc} : {f32_type}")
        return result

    def _gen_rng_normal(self, expr: CallExpr, env: dict[str, str]) -> str:
        """rng_normal(key, mean, std, d1, ...) -> f32[d1, ...].

        Uses Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
        Generates 2x elements, splits into pairs, applies transform.
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        mean_ssa = self._gen_expr(expr.args[1], env)
        std_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        numel = 1
        for d in shape:
            numel *= d

        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"
        flat_f32 = f"tensor<{numel}xf32>"

        # Generate 2*numel uniform values as a flat array
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, (2 * numel,))
        uniform_2n = self._gen_bits_to_uniform(bits, (2 * numel,))

        flat_2n = f"tensor<{2 * numel}xf32>"

        # Split into u1 and u2
        u1 = self._fresh()
        self._emit(f"{u1} = stablehlo.slice {uniform_2n} [0:{numel}] : ({flat_2n}) -> {flat_f32}")
        u2 = self._fresh()
        self._emit(f"{u2} = stablehlo.slice {uniform_2n} [{numel}:{2 * numel}] : ({flat_2n}) -> {flat_f32}")

        # Clamp u1 away from 0 (log safety)
        eps_s = self._fresh()
        self._emit(f"{eps_s} = stablehlo.constant dense<1.000000e-07> : tensor<f32>")
        eps = self._fresh()
        self._emit(f"{eps} = stablehlo.broadcast_in_dim {eps_s}, dims = [] : (tensor<f32>) -> {flat_f32}")
        u1_safe = self._fresh()
        self._emit(f"{u1_safe} = stablehlo.maximum {u1}, {eps} : {flat_f32}")

        # Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2*pi*u2)
        log_u1 = self._fresh()
        self._emit(f"{log_u1} = stablehlo.log {u1_safe} : {flat_f32}")

        neg2_s = self._fresh()
        self._emit(f"{neg2_s} = stablehlo.constant dense<-2.000000e+00> : tensor<f32>")
        neg2 = self._fresh()
        self._emit(f"{neg2} = stablehlo.broadcast_in_dim {neg2_s}, dims = [] : (tensor<f32>) -> {flat_f32}")

        neg2_log = self._fresh()
        self._emit(f"{neg2_log} = stablehlo.multiply {neg2}, {log_u1} : {flat_f32}")
        radius = self._fresh()
        self._emit(f"{radius} = stablehlo.sqrt {neg2_log} : {flat_f32}")

        two_pi_s = self._fresh()
        self._emit(f"{two_pi_s} = stablehlo.constant dense<6.28318530e+00> : tensor<f32>")
        two_pi = self._fresh()
        self._emit(f"{two_pi} = stablehlo.broadcast_in_dim {two_pi_s}, dims = [] : (tensor<f32>) -> {flat_f32}")

        angle = self._fresh()
        self._emit(f"{angle} = stablehlo.multiply {two_pi}, {u2} : {flat_f32}")
        cos_angle = self._fresh()
        self._emit(f"{cos_angle} = stablehlo.cosine {angle} : {flat_f32}")

        z_flat = self._fresh()
        self._emit(f"{z_flat} = stablehlo.multiply {radius}, {cos_angle} : {flat_f32}")

        # Scale: result = z * std + mean
        std_bc = self._fresh()
        self._emit(f"{std_bc} = stablehlo.broadcast_in_dim {std_ssa}, dims = [] : (tensor<f32>) -> {flat_f32}")
        mean_bc = self._fresh()
        self._emit(f"{mean_bc} = stablehlo.broadcast_in_dim {mean_ssa}, dims = [] : (tensor<f32>) -> {flat_f32}")

        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {z_flat}, {std_bc} : {flat_f32}")
        shifted = self._fresh()
        self._emit(f"{shifted} = stablehlo.add {scaled}, {mean_bc} : {flat_f32}")

        # Reshape from flat to target shape
        if len(shape) == 1:
            return shifted
        result = self._fresh()
        self._emit(f"{result} = stablehlo.reshape {shifted} : ({flat_f32}) -> {f32_type}")
        return result
