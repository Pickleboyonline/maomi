from __future__ import annotations

from ...ast_nodes import (
    CallExpr,
)
from ...types import ArrayType, ScalarType
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
        elif callee == "random.bernoulli":
            return self._gen_rng_bernoulli(expr, env)
        elif callee == "random.categorical":
            return self._gen_rng_categorical(expr, env)
        elif callee == "random.truncated_normal":
            return self._gen_rng_truncated_normal(expr, env)
        elif callee == "random.exponential":
            return self._gen_rng_exponential(expr, env)
        elif callee == "random.randint":
            return self._gen_rng_randint(expr, env)
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

    def _gen_box_muller(self, key_ssa: str, numel: int) -> str:
        """Generate standard normal (mean=0, std=1) flat f32 array via Box-Muller.

        Returns SSA var of shape tensor<{numel}xf32>.
        Shared by _gen_rng_normal and _gen_rng_truncated_normal.
        """
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
        return z_flat

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

        z_flat = self._gen_box_muller(key_ssa, numel)

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

    def _gen_rng_bernoulli(self, expr: CallExpr, env: dict[str, str]) -> str:
        """random.bernoulli(key, prob, d1, ...) -> f32[d1, ...].

        Generate uniform in [0, 1), compare < prob, cast to f32.
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        prob_ssa = self._gen_expr(expr.args[1], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"
        bool_type = f"tensor<{shape_str}xi1>"

        # Generate uniform [0, 1)
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, shape)
        base_uniform = self._gen_bits_to_uniform(bits, shape)

        # Broadcast prob to shape
        prob_bc = self._fresh()
        self._emit(f"{prob_bc} = stablehlo.broadcast_in_dim {prob_ssa}, dims = [] : (tensor<f32>) -> {f32_type}")

        # Compare: uniform < prob
        cmp = self._fresh()
        self._emit(f"{cmp} = stablehlo.compare LT, {base_uniform}, {prob_bc} : ({f32_type}, {f32_type}) -> {bool_type}")

        # Convert bool to f32 (true=1.0, false=0.0)
        result = self._fresh()
        self._emit(f"{result} = stablehlo.convert {cmp} : ({bool_type}) -> {f32_type}")
        return result

    def _gen_rng_categorical(self, expr: CallExpr, env: dict[str, str]) -> str:
        """random.categorical(key, logits) -> i32[...] via Gumbel-max trick.

        1. Generate uniform random with same shape as logits
        2. Compute gumbel noise: -log(-log(uniform))
        3. Add to logits: logits + gumbel
        4. Argmax over last dimension (variadic reduce)
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        logits_ssa = self._gen_expr(expr.args[1], env)

        logits_type = self._type_of(expr.args[1])
        assert isinstance(logits_type, ArrayType)
        logits_shape = tuple(logits_type.dims)
        logits_shape_str = "x".join(str(d) for d in logits_shape)
        logits_f32_type = f"tensor<{logits_shape_str}xf32>"

        # Generate uniform in (0, 1) with same shape as logits
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, logits_shape)
        base_uniform = self._gen_bits_to_uniform(bits, logits_shape)

        # Clamp away from 0 and 1 for log safety
        eps_s = self._fresh()
        self._emit(f"{eps_s} = stablehlo.constant dense<1.000000e-07> : tensor<f32>")
        eps = self._fresh()
        self._emit(f"{eps} = stablehlo.broadcast_in_dim {eps_s}, dims = [] : (tensor<f32>) -> {logits_f32_type}")
        u_safe = self._fresh()
        self._emit(f"{u_safe} = stablehlo.maximum {base_uniform}, {eps} : {logits_f32_type}")

        one_s = self._fresh()
        self._emit(f"{one_s} = stablehlo.constant dense<1.000000e+00> : tensor<f32>")
        one_bc = self._fresh()
        self._emit(f"{one_bc} = stablehlo.broadcast_in_dim {one_s}, dims = [] : (tensor<f32>) -> {logits_f32_type}")
        one_minus_eps = self._fresh()
        self._emit(f"{one_minus_eps} = stablehlo.subtract {one_bc}, {eps} : {logits_f32_type}")
        u_clamped = self._fresh()
        self._emit(f"{u_clamped} = stablehlo.minimum {u_safe}, {one_minus_eps} : {logits_f32_type}")

        # Gumbel noise: -log(-log(u))
        log_u = self._fresh()
        self._emit(f"{log_u} = stablehlo.log {u_clamped} : {logits_f32_type}")
        neg_log_u = self._fresh()
        self._emit(f"{neg_log_u} = stablehlo.negate {log_u} : {logits_f32_type}")
        log_neg_log_u = self._fresh()
        self._emit(f"{log_neg_log_u} = stablehlo.log {neg_log_u} : {logits_f32_type}")
        gumbel = self._fresh()
        self._emit(f"{gumbel} = stablehlo.negate {log_neg_log_u} : {logits_f32_type}")

        # Add gumbel noise to logits
        perturbed = self._fresh()
        self._emit(f"{perturbed} = stablehlo.add {logits_ssa}, {gumbel} : {logits_f32_type}")

        # Argmax over last dimension using variadic reduce (same pattern as _gen_argmax)
        last_dim = len(logits_shape) - 1
        result_type = self._type_of(expr)

        # Compute result MLIR types
        mlir_result = _mlir_type(result_type)
        if isinstance(result_type, ScalarType):
            val_result_type = ScalarType("f32")
        else:
            val_result_type = ArrayType("f32", result_type.dims)
        mlir_val_result = _mlir_type(val_result_type)

        # Init values for reduce
        init_val = self._fresh()
        self._emit(f"{init_val} = stablehlo.constant dense<0xFF800000> : tensor<f32>")
        init_idx = self._fresh()
        self._emit(f"{init_idx} = stablehlo.constant dense<0> : tensor<i32>")

        # Iota for indices along last dim
        iota_type_str = f"tensor<{logits_shape_str}xi32>"
        iota_var = self._fresh()
        self._emit(f"{iota_var} = stablehlo.iota dim = {last_dim} : {iota_type_str}")

        # Variadic reduce with argmax body
        result_var = self._fresh()
        self._emit(
            f"{result_var}:2 = stablehlo.reduce({perturbed} init: {init_val}, {iota_var} init: {init_idx}) "
            f"across dimensions = [{last_dim}] "
            f": ({logits_f32_type}, {iota_type_str}, tensor<f32>, tensor<i32>) "
            f"-> ({mlir_val_result}, {mlir_result})"
        )
        self._indent += 1
        a_val = self._fresh()
        b_val = self._fresh()
        a_idx = self._fresh()
        b_idx = self._fresh()
        self._emit(f"reducer({a_val}: tensor<f32>, {b_val}: tensor<f32>, "
                   f"{a_idx}: tensor<i32>, {b_idx}: tensor<i32>) {{")
        self._indent += 1
        cmp_var = self._fresh()
        self._emit(f"{cmp_var} = stablehlo.compare GT, {a_val}, {b_val}, FLOAT "
                   f": (tensor<f32>, tensor<f32>) -> tensor<i1>")
        sel_val = self._fresh()
        self._emit(f"{sel_val} = stablehlo.select {cmp_var}, {a_val}, {b_val} "
                   f": (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>")
        sel_idx = self._fresh()
        self._emit(f"{sel_idx} = stablehlo.select {cmp_var}, {a_idx}, {b_idx} "
                   f": (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>")
        self._emit(f"stablehlo.return {sel_val}, {sel_idx} : tensor<f32>, tensor<i32>")
        self._indent -= 1
        self._emit("}")
        self._indent -= 1

        return f"{result_var}#1"

    def _gen_rng_truncated_normal(self, expr: CallExpr, env: dict[str, str]) -> str:
        """random.truncated_normal(key, lo, hi, d1, ...) -> f32[d1, ...].

        Generate normal(0, 1) values via Box-Muller and clamp to [lo, hi].
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        lo_ssa = self._gen_expr(expr.args[1], env)
        hi_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        numel = 1
        for d in shape:
            numel *= d

        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"
        flat_f32 = f"tensor<{numel}xf32>"

        z_flat = self._gen_box_muller(key_ssa, numel)

        # Reshape from flat to target shape if needed
        if len(shape) == 1:
            normal_vals = z_flat
        else:
            normal_vals = self._fresh()
            self._emit(f"{normal_vals} = stablehlo.reshape {z_flat} : ({flat_f32}) -> {f32_type}")

        # Clamp to [lo, hi]
        lo_bc = self._fresh()
        self._emit(f"{lo_bc} = stablehlo.broadcast_in_dim {lo_ssa}, dims = [] : (tensor<f32>) -> {f32_type}")
        hi_bc = self._fresh()
        self._emit(f"{hi_bc} = stablehlo.broadcast_in_dim {hi_ssa}, dims = [] : (tensor<f32>) -> {f32_type}")

        clamped = self._fresh()
        self._emit(f"{clamped} = stablehlo.clamp {lo_bc}, {normal_vals}, {hi_bc} : {f32_type}")
        return clamped

    def _gen_rng_exponential(self, expr: CallExpr, env: dict[str, str]) -> str:
        """random.exponential(key, d1, ...) -> f32[d1, ...].

        Exponential(rate=1) = -log(uniform(0, 1)).
        """
        key_ssa = self._gen_expr(expr.args[0], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"

        # Generate uniform [0, 1)
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, shape)
        base_uniform = self._gen_bits_to_uniform(bits, shape)

        # Clamp away from 0 for log safety
        eps_s = self._fresh()
        self._emit(f"{eps_s} = stablehlo.constant dense<1.000000e-07> : tensor<f32>")
        eps = self._fresh()
        dims = "" if not shape else "dims = []"
        self._emit(f"{eps} = stablehlo.broadcast_in_dim {eps_s}, {dims} : (tensor<f32>) -> {f32_type}")
        u_safe = self._fresh()
        self._emit(f"{u_safe} = stablehlo.maximum {base_uniform}, {eps} : {f32_type}")

        # -log(u)
        log_u = self._fresh()
        self._emit(f"{log_u} = stablehlo.log {u_safe} : {f32_type}")
        result = self._fresh()
        self._emit(f"{result} = stablehlo.negate {log_u} : {f32_type}")
        return result

    def _gen_rng_randint(self, expr: CallExpr, env: dict[str, str]) -> str:
        """random.randint(key, low, high, d1, ...) -> i32[d1, ...].

        cast(floor(uniform(0, 1) * cast(high - low, f32)) + cast(low, f32), i32)
        """
        key_ssa = self._gen_expr(expr.args[0], env)
        low_ssa = self._gen_expr(expr.args[1], env)
        high_ssa = self._gen_expr(expr.args[2], env)

        result_type = self._type_of(expr)
        assert isinstance(result_type, ArrayType)
        shape = tuple(result_type.dims)
        shape_str = "x".join(str(d) for d in shape)
        f32_type = f"tensor<{shape_str}xf32>"
        i32_type = f"tensor<{shape_str}xi32>"

        # Generate uniform [0, 1)
        key_u64 = self._gen_key_to_u64(key_ssa)
        _, bits = self._gen_rng_bits(key_u64, shape)
        base_uniform = self._gen_bits_to_uniform(bits, shape)

        dims = "dims = []"

        # range = high - low (scalar i32)
        range_i32 = self._fresh()
        self._emit(f"{range_i32} = stablehlo.subtract {high_ssa}, {low_ssa} : tensor<i32>")

        # cast range to f32
        range_f32 = self._fresh()
        self._emit(f"{range_f32} = stablehlo.convert {range_i32} : (tensor<i32>) -> tensor<f32>")

        # broadcast range to output shape
        range_bc = self._fresh()
        self._emit(f"{range_bc} = stablehlo.broadcast_in_dim {range_f32}, {dims} : (tensor<f32>) -> {f32_type}")

        # scaled = uniform * range
        scaled = self._fresh()
        self._emit(f"{scaled} = stablehlo.multiply {base_uniform}, {range_bc} : {f32_type}")

        # floored = floor(scaled)
        floored = self._fresh()
        self._emit(f"{floored} = stablehlo.floor {scaled} : {f32_type}")

        # cast low to f32 and broadcast
        low_f32 = self._fresh()
        self._emit(f"{low_f32} = stablehlo.convert {low_ssa} : (tensor<i32>) -> tensor<f32>")
        low_bc = self._fresh()
        self._emit(f"{low_bc} = stablehlo.broadcast_in_dim {low_f32}, {dims} : (tensor<f32>) -> {f32_type}")

        # sum = floored + low
        sum_val = self._fresh()
        self._emit(f"{sum_val} = stablehlo.add {floored}, {low_bc} : {f32_type}")

        # cast to i32
        result = self._fresh()
        self._emit(f"{result} = stablehlo.convert {sum_val} : ({f32_type}) -> {i32_type}")
        return result
