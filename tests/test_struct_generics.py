"""Tests for struct type variables, struct arithmetic, and struct-level builtins."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen_stablehlo import StableHLOCodegen


# -- Helpers --

def check(source: str) -> list[str]:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    errors = TypeChecker().check(program)
    return [e.message for e in errors]


def check_ok(source: str):
    errors = check(source)
    assert errors == [], f"Expected no errors, got: {errors}"


def check_error(source: str, expected_substring: str):
    errors = check(source)
    assert errors, f"Expected type error containing '{expected_substring}' but got none"
    msgs = " ".join(errors)
    assert expected_substring in msgs, f"Expected '{expected_substring}' in errors, got: {msgs}"


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    tc = TypeChecker()
    errors = tc.check(program)
    assert errors == [], f"Expected no errors, got: {[e.message for e in errors]}"
    return StableHLOCodegen(program, tc.type_map).generate()


# ────────────────────── Struct Type Variables ──────────────────────


class TestStructTypeVar:
    def test_basic_identity(self):
        check_ok("""
            struct Point { x: f32, y: f32 }
            fn identity(p: T) -> T { p }
            fn main(p: Point) -> Point { identity(p) }
        """)

    def test_two_params_same_typevar(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn add_structs(a: T, b: T) -> T { a + b }
            fn main(a: V, b: V) -> V { add_structs(a, b) }
        """)

    def test_typevar_mismatch_error(self):
        check_error("""
            struct A { x: f32 }
            struct B { y: f32 }
            fn f(a: T, b: T) -> T { a }
            fn main(a: A, b: B) -> A { f(a, b) }
        """, "type variable 'T' bound to")

    def test_non_struct_for_typevar_error(self):
        check_error("""
            struct S { x: f32 }
            fn f(a: T) -> T { a }
            fn main(x: f32) -> f32 { f(x) }
        """, "type variable 'T' expects a struct type")

    def test_multi_char_unknown_type_error(self):
        check_error("""
            fn f(x: Modle) -> Modle { x }
        """, "unknown type")

    def test_multiple_typevars(self):
        check_ok("""
            struct A { x: f32 }
            struct B { y: f32 }
            fn f(a: T, b: U) -> T { a }
            fn main(a: A, b: B) -> A { f(a, b) }
        """)

    def test_typevar_with_scalar_params(self):
        check_ok("""
            struct M { w: f32 }
            fn scale(s: T, lr: f32) -> T { s * lr }
            fn main(m: M, lr: f32) -> M { scale(m, lr) }
        """)

    def test_sgd_update(self):
        check_ok("""
            struct Model { w: f32[4, 4], b: f32[4] }
            fn sgd_update(params: T, grads: T, lr: f32) -> T {
                params - lr * grads
            }
            fn main(m: Model, g: Model, lr: f32) -> Model {
                sgd_update(m, g, lr)
            }
        """)


# ────────────────────── Struct Arithmetic ──────────────────────


class TestStructArithmetic:
    def test_struct_add(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, b: V) -> V { a + b }
        """)

    def test_struct_sub(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, b: V) -> V { a - b }
        """)

    def test_scalar_mul_struct(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, s: f32) -> V { s * a }
        """)

    def test_struct_mul_scalar(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, s: f32) -> V { a * s }
        """)

    def test_struct_div_scalar(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, s: f32) -> V { a / s }
        """)

    def test_scalar_add_struct(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, s: f32) -> V { s + a }
        """)

    def test_struct_negate(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V) -> V { -a }
        """)

    def test_nested_struct_arithmetic(self):
        check_ok("""
            struct Inner { x: f32 }
            struct Outer { a: Inner, b: f32 }
            fn f(a: Outer, b: Outer) -> Outer { a + b }
        """)

    def test_compound_expr(self):
        check_ok("""
            struct Model { w: f32[4, 4], b: f32[4] }
            fn f(params: Model, grads: Model, lr: f32) -> Model {
                params - lr * grads
            }
        """)

    def test_struct_mul_struct(self):
        """struct * struct is allowed (element-wise, e.g., grads * grads for Adam)."""
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, b: V) -> V { a * b }
        """)

    def test_struct_div_struct(self):
        """struct / struct is allowed (element-wise)."""
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(a: V, b: V) -> V { a / b }
        """)

    def test_struct_add_mismatched_types_error(self):
        check_error("""
            struct A { x: f32 }
            struct B { y: f32 }
            fn f(a: A, b: B) -> A { a + b }
        """, "mismatched struct types")

    def test_struct_with_non_numeric_field_error(self):
        check_error("""
            struct S { x: f32, flag: bool }
            fn f(a: S, b: S) -> S { a + b }
        """, "non-numeric fields")


# ────────────────────── Codegen ──────────────────────


class TestStructArithmeticCodegen:
    def test_struct_add_compiles(self):
        mlir = codegen("""
            struct V { x: f32, y: f32 }
            fn f(a: V, b: V) -> V { a + b }
        """)
        assert "stablehlo.add" in mlir
        assert "stablehlo.tuple" in mlir

    def test_scalar_mul_struct_compiles(self):
        mlir = codegen("""
            struct V { x: f32, y: f32 }
            fn f(a: V, s: f32) -> V { s * a }
        """)
        assert "stablehlo.multiply" in mlir
        assert "stablehlo.tuple" in mlir

    def test_struct_negate_compiles(self):
        mlir = codegen("""
            struct V { x: f32, y: f32 }
            fn f(a: V) -> V { -a }
        """)
        assert "stablehlo.negate" in mlir
        assert "stablehlo.tuple" in mlir

    def test_generic_sgd_compiles(self):
        mlir = codegen("""
            struct Model { w: f32[4, 4], b: f32[4] }
            fn sgd_update(params: T, grads: T, lr: f32) -> T {
                params - lr * grads
            }
            fn main(m: Model, g: Model, lr: f32) -> Model {
                sgd_update(m, g, lr)
            }
        """)
        assert "sgd_update$Model" in mlir
        assert "stablehlo.subtract" in mlir
        assert "stablehlo.multiply" in mlir

    def test_nested_struct_codegen(self):
        mlir = codegen("""
            struct Inner { x: f32 }
            struct Outer { a: Inner, b: f32 }
            fn f(a: Outer, b: Outer) -> Outer { a - b }
        """)
        assert "stablehlo.subtract" in mlir
        assert "get_tuple_element" in mlir


# ────────────────────── Struct-Level Builtins ──────────────────────


class TestStructBuiltins:
    def test_sqrt_struct(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(v: V) -> V { sqrt(v) }
        """)

    def test_exp_struct(self):
        check_ok("""
            struct V { x: f32, y: f32 }
            fn f(v: V) -> V { exp(v) }
        """)

    def test_sqrt_struct_with_arrays(self):
        check_ok("""
            struct M { w: f32[4, 4], b: f32[4] }
            fn f(m: M) -> M { sqrt(m) }
        """)

    def test_sqrt_nested_struct(self):
        check_ok("""
            struct Inner { x: f32 }
            struct Outer { a: Inner, b: f32 }
            fn f(o: Outer) -> Outer { sqrt(o) }
        """)

    def test_sqrt_struct_with_int_field_error(self):
        check_error("""
            struct S { x: f32, n: i32 }
            fn f(s: S) -> S { sqrt(s) }
        """, "all leaf fields must be float")

    def test_sqrt_struct_codegen(self):
        mlir = codegen("""
            struct V { x: f32, y: f32 }
            fn f(v: V) -> V { sqrt(v) }
        """)
        assert "stablehlo.sqrt" in mlir
        assert "stablehlo.tuple" in mlir

    def test_struct_builtin_in_generic(self):
        """sqrt in a generic function — the Adam use case."""
        check_ok("""
            struct M { w: f32[4, 4], b: f32[4] }
            fn apply_sqrt(v: T) -> T { sqrt(v) }
            fn main(m: M) -> M { apply_sqrt(m) }
        """)


# ────────────────────── cos/sin builtins ──────────────────────


class TestCosSin:
    def test_cos_scalar(self):
        check_ok("""
            fn f(x: f32) -> f32 { cos(x) }
        """)

    def test_sin_scalar(self):
        check_ok("""
            fn f(x: f32) -> f32 { sin(x) }
        """)

    def test_cos_array(self):
        check_ok("""
            fn f(x: f32[4]) -> f32[4] { cos(x) }
        """)

    def test_cos_codegen(self):
        mlir = codegen("""
            fn f(x: f32) -> f32 { cos(x) }
        """)
        assert "stablehlo.cosine" in mlir

    def test_sin_codegen(self):
        mlir = codegen("""
            fn f(x: f32) -> f32 { sin(x) }
        """)
        assert "stablehlo.sine" in mlir


# ────────────────────── Stdlib optim ──────────────────────


class TestStdlibOptim:
    def _compile(self, source: str) -> str:
        from maomi.cli import compile_source
        result = compile_source(source, filename="/tmp/test.mao")
        return result.mlir_text

    def test_sgd_import(self):
        mlir = self._compile("""
            from optim import { sgd_update };
            struct M { w: f32[4, 4], b: f32[4] }
            fn f(m: M, g: M, lr: f32) -> M { sgd_update(m, g, lr) }
        """)
        assert "sgd_update" in mlir

    def test_adam_import(self):
        mlir = self._compile("""
            from optim import { adam_update };
            struct M { w: f32[4, 4], b: f32[4] }
            fn f(m: M, g: M, mm: M, vv: M, step: i32, lr: f32) -> M {
                adam_update(m, g, mm, vv, step, lr, 0.9, 0.999, 0.00000001)
            }
        """)
        assert "adam_update" in mlir

    def test_adam_m_v_import(self):
        mlir = self._compile("""
            from optim import { adam_m_update, adam_v_update };
            struct M { w: f32[4, 4], b: f32[4] }
            fn f(m: M, g: M) -> M { adam_m_update(m, g, 0.9) }
            fn g(v: M, g2: M) -> M { adam_v_update(v, g2, 0.999) }
        """)
        assert "adam_m_update" in mlir

    def test_linear_decay_import(self):
        mlir = self._compile("""
            from optim import { linear_decay };
            fn f(step: i32) -> f32 { linear_decay(step, 0.01, 0.0001, 1000) }
        """)
        assert "linear_decay" in mlir

    def test_cosine_decay_import(self):
        mlir = self._compile("""
            from optim import { cosine_decay };
            fn f(step: i32) -> f32 { cosine_decay(step, 0.01, 1000) }
        """)
        assert "cosine_decay" in mlir
        assert "stablehlo.cosine" in mlir

    def test_sgd_with_grad(self):
        """Full training step: loss → grad → sgd_update."""
        mlir = self._compile("""
            from optim import { sgd_update };
            struct Model { w: f32[4, 4], b: f32[4] }
            fn loss(m: Model, x: f32[32, 4], y: f32[32, 4]) -> f32 {
                let pred = x @ m.w + m.b;
                mean((pred - y) * (pred - y))
            }
            fn train_step(m: Model, x: f32[32, 4], y: f32[32, 4]) -> Model {
                let g = grad(loss(m, x, y), m);
                sgd_update(m, g, 0.01)
            }
        """)
        assert "sgd_update" in mlir

    def test_qualified_import(self):
        mlir = self._compile("""
            import optim;
            struct M { w: f32 }
            fn f(m: M, g: M) -> M { optim.sgd_update(m, g, 0.01) }
        """)
        assert "sgd_update" in mlir
