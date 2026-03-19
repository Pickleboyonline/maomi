"""Tests for config() requiring typed let bindings."""

import unittest

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.cli import compile_source


def _type_check(source: str):
    """Type-check source, return list of error messages."""
    tokens = Lexer(source, filename="<test>").tokenize()
    program = Parser(tokens, filename="<test>").parse()
    checker = TypeChecker(filename="<test>")
    errors = checker.check(program)
    return [e.message for e in errors]


class TestConfigTypedLet(unittest.TestCase):
    # -- Happy paths --

    def test_config_f32(self):
        errors = _type_check('fn f() -> f32 { let lr: f32 = config("lr"); lr }')
        self.assertEqual(errors, [])

    def test_config_i32(self):
        errors = _type_check('fn f() -> i32 { let bs: i32 = config("batch_size"); bs }')
        self.assertEqual(errors, [])

    def test_config_f64(self):
        errors = _type_check('fn f() -> f64 { let x: f64 = config("x"); x }')
        self.assertEqual(errors, [])

    def test_config_i64(self):
        errors = _type_check('fn f() -> i64 { let x: i64 = config("x"); x }')
        self.assertEqual(errors, [])

    def test_config_bool(self):
        errors = _type_check('fn f() -> bool { let x: bool = config("flag"); x }')
        self.assertEqual(errors, [])

    # -- Error: missing type annotation --

    def test_config_no_annotation(self):
        errors = _type_check('fn f() -> f32 { let lr = config("lr"); lr }')
        self.assertTrue(len(errors) >= 1)
        self.assertIn("type annotation", errors[0])

    # -- Error: bare config() outside let --

    def test_config_in_call_arg(self):
        errors = _type_check('fn g(x: f32) -> f32 { x } fn f() -> f32 { g(config("lr")) }')
        self.assertEqual(len(errors), 1)
        self.assertIn("typed let binding", errors[0])

    def test_config_in_binop(self):
        errors = _type_check('fn f() -> f32 { config("lr") + 1.0 }')
        self.assertEqual(len(errors), 1)
        self.assertIn("typed let binding", errors[0])

    def test_config_bare_expr(self):
        errors = _type_check('fn f() -> f32 { config("lr") }')
        self.assertEqual(len(errors), 1)
        self.assertIn("typed let binding", errors[0])

    # -- Error: non-scalar type --

    def test_config_array_type(self):
        errors = _type_check('fn f() -> f32 { let x: f32[3] = config("x"); x }')
        self.assertTrue(len(errors) >= 1)
        self.assertIn("scalar", errors[0])

    # -- Error: bad arguments --

    def test_config_no_args(self):
        errors = _type_check('fn f() -> f32 { let x: f32 = config(); x }')
        self.assertTrue(len(errors) >= 1)
        self.assertIn("one string argument", errors[0])

    def test_config_int_arg(self):
        errors = _type_check('fn f() -> f32 { let x: f32 = config(42); x }')
        self.assertTrue(len(errors) >= 1)
        self.assertIn("one string argument", errors[0])

    # -- End-to-end with config dict --

    def test_config_substitution_f32(self):
        source = 'fn f() -> f32 { let lr: f32 = config("lr"); lr }'
        result = compile_source(source, config={"lr": 0.01})
        self.assertIn("stablehlo", result.mlir_text.lower())

    def test_config_substitution_i32(self):
        source = 'fn f() -> i32 { let bs: i32 = config("batch_size"); bs }'
        result = compile_source(source, config={"batch_size": 32})
        self.assertIn("stablehlo", result.mlir_text.lower())


if __name__ == "__main__":
    unittest.main()
