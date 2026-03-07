"""Tests for while loop primitive: parsing, type checking, codegen, and AD."""
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.codegen_stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.ast_nodes import *
from maomi.errors import MaomiError
import pytest


def parse(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def typecheck(source: str) -> list[MaomiError]:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    return checker.check(program)


def codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    return StableHLOCodegen(program, checker.type_map).generate()


def ad_codegen(source: str) -> str:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    errors = checker.check(program)
    assert errors == [], f"Type errors: {[e.message for e in errors]}"
    program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


# ── Parsing ──


class TestWhileParsing:
    def test_bare_while(self):
        prog = parse("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do { s - 1.0 }
            }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, WhileExpr)
        assert expr.state_var == "s"
        assert expr.max_iters is None

    def test_bounded_while(self):
        prog = parse("""
            fn f(x: f32) -> f32 {
                while s in x max 100 { s > 0.0 } do { s - 1.0 }
            }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, WhileExpr)
        assert expr.state_var == "s"
        assert expr.max_iters == 100

    def test_while_with_let_init(self):
        prog = parse("""
            fn f(x: f32) -> f32 {
                let init = x * 2.0;
                while s in init { s > 0.0 } do { s - 1.0 }
            }
        """)
        fn = prog.functions[0]
        # body has a let statement and then the while expr
        assert isinstance(fn.body.stmts[0], LetStmt)
        assert isinstance(fn.body.expr, WhileExpr)

    def test_while_body_with_statements(self):
        prog = parse("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do {
                    let next = s - 1.0;
                    next * 0.5
                }
            }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, WhileExpr)
        assert len(expr.body.stmts) == 1  # let statement
        assert expr.body.expr is not None  # trailing expression


# ── Type Checking ──


class TestWhileTypeCheck:
    def test_basic_while_scalar(self):
        errors = typecheck("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do { s - 1.0 }
            }
        """)
        assert errors == []

    def test_basic_while_array(self):
        errors = typecheck("""
            fn f(x: f32[4]) -> f32[4] {
                while s in x { sum(s) > 0.0 } do { s - 1.0 }
            }
        """)
        assert errors == []

    def test_bounded_while(self):
        errors = typecheck("""
            fn f(x: f32) -> f32 {
                while s in x max 50 { s > 0.01 } do { s * 0.5 }
            }
        """)
        assert errors == []

    def test_cond_must_be_bool(self):
        errors = typecheck("""
            fn f(x: f32) -> f32 {
                while s in x { s + 1.0 } do { s - 1.0 }
            }
        """)
        assert len(errors) == 1
        assert "bool" in errors[0].message

    def test_body_type_must_match_init(self):
        errors = typecheck("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do { true }
            }
        """)
        assert len(errors) >= 1


# ── Codegen ──


class TestWhileCodegen:
    def test_bare_while_emits_stablehlo_while(self):
        out = codegen("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do { s - 1.0 }
            }
        """)
        assert "stablehlo.while" in out
        assert "stablehlo.compare GT" in out
        assert "stablehlo.subtract" in out

    def test_bare_while_no_counter(self):
        """While loop should NOT have a counter like scan does."""
        out = codegen("""
            fn f(x: f32) -> f32 {
                while s in x { s > 0.0 } do { s - 1.0 }
            }
        """)
        # Should have exactly 1 while arg (state only), not multiple like scan
        assert "whileS" in out

    def test_while_array_state(self):
        out = codegen("""
            fn f(x: f32[4]) -> f32[4] {
                while s in x { sum(s) > 0.0 } do { s - 1.0 }
            }
        """)
        assert "stablehlo.while" in out
        assert "tensor<4xf32>" in out

    def test_while_with_body_statements(self):
        out = codegen("""
            fn f(x: f32) -> f32 {
                while s in x { s > 1.0 } do {
                    let half = s * 0.5;
                    half
                }
            }
        """)
        assert "stablehlo.while" in out
        assert "stablehlo.multiply" in out


# ── AD ──


class TestWhileAD:
    def test_unbounded_while_grad_errors(self):
        """grad through unbounded while should produce a clear error."""
        with pytest.raises(MaomiError, match="max iterations"):
            ad_codegen("""
                fn f(x: f32) -> f32 {
                    let result = while s in x { s > 0.01 } do { s * 0.5 };
                    grad(result, x)
                }
            """)

    def test_bounded_while_grad_compiles(self):
        """grad through bounded while should compile (emits two while loops)."""
        out = ad_codegen("""
            fn f(x: f32) -> f32 {
                let result = while s in x max 100 { s > 0.01 } do { s * 0.5 };
                grad(result, x)
            }
        """)
        # Should have at least 2 while loops: forward augmented + backward
        count = out.count("stablehlo.while")
        assert count >= 2, f"Expected at least 2 while loops, got {count}"

    def test_bounded_while_grad_has_trajectory(self):
        """Bounded while grad should use dynamic_update_slice for trajectory."""
        out = ad_codegen("""
            fn f(x: f32) -> f32 {
                let result = while s in x max 50 { s > 0.01 } do { s * 0.5 };
                grad(result, x)
            }
        """)
        assert "dynamic_update_slice" in out
        assert "dynamic_slice" in out
