from pathlib import Path
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.ast_nodes import *
from maomi.errors import ParseError
import pytest


def parse(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def parse_expr(source: str) -> Expr:
    """Parse a single expression by wrapping it in a function."""
    prog = parse(f"fn _() -> f32 {{ {source} }}")
    block = prog.functions[0].body
    assert block.expr is not None
    return block.expr


class TestFnDef:
    def test_simple_function(self):
        prog = parse("fn add(a: f32, b: f32) -> f32 { a + b }")
        assert len(prog.functions) == 1
        fn = prog.functions[0]
        assert fn.name == "add"
        assert len(fn.params) == 2
        assert fn.params[0].name == "a"
        assert fn.params[0].type_annotation.base == "f32"
        assert fn.params[0].type_annotation.dims is None
        assert fn.return_type.base == "f32"

    def test_no_params(self):
        prog = parse("fn zero() -> f32 { 0.0 }")
        assert len(prog.functions[0].params) == 0

    def test_array_type_params(self):
        prog = parse("fn f(x: f32[B, 128]) -> f32[B, 64] { x }")
        fn = prog.functions[0]
        assert fn.params[0].type_annotation.dims is not None
        assert len(fn.params[0].type_annotation.dims) == 2
        assert fn.params[0].type_annotation.dims[0].value == "B"
        assert fn.params[0].type_annotation.dims[1].value == 128
        assert fn.return_type.dims[0].value == "B"
        assert fn.return_type.dims[1].value == 64

    def test_effect_annotation_rejected(self):
        with pytest.raises(Exception):
            parse("fn f() -> f32 ! State { 0.0 }")

    def test_multiple_functions(self):
        prog = parse("fn a() -> f32 { 1.0 }\nfn b() -> f32 { 2.0 }")
        assert len(prog.functions) == 2
        assert prog.functions[0].name == "a"
        assert prog.functions[1].name == "b"


class TestBlock:
    def test_trailing_expr_only(self):
        prog = parse("fn f() -> f32 { 42 }")
        block = prog.functions[0].body
        assert len(block.stmts) == 0
        assert isinstance(block.expr, IntLiteral)

    def test_stmt_then_trailing_expr(self):
        prog = parse("fn f() -> f32 { let x = 1; x }")
        block = prog.functions[0].body
        assert len(block.stmts) == 1
        assert isinstance(block.stmts[0], LetStmt)
        assert isinstance(block.expr, Identifier)

    def test_stmt_only_no_trailing(self):
        prog = parse("fn f() -> f32 { 1; }")
        block = prog.functions[0].body
        assert len(block.stmts) == 1
        assert isinstance(block.stmts[0], ExprStmt)
        assert block.expr is None

    def test_multiple_stmts_with_trailing(self):
        prog = parse("fn f() -> f32 { let a = 1; let b = 2; a + b }")
        block = prog.functions[0].body
        assert len(block.stmts) == 2
        assert isinstance(block.expr, BinOp)


class TestLetStmt:
    def test_basic_let(self):
        prog = parse("fn f() -> f32 { let x = 1; x }")
        stmt = prog.functions[0].body.stmts[0]
        assert isinstance(stmt, LetStmt)
        assert stmt.name == "x"
        assert stmt.type_annotation is None
        assert isinstance(stmt.value, IntLiteral)

    def test_let_with_type(self):
        prog = parse("fn f() -> f32 { let x: f32 = 1.0; x }")
        stmt = prog.functions[0].body.stmts[0]
        assert isinstance(stmt, LetStmt)
        assert stmt.type_annotation is not None
        assert stmt.type_annotation.base == "f32"

    def test_let_with_array_type(self):
        prog = parse("fn f() -> f32 { let x: f32[N] = y; x }")
        stmt = prog.functions[0].body.stmts[0]
        assert stmt.type_annotation.dims[0].value == "N"


class TestOperatorPrecedence:
    def test_addition_multiplication(self):
        """a + b * c → BinOp(+, a, BinOp(*, b, c))"""
        expr = parse_expr("a + b * c")
        assert isinstance(expr, BinOp)
        assert expr.op == "+"
        assert isinstance(expr.left, Identifier)
        assert isinstance(expr.right, BinOp)
        assert expr.right.op == "*"

    def test_multiplication_addition(self):
        """a * b + c → BinOp(+, BinOp(*, a, b), c)"""
        expr = parse_expr("a * b + c")
        assert isinstance(expr, BinOp)
        assert expr.op == "+"
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == "*"

    def test_matmul_plus(self):
        """x @ w + b → BinOp(+, BinOp(@, x, w), b)"""
        expr = parse_expr("x @ w + b")
        assert isinstance(expr, BinOp)
        assert expr.op == "+"
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == "@"

    def test_power_right_associative(self):
        """2 ** 3 ** 4 → BinOp(**, 2, BinOp(**, 3, 4))"""
        expr = parse_expr("2 ** 3 ** 4")
        assert isinstance(expr, BinOp)
        assert expr.op == "**"
        assert isinstance(expr.left, IntLiteral)
        assert expr.left.value == 2
        assert isinstance(expr.right, BinOp)
        assert expr.right.op == "**"

    def test_parentheses_override(self):
        """(a + b) * c → BinOp(*, BinOp(+, a, b), c)"""
        expr = parse_expr("(a + b) * c")
        assert isinstance(expr, BinOp)
        assert expr.op == "*"
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == "+"

    def test_unary_minus(self):
        expr = parse_expr("-x")
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"
        assert isinstance(expr.operand, Identifier)

    def test_comparison(self):
        expr = parse_expr("x > 0.0")
        assert isinstance(expr, BinOp)
        assert expr.op == ">"
        assert isinstance(expr.right, FloatLiteral)


class TestIfExpr:
    def test_simple_if(self):
        expr = parse_expr("if x > 0.0 { x } else { 0.0 }")
        assert isinstance(expr, IfExpr)
        assert isinstance(expr.condition, BinOp)
        assert expr.then_block.expr is not None
        assert expr.else_block.expr is not None


class TestCallExpr:
    def test_simple_call(self):
        expr = parse_expr("foo(x)")
        assert isinstance(expr, CallExpr)
        assert expr.callee == "foo"
        assert len(expr.args) == 1

    def test_multi_arg_call(self):
        expr = parse_expr("linear(x, w, b)")
        assert isinstance(expr, CallExpr)
        assert expr.callee == "linear"
        assert len(expr.args) == 3

    def test_no_arg_call(self):
        expr = parse_expr("foo()")
        assert isinstance(expr, CallExpr)
        assert len(expr.args) == 0

    def test_nested_call(self):
        expr = parse_expr("relu(linear(x, w, b))")
        assert isinstance(expr, CallExpr)
        assert expr.callee == "relu"
        assert len(expr.args) == 1
        inner = expr.args[0]
        assert isinstance(inner, CallExpr)
        assert inner.callee == "linear"


class TestLiterals:
    def test_int(self):
        expr = parse_expr("42")
        assert isinstance(expr, IntLiteral)
        assert expr.value == 42

    def test_float(self):
        expr = parse_expr("3.14")
        assert isinstance(expr, FloatLiteral)
        assert expr.value == 3.14

    def test_true(self):
        expr = parse_expr("true")
        assert isinstance(expr, BoolLiteral)
        assert expr.value is True

    def test_false(self):
        expr = parse_expr("false")
        assert isinstance(expr, BoolLiteral)
        assert expr.value is False


class TestErrors:
    def test_missing_brace(self):
        with pytest.raises(ParseError):
            parse("fn f() -> f32 { x")

    def test_missing_semicolon(self):
        with pytest.raises(ParseError):
            parse("fn f() -> f32 { let x = 1 x }")

    def test_unexpected_token(self):
        with pytest.raises(ParseError):
            parse("fn f() -> f32 { ; }")


class TestFixtures:
    """Parse real .mao files to verify end-to-end."""

    fixtures_dir = Path(__file__).parent / "fixtures"

    def test_linear(self):
        source = (self.fixtures_dir / "linear.mao").read_text()
        prog = parse(source)
        assert len(prog.functions) == 1
        assert prog.functions[0].name == "linear"

    def test_relu(self):
        source = (self.fixtures_dir / "relu.mao").read_text()
        prog = parse(source)
        assert len(prog.functions) == 1
        fn = prog.functions[0]
        assert fn.name == "relu"
        assert isinstance(fn.body.expr, IfExpr)

    def test_mlp(self):
        source = (self.fixtures_dir / "mlp.mao").read_text()
        prog = parse(source)
        assert len(prog.functions) == 3
        names = [fn.name for fn in prog.functions]
        assert names == ["linear", "relu2d", "mlp"]


class TestScan:
    def test_basic_scan(self):
        prog = parse("""
            fn rnn(xs: f32[10, 4], h0: f32[8], w: f32[4, 8]) -> f32[10, 8] {
                scan (h, x) in (h0, xs) {
                    tanh(h + x @ w)
                }
            }
        """)
        fn = prog.functions[0]
        assert isinstance(fn.body.expr, ScanExpr)
        scan = fn.body.expr
        assert scan.carry_var == "h"
        assert scan.elem_vars == ["x"]

    def test_scan_with_let(self):
        prog = parse("""
            fn f(xs: f32[5], init: f32) -> f32[5] {
                scan (acc, x) in (init, xs) {
                    let next = acc + x;
                    next
                }
            }
        """)
        fn = prog.functions[0]
        assert isinstance(fn.body.expr, ScanExpr)

    def test_multi_sequence_scan(self):
        prog = parse("""
            fn f(xs: f32[5], ys: f32[5]) -> f32[5] {
                scan (acc, (x, y)) in (0.0, (xs, ys)) {
                    acc + x * y
                }
            }
        """)
        fn = prog.functions[0]
        scan = fn.body.expr
        assert isinstance(scan, ScanExpr)
        assert scan.carry_var == "acc"
        assert scan.elem_vars == ["x", "y"]
        assert len(scan.sequences) == 2


class TestMap:
    def test_basic_map(self):
        prog = parse("""
            fn batch_relu(xs: f32[32, 64]) -> f32[32, 64] {
                map x in xs {
                    if x > 0.0 { x } else { 0.0 }
                }
            }
        """)
        fn = prog.functions[0]
        assert isinstance(fn.body.expr, MapExpr)
        m = fn.body.expr
        assert m.elem_var == "x"
        assert isinstance(m.sequence, Identifier)

    def test_map_simple(self):
        prog = parse("""
            fn double(xs: f32[10]) -> f32[10] {
                map x in xs { x * 2.0 }
            }
        """)
        fn = prog.functions[0]
        assert isinstance(fn.body.expr, MapExpr)


class TestGrad:
    def test_basic_grad(self):
        expr = parse_expr("grad(x * x, x)")
        assert isinstance(expr, GradExpr)
        assert expr.wrt == "x"
        assert isinstance(expr.expr, BinOp)

    def test_grad_complex(self):
        prog = parse("""
            fn train(x: f32[4], w: f32[4, 2]) -> f32[4, 2] {
                let loss = mean((x @ w) ** 2);
                grad(loss, w)
            }
        """)
        fn = prog.functions[0]
        assert isinstance(fn.body.expr, GradExpr)
        assert fn.body.expr.wrt == "w"


class TestStructParsing:
    def test_struct_def(self):
        prog = parse("struct Point { x: f32, y: f32 } fn f() -> f32 { 0.0 }")
        assert len(prog.struct_defs) == 1
        sd = prog.struct_defs[0]
        assert sd.name == "Point"
        assert len(sd.fields) == 2
        assert sd.fields[0][0] == "x"
        assert sd.fields[1][0] == "y"

    def test_struct_literal(self):
        prog = parse("struct Point { x: f32, y: f32 } fn f() -> Point { Point { x: 1.0, y: 2.0 } }")
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, StructLiteral)
        assert expr.name == "Point"
        assert len(expr.fields) == 2

    def test_field_access(self):
        prog = parse("struct Point { x: f32 } fn f(p: Point) -> f32 { p.x }")
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, FieldAccess)
        assert expr.field == "x"

    def test_nested_field_access(self):
        prog = parse("""
            struct Inner { w: f32 }
            struct Outer { pcn: Inner }
            fn f(b: Outer) -> f32 { b.pcn.w }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, FieldAccess)
        assert expr.field == "w"
        assert isinstance(expr.object, FieldAccess)
        assert expr.object.field == "pcn"

    def test_with_expr(self):
        prog = parse("struct Point { x: f32, y: f32 } fn f(p: Point) -> Point { p with { x = 1.0 } }")
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, WithExpr)
        assert len(expr.updates) == 1
        assert expr.updates[0][0] == ["x"]

    def test_with_nested_path(self):
        prog = parse("""
            struct Inner { w: f32 }
            struct Outer { pcn: Inner }
            fn f(b: Outer) -> Outer { b with { pcn.w = 1.0 } }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, WithExpr)
        assert expr.updates[0][0] == ["pcn", "w"]

    def test_struct_type_annotation(self):
        prog = parse("struct Point { x: f32 } fn f(p: Point) -> Point { p }")
        fn = prog.functions[0]
        assert fn.params[0].type_annotation.base == "Point"
        assert fn.return_type.base == "Point"


class TestPipe:
    def test_pipe_bare_function(self):
        expr = parse_expr("x |> f")
        assert isinstance(expr, CallExpr)
        assert expr.callee == "f"
        assert len(expr.args) == 1
        assert isinstance(expr.args[0], Identifier)
        assert expr.args[0].name == "x"

    def test_pipe_with_args(self):
        expr = parse_expr("x |> f(y)")
        assert isinstance(expr, CallExpr)
        assert expr.callee == "f"
        assert len(expr.args) == 2
        assert expr.args[0].name == "x"
        assert expr.args[1].name == "y"

    def test_pipe_chain(self):
        expr = parse_expr("x |> f |> g")
        # g(f(x))
        assert isinstance(expr, CallExpr)
        assert expr.callee == "g"
        assert len(expr.args) == 1
        inner = expr.args[0]
        assert isinstance(inner, CallExpr)
        assert inner.callee == "f"
        assert len(inner.args) == 1
        assert inner.args[0].name == "x"

    def test_pipe_chain_with_args(self):
        expr = parse_expr("x |> f(y) |> g(z)")
        # g(f(x, y), z)
        assert isinstance(expr, CallExpr)
        assert expr.callee == "g"
        assert len(expr.args) == 2
        inner = expr.args[0]
        assert isinstance(inner, CallExpr)
        assert inner.callee == "f"
        assert len(inner.args) == 2

    def test_pipe_precedence_below_arithmetic(self):
        expr = parse_expr("a + b |> f")
        # f(a + b)
        assert isinstance(expr, CallExpr)
        assert expr.callee == "f"
        assert isinstance(expr.args[0], BinOp)
        assert expr.args[0].op == "+"

    def test_pipe_in_full_function(self):
        prog = parse("""
            fn pipeline(x: f32, y: f32) -> f32 {
                x |> relu |> linear(y)
            }
        """)
        fn = prog.functions[0]
        expr = fn.body.expr
        assert isinstance(expr, CallExpr)
        assert expr.callee == "linear"


class TestDocComments:
    def test_fn_with_doc_comment(self):
        prog = parse("/// Add two numbers.\nfn add(a: f32, b: f32) -> f32 { a + b }")
        assert prog.functions[0].doc == "Add two numbers."

    def test_fn_without_doc_comment(self):
        prog = parse("fn add(a: f32, b: f32) -> f32 { a + b }")
        assert prog.functions[0].doc is None

    def test_multiline_doc_comment(self):
        prog = parse("/// Line 1.\n/// Line 2.\nfn f(x: f32) -> f32 { x }")
        assert prog.functions[0].doc == "Line 1.\nLine 2."

    def test_struct_with_doc_comment(self):
        prog = parse("/// A 2D point.\nstruct Point { x: f32, y: f32 }")
        assert prog.struct_defs[0].doc == "A 2D point."

    def test_struct_without_doc_comment(self):
        prog = parse("struct Point { x: f32, y: f32 }")
        assert prog.struct_defs[0].doc is None

    def test_regular_comment_not_captured(self):
        prog = parse("// Just a comment.\nfn f(x: f32) -> f32 { x }")
        assert prog.functions[0].doc is None

    def test_doc_comment_only_attaches_to_next_def(self):
        prog = parse("/// Doc for f.\nfn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }")
        assert prog.functions[0].doc == "Doc for f."
        assert prog.functions[1].doc is None


class TestStringLiteral:
    def test_string_literal_parsed(self):
        expr = parse_expr('"hello"')
        assert isinstance(expr, StringLiteral)
        assert expr.value == "hello"

    def test_string_literal_in_callback(self):
        prog = parse('fn f(x: f32) -> f32 { callback("label", x); x }')
        block = prog.functions[0].body
        stmt = block.stmts[0]
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, CallExpr)
        assert isinstance(call.args[0], StringLiteral)
        assert call.args[0].value == "label"

    def test_empty_string(self):
        expr = parse_expr('""')
        assert isinstance(expr, StringLiteral)
        assert expr.value == ""

    def test_string_span_includes_quotes(self):
        expr = parse_expr('"hi"')
        assert isinstance(expr, StringLiteral)
        assert expr.span.col_end - expr.span.col_start == 4  # "hi" = 4 chars
