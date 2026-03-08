"""Tests for the import/module system."""
import os
import pytest

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.resolver import resolve
from maomi.type_checker import TypeChecker
from maomi.codegen.stablehlo import StableHLOCodegen
from maomi.ad import transform_grad
from maomi.errors import MaomiError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "modules")


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


def _parse_file(path: str):
    with open(path) as f:
        source = f.read()
    tokens = Lexer(source, filename=path).tokenize()
    return Parser(tokens, filename=path).parse()


def _compile_file(path: str) -> str:
    """Full pipeline: parse -> resolve -> typecheck -> AD -> codegen. Returns MLIR."""
    program = _parse_file(path)
    program = resolve(program, path)
    checker = TypeChecker(filename=path)
    errors = checker.check(program)
    assert not errors, f"Type errors: {errors}"
    program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


# ---- Lexer: new tokens ----


class TestLexerModuleTokens:
    def test_import_keyword(self):
        tokens = Lexer("import").tokenize()
        assert tokens[0].type.value == "import"

    def test_from_keyword(self):
        tokens = Lexer("from").tokenize()
        assert tokens[0].type.value == "from"

    def test_as_keyword(self):
        tokens = Lexer("as").tokenize()
        assert tokens[0].type.value == "as"

    def test_dot_token(self):
        tokens = Lexer("math.relu").tokenize()
        assert tokens[0].type.value == "ident"
        assert tokens[1].type.value == "dot"
        assert tokens[2].type.value == "ident"

    def test_string_literal(self):
        tokens = Lexer('"../lib/nn"').tokenize()
        assert tokens[0].type.value == "string_lit"
        assert tokens[0].value == "../lib/nn"

    def test_dot_not_in_float(self):
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type.value == "float_lit"
        assert len(tokens) == 2  # float + EOF


# ---- Parser: import declarations ----


class TestParserImports:
    def test_qualified_import(self):
        tokens = Lexer("import math; fn f() -> f32 { 1.0 }").tokenize()
        program = Parser(tokens).parse()
        assert len(program.imports) == 1
        imp = program.imports[0]
        assert imp.module_path == "math"
        assert imp.alias is None
        assert imp.names is None

    def test_qualified_import_with_alias(self):
        tokens = Lexer('import "lib/nn" as nn; fn f() -> f32 { 1.0 }').tokenize()
        program = Parser(tokens).parse()
        imp = program.imports[0]
        assert imp.module_path == "lib/nn"
        assert imp.alias == "nn"
        assert imp.names is None

    def test_from_import(self):
        tokens = Lexer("from math import { relu, linear }; fn f() -> f32 { 1.0 }").tokenize()
        program = Parser(tokens).parse()
        imp = program.imports[0]
        assert imp.module_path == "math"
        assert imp.names == ["relu", "linear"]

    def test_from_import_path(self):
        tokens = Lexer('from "../lib/nn" as nn import { relu }; fn f() -> f32 { 1.0 }').tokenize()
        program = Parser(tokens).parse()
        imp = program.imports[0]
        assert imp.module_path == "../lib/nn"
        assert imp.alias == "nn"
        assert imp.names == ["relu"]

    def test_multiple_imports(self):
        source = 'import math; from nn import { relu }; fn f() -> f32 { 1.0 }'
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.imports) == 2

    def test_qualified_call_parsing(self):
        tokens = Lexer("fn f(x: f32) -> f32 { math.double(x) }").tokenize()
        program = Parser(tokens).parse()
        from maomi.ast_nodes import CallExpr
        body_expr = program.functions[0].body.expr
        assert isinstance(body_expr, CallExpr)
        assert body_expr.callee == "math.double"

    def test_no_imports_still_works(self):
        tokens = Lexer("fn f() -> f32 { 1.0 }").tokenize()
        program = Parser(tokens).parse()
        assert len(program.imports) == 0
        assert len(program.functions) == 1


# ---- Resolver: integration ----


class TestResolver:
    def test_qualified_import(self):
        path = _fixture("uses_mathlib.mao")
        program = _parse_file(path)
        resolved = resolve(program, path)
        assert resolved.imports == []
        fn_names = {fn.name for fn in resolved.functions}
        assert "mathlib.double" in fn_names
        assert "mathlib.square" in fn_names
        assert "quad" in fn_names

    def test_from_import(self):
        path = _fixture("uses_from.mao")
        program = _parse_file(path)
        resolved = resolve(program, path)
        fn_names = {fn.name for fn in resolved.functions}
        assert "double" in fn_names
        assert "quad" in fn_names

    def test_path_import(self):
        path = _fixture("uses_path.mao")
        program = _parse_file(path)
        resolved = resolve(program, path)
        fn_names = {fn.name for fn in resolved.functions}
        assert "helpers.add_one" in fn_names
        assert "add_two" in fn_names

    def test_circular_import_error(self):
        path = _fixture("circular_a.mao")
        program = _parse_file(path)
        with pytest.raises(MaomiError, match="circular import"):
            resolve(program, path)

    def test_diamond_import(self):
        path = _fixture("diamond_top.mao")
        program = _parse_file(path)
        resolved = resolve(program, path)
        fn_names = {fn.name for fn in resolved.functions}
        assert "diamond_left.left_fn" in fn_names
        assert "diamond_right.right_fn" in fn_names
        assert "combined" in fn_names

    def test_missing_module_error(self):
        source = 'import nonexistent; fn f() -> f32 { 1.0 }'
        tokens = Lexer(source, filename=_fixture("test.mao")).tokenize()
        program = Parser(tokens, filename=_fixture("test.mao")).parse()
        with pytest.raises(MaomiError, match="not found"):
            resolve(program, _fixture("test.mao"))

    def test_bad_name_in_from_import(self):
        source = 'from mathlib import { nonexistent }; fn f() -> f32 { 1.0 }'
        path = _fixture("test.mao")
        tokens = Lexer(source, filename=path).tokenize()
        program = Parser(tokens, filename=path).parse()
        with pytest.raises(MaomiError, match="no function 'nonexistent'"):
            resolve(program, path)


# ---- Full pipeline: typecheck + codegen ----


class TestModuleCompilation:
    def test_qualified_import_compiles(self):
        mlir = _compile_file(_fixture("uses_mathlib.mao"))
        assert "func.func @quad" in mlir
        assert "func.call @mathlib.double" in mlir

    def test_from_import_compiles(self):
        mlir = _compile_file(_fixture("uses_from.mao"))
        assert "func.func @quad" in mlir
        assert "func.call @double" in mlir

    def test_path_import_compiles(self):
        mlir = _compile_file(_fixture("uses_path.mao"))
        assert "func.func @add_two" in mlir
        assert "func.call @helpers.add_one" in mlir

    def test_module_with_if_expr_compiles(self):
        mlir = _compile_file(_fixture("uses_internal.mao"))
        assert "func.func @apply_relu" in mlir
        assert "func.call @nn.relu" in mlir

    def test_diamond_compiles(self):
        mlir = _compile_file(_fixture("diamond_top.mao"))
        assert "func.func @combined" in mlir
        assert "func.call @diamond_left.left_fn" in mlir
        assert "func.call @diamond_right.right_fn" in mlir
