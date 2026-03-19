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
        assert len(resolved.imports) == 1  # preserved for LSP semantic tokens
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
        with pytest.raises(MaomiError, match="no function or struct 'nonexistent'"):
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


# ---- Struct imports ----


def _compile_source(source: str, filename: str = "test.mao") -> str:
    """Compile source string through full pipeline. Returns MLIR."""
    path = _fixture(filename)
    tokens = Lexer(source, filename=path).tokenize()
    program = Parser(tokens, filename=path).parse()
    program = resolve(program, path)
    checker = TypeChecker(filename=path)
    errors = checker.check(program)
    assert not errors, f"Type errors: {errors}"
    program = transform_grad(program, checker.type_map)
    return StableHLOCodegen(program, checker.type_map).generate()


def _typecheck_source(source: str, filename: str = "test.mao"):
    """Parse + resolve + typecheck. Returns (program, checker)."""
    path = _fixture(filename)
    tokens = Lexer(source, filename=path).tokenize()
    program = Parser(tokens, filename=path).parse()
    program = resolve(program, path)
    checker = TypeChecker(filename=path)
    errors = checker.check(program)
    return program, checker, errors


class TestStructImports:
    """Tests for importing struct definitions across modules."""

    def test_from_import_struct_as_param_and_return(self):
        """from ... import { Point } — use as param type, return type, literal, field access."""
        source = '''
from with_struct import { Point };

fn mirror(p: Point) -> Point {
    Point { x: p.y, y: p.x }
}
'''
        mlir = _compile_source(source)
        assert "func.func @mirror" in mlir
        assert "tuple" in mlir  # struct compiles to tuple

    def test_from_import_struct_and_function(self):
        """from ... import { Point, make_point } — mixed function + struct import."""
        source = '''
from with_struct import { Point, make_point };

fn origin() -> Point {
    make_point(0.0, 0.0)
}
'''
        mlir = _compile_source(source)
        assert "func.func @origin" in mlir
        assert "func.call @make_point" in mlir

    def test_qualified_import_struct(self):
        """import ... as alias — use alias.Struct in type annotations and literals."""
        source = '''
import with_struct as ws;

fn mirror(p: ws.Point) -> ws.Point {
    ws.Point { x: p.y, y: p.x }
}
'''
        mlir = _compile_source(source)
        assert "func.func @mirror" in mlir
        assert "tuple" in mlir

    def test_qualified_import_struct_with_function(self):
        """import ... as alias — call alias.fn() returning alias.Struct."""
        source = '''
import with_struct as ws;

fn origin() -> ws.Point {
    ws.make_point(0.0, 0.0)
}
'''
        mlir = _compile_source(source)
        assert "func.func @origin" in mlir
        assert "func.call @ws.make_point" in mlir

    def test_nested_struct_import(self):
        """Import module with nested structs — both types available."""
        source = '''
from nested_struct import { Inner, Outer };

fn get_val(o: Outer) -> f32 {
    o.inner.val
}
'''
        mlir = _compile_source(source)
        assert "func.func @get_val" in mlir

    def test_nested_struct_qualified(self):
        """Qualified import of nested structs."""
        source = '''
import nested_struct as ns;

fn wrap(v: f32) -> ns.Outer {
    ns.Outer { inner: ns.Inner { val: v }, scale: 1.0 }
}
'''
        mlir = _compile_source(source)
        assert "func.func @wrap" in mlir

    def test_struct_with_grad(self):
        """Import struct, use in function with grad — struct-shaped gradients work."""
        source = '''
from with_struct import { Point };

fn loss_and_grad(p: Point) -> Point {
    let loss = p.x * p.x + p.y * p.y;
    grad(loss, p)
}
'''
        mlir = _compile_source(source)
        assert "func.func @loss_and_grad" in mlir

    def test_struct_field_access_imported(self):
        """Import struct, use field access and with-expression."""
        source = '''
from with_struct import { Point };

fn update_x(p: Point, new_x: f32) -> Point {
    p with { x = new_x }
}
'''
        mlir = _compile_source(source)
        assert "func.func @update_x" in mlir

    def test_collision_imported_vs_local(self):
        """Imported struct conflicts with local struct → error."""
        source = '''
from with_struct import { Point };
struct Point { a: f32, b: f32, c: f32 }
fn f(p: Point) -> f32 { p.a }
'''
        path = _fixture("test.mao")
        tokens = Lexer(source, filename=path).tokenize()
        program = Parser(tokens, filename=path).parse()
        with pytest.raises(MaomiError, match="conflicts with local struct"):
            resolve(program, path)

    def test_collision_two_modules(self):
        """Two modules define same-named struct → error on selective import."""
        source = '''
from with_struct import { Point };
from with_struct2 import { Point };
fn f(p: Point) -> f32 { p.x }
'''
        path = _fixture("test.mao")
        tokens = Lexer(source, filename=path).tokenize()
        program = Parser(tokens, filename=path).parse()
        with pytest.raises(MaomiError, match="imported from multiple modules"):
            resolve(program, path)

    def test_type_identity_selective_import(self):
        """from ... import { Point } — Point and qualified name resolve to same type."""
        source = '''
from with_struct import { Point, make_point };

fn use_it() -> Point {
    make_point(1.0, 2.0)
}
'''
        _, checker, errors = _typecheck_source(source)
        assert not errors
        # Both "Point" and "with_struct.Point" should resolve to same StructType
        assert "Point" in checker.struct_defs
        assert "with_struct.Point" in checker.struct_defs
        assert checker.struct_defs["Point"] is checker.struct_defs["with_struct.Point"]

    def test_resolver_merges_struct_defs(self):
        """Resolver merges struct_defs from imported modules."""
        source = '''
from with_struct import { Point };
fn f(p: Point) -> f32 { p.x }
'''
        path = _fixture("test.mao")
        tokens = Lexer(source, filename=path).tokenize()
        program = Parser(tokens, filename=path).parse()
        resolved = resolve(program, path)
        struct_names = {sd.name for sd in resolved.struct_defs}
        assert "with_struct.Point" in struct_names
        assert "Point" in struct_names  # alias

    def test_qualified_struct_no_unqualified_leak(self):
        """import ... as alias — struct only available qualified, not bare name."""
        source = '''
import with_struct as ws;
fn f(p: Point) -> f32 { p.x }
'''
        _, _, errors = _typecheck_source(source)
        assert len(errors) > 0  # "Point" not found, only "ws.Point"

    def test_parser_dotted_type_annotation(self):
        """Parser handles dotted type annotations like ws.Point."""
        tokens = Lexer("fn f(p: ws.Point) -> ws.Point { p }").tokenize()
        program = Parser(tokens).parse()
        fn = program.functions[0]
        assert fn.params[0].type_annotation.base == "ws.Point"
        assert fn.return_type.base == "ws.Point"

    def test_parser_dotted_struct_literal(self):
        """Parser handles dotted struct literals like ws.Point { x: 1.0 }."""
        from maomi.ast_nodes import StructLiteral
        tokens = Lexer("fn f() -> f32 { ws.Point { x: 1.0, y: 2.0 } }").tokenize()
        program = Parser(tokens).parse()
        body_expr = program.functions[0].body.expr
        assert isinstance(body_expr, StructLiteral)
        assert body_expr.name == "ws.Point"
