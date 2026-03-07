"""Tests for the Maomi LSP server validation and hover logic."""

import pytest
from lsprotocol import types

from maomi.lsp import (
    validate, _find_node_at, _error_to_diagnostic, _span_contains,
    _complete_general, _complete_dot, _vars_in_scope, AnalysisResult,
    _children_of, _goto_find_definition, _span_to_range,
    _refs_classify_node, _refs_collect_all,
    _build_document_symbols,
    prepare_rename_at, rename_at,
    _sig_parse_call_context, _BUILTIN_SIGNATURES, _BUILTIN_DOCS,
    _get_hover_text, _build_signature_help,
    _build_inlay_hints,
    _sem_collect_tokens, _sem_delta_encode,
    _ST_FUNCTION, _ST_PARAMETER, _ST_VARIABLE, _ST_STRUCT,
    _ST_PROPERTY, _ST_TYPE, _ST_NUMBER, _ST_KEYWORD,
    _MOD_DECLARATION, _MOD_DEFINITION,
    _ca_edit_distance, _ca_find_similar, code_actions, _cache,
    _build_folding_ranges,
    _sel_collect_ancestors, _sel_build_chain,
    _workspace_symbols,
)
from maomi.ast_nodes import (
    Identifier, IntLiteral, FloatLiteral, BinOp, CallExpr,
    LetStmt, FnDef, Block, Span, StructLiteral, FieldAccess,
    ScanExpr, MapExpr,
)
from maomi.errors import MaomiError


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_source_no_diagnostics(self):
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert diags == []
        assert result.program is not None
        assert len(result.type_map) > 0

    def test_lexer_error(self):
        source = "fn f() -> f32 { ! }"  # '!' alone is invalid
        diags, result = validate(source, "<test>")
        assert len(diags) == 1
        assert "unexpected" in diags[0].message.lower() or "!" in diags[0].message
        assert result.program is None

    def test_parse_error(self):
        source = "fn f( -> f32 { x }"  # missing ')'
        diags, result = validate(source, "<test>")
        assert len(diags) == 1
        assert result.program is None

    def test_type_error(self):
        source = "fn f(x: f32) -> i32 { x }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert result.program is not None  # AST still available

    def test_multiple_type_errors(self):
        source = """
fn f(x: f32) -> i32 { x }
fn g(y: i32) -> f32 { y }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 2

    def test_type_map_populated_despite_errors(self):
        source = "fn f(x: f32) -> i32 { x + 1.0 }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert len(result.type_map) > 0


# ---------------------------------------------------------------------------
# Error to diagnostic conversion
# ---------------------------------------------------------------------------

class TestErrorConversion:
    def test_1indexed_to_0indexed(self):
        err = MaomiError("test error", "<test>", line=1, col=1)
        diag = _error_to_diagnostic(err)
        assert diag.range.start.line == 0
        assert diag.range.start.character == 0
        assert diag.message == "test error"
        assert diag.severity == types.DiagnosticSeverity.Error
        assert diag.source == "maomi"

    def test_multiline_position(self):
        err = MaomiError("bad thing", "<test>", line=10, col=5)
        diag = _error_to_diagnostic(err)
        assert diag.range.start.line == 9
        assert diag.range.start.character == 4


# ---------------------------------------------------------------------------
# AST node finding (hover support)
# ---------------------------------------------------------------------------

class TestSpanContains:
    def test_inside(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 5) is True

    def test_at_start(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 1) is True

    def test_at_end(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 10) is True

    def test_before(self):
        span = Span(2, 5, 2, 10)
        assert _span_contains(span, 1, 5) is False

    def test_after(self):
        span = Span(1, 1, 1, 10)
        assert _span_contains(span, 1, 11) is False

    def test_multiline(self):
        span = Span(1, 5, 3, 10)
        assert _span_contains(span, 2, 1) is True  # middle line, any col


class TestFindNodeAt:
    def test_find_identifier_in_function(self):
        source = "fn f(x: f32) -> f32 { x }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        # The trailing expr is an Identifier
        body_expr = fn.body.expr
        assert isinstance(body_expr, Identifier)

        # Find node at the identifier's position
        node = _find_node_at(fn, body_expr.span.line_start, body_expr.span.col_start)
        assert node is not None
        assert isinstance(node, Identifier)
        assert node.name == "x"

    def test_find_innermost_in_binop(self):
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        body_expr = fn.body.expr
        assert isinstance(body_expr, BinOp)

        # Cursor on 'a' should find Identifier("a"), not the BinOp
        left = body_expr.left
        assert isinstance(left, Identifier)
        node = _find_node_at(fn, left.span.line_start, left.span.col_start)
        assert isinstance(node, Identifier)
        assert node.name == "a"

    def test_returns_none_outside_span(self):
        source = "fn f(x: f32) -> f32 { x }"
        from maomi.lexer import Lexer
        from maomi.parser import Parser
        tokens = Lexer(source, "<test>").tokenize()
        program = Parser(tokens, "<test>").parse()
        fn = program.functions[0]

        # Line 99 is way outside the function
        node = _find_node_at(fn, 99, 1)
        assert node is None


# ---------------------------------------------------------------------------
# Completion tests
# ---------------------------------------------------------------------------

class TestCompletion:
    def test_general_includes_keywords(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)  # inside body
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "fn" in labels
        assert "let" in labels
        assert "if" in labels
        assert "scan" in labels
        assert "grad" in labels

    def test_general_includes_builtins(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "exp" in labels
        assert "mean" in labels
        assert "tanh" in labels
        assert "iota" in labels
        assert "random" in labels  # builtin namespace

    def test_general_includes_type_names(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=23)
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "f32" in labels
        assert "i32" in labels
        assert "bool" in labels

    def test_general_includes_user_functions(self):
        source = """
fn helper(x: f32) -> f32 { x }
fn main(y: f32) -> f32 { y }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=26)
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "helper" in labels
        assert "main" in labels

    def test_general_includes_struct_names(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        pos = types.Position(line=2, character=25)
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "Point" in labels

    def test_vars_in_scope_includes_params(self):
        source = "fn f(x: f32, y: i32) -> f32 { x }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=32)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "x" in var_names
        assert "y" in var_names

    def test_vars_in_scope_includes_let_bindings(self):
        source = """fn f(x: f32) -> f32 {
    let a = 1.0;
    let b = 2.0;
    x
}"""
        _, result = validate(source, "<test>")
        # Cursor on line 3 (0-indexed), after both let bindings
        pos = types.Position(line=3, character=4)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "x" in var_names
        assert "a" in var_names
        assert "b" in var_names

    def test_vars_in_scope_excludes_later_bindings(self):
        source = """fn f(x: f32) -> f32 {
    let a = 1.0;
    x
}"""
        _, result = validate(source, "<test>")
        # Cursor on line 1 (0-indexed), before the let binding completes
        # Actually let's put cursor at start of line 1, before the let
        pos = types.Position(line=1, character=0)
        variables = _vars_in_scope(result, pos)
        var_names = [v[0] for v in variables]
        assert "x" in var_names
        # 'a' should not be in scope yet (defined on this line)
        assert "a" not in var_names

    def test_dot_completion_struct_fields(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        # Position at the dot after 'p' — need to find 'p' node
        # 'p.' is at line 2, the dot is after 'p'
        # In the source, line 2 is "fn f(p: Point) -> f32 { p.x }"
        # 'p' starts at character 25, dot at 26
        pos = types.Position(line=2, character=26)
        comp = _complete_dot(result, pos)
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "x" in labels
        assert "y" in labels
# ---------------------------------------------------------------------------
# Go to Definition tests
# ---------------------------------------------------------------------------

class TestGoToDefinition:
    def _find_node_by_type_and_attr(self, fn, node_type, attr_name, attr_value):
        """Recursively find a node of given type with matching attribute."""
        if isinstance(fn, node_type):
            if getattr(fn, attr_name, None) == attr_value:
                return fn
        for child in _children_of(fn):
            result = self._find_node_by_type_and_attr(child, node_type, attr_name, attr_value)
            if result is not None:
                return result
        return None

    def test_goto_function_definition(self):
        source = "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        main_fn = result.program.functions[1]
        helper_fn = result.program.functions[0]
        # Find the CallExpr for helper(y) in main's body
        call_node = self._find_node_by_type_and_attr(main_fn, CallExpr, "callee", "helper")
        assert call_node is not None
        defn_span = _goto_find_definition(call_node, main_fn, result)
        assert defn_span is not None
        assert defn_span == helper_fn.span

    def test_goto_param_definition(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        # Find the Identifier 'x' in the body
        ident_node = self._find_node_by_type_and_attr(fn.body, Identifier, "name", "x")
        assert ident_node is not None
        defn_span = _goto_find_definition(ident_node, fn, result)
        assert defn_span is not None
        assert defn_span == fn.params[0].span

    def test_goto_let_definition(self):
        source = "fn f(x: f32) -> f32 {\n    let a = x;\n    a\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        # The trailing expression is Identifier 'a'
        assert isinstance(fn.body.expr, Identifier)
        assert fn.body.expr.name == "a"
        defn_span = _goto_find_definition(fn.body.expr, fn, result)
        assert defn_span is not None
        # Should point to the LetStmt
        let_stmt = fn.body.stmts[0]
        assert isinstance(let_stmt, LetStmt)
        assert defn_span == let_stmt.span

    def test_goto_struct_definition(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        struct_def = result.program.struct_defs[0]
        # Find the StructLiteral node
        struct_lit = self._find_node_by_type_and_attr(fn, StructLiteral, "name", "Point")
        assert struct_lit is not None
        defn_span = _goto_find_definition(struct_lit, fn, result)
        assert defn_span is not None
        assert defn_span == struct_def.span

    def test_goto_field_access_definition(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        struct_def = result.program.struct_defs[0]
        # Find the FieldAccess node
        field_node = self._find_node_by_type_and_attr(fn, FieldAccess, "field", "x")
        assert field_node is not None
        defn_span = _goto_find_definition(field_node, fn, result)
        assert defn_span is not None
        assert defn_span == struct_def.span

    def test_returns_none_for_builtin(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        # Find the CallExpr for exp(x)
        call_node = self._find_node_by_type_and_attr(fn, CallExpr, "callee", "exp")
        assert call_node is not None
        defn_span = _goto_find_definition(call_node, fn, result)
        assert defn_span is None

    def test_span_to_range_conversion(self):
        span = Span(1, 1, 1, 10)
        r = _span_to_range(span)
        assert r.start.line == 0
        assert r.start.character == 0
        assert r.end.line == 0
        assert r.end.character == 9

    def test_goto_scan_carry_var(self):
        source = "fn f(xs: f32[3]) -> f32 {\n    scan (acc, x) in (0.0, xs) { acc + x }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        # Find the scan expr
        scan_node = self._find_node_by_type_and_attr(fn, ScanExpr, "carry_var", "acc")
        assert scan_node is not None
        # Find 'acc' identifier inside scan body
        acc_ident = self._find_node_by_type_and_attr(scan_node.body, Identifier, "name", "acc")
        assert acc_ident is not None
        defn_span = _goto_find_definition(acc_ident, fn, result)
        assert defn_span is not None
        assert defn_span == scan_node.span

    def test_goto_map_elem_var(self):
        source = "fn f(xs: f32[3]) -> f32[3] {\n    map x in xs { x + 1.0 }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        map_node = self._find_node_by_type_and_attr(fn, MapExpr, "elem_var", "x")
        assert map_node is not None
        # Find 'x' identifier inside map body
        x_ident = self._find_node_by_type_and_attr(map_node.body, Identifier, "name", "x")
        assert x_ident is not None
        defn_span = _goto_find_definition(x_ident, fn, result)
        assert defn_span is not None
        assert defn_span == map_node.span

    def test_returns_none_for_literal(self):
        source = "fn f() -> f32 { 1.0 }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        float_node = fn.body.expr
        assert isinstance(float_node, FloatLiteral)
        defn_span = _goto_find_definition(float_node, fn, result)
        assert defn_span is None


# ---------------------------------------------------------------------------
# Find References tests
# ---------------------------------------------------------------------------


def _find_refs(source, line_0, col_0, include_declaration=False):
    """Helper: validate source, find the node at (line_0, col_0) 0-indexed,
    classify it, and collect all reference spans. Returns list of (line_start, col_start)
    pairs (1-indexed) for easy assertion."""
    _, result = validate(source, "<test>")
    assert result.program is not None

    line = line_0 + 1  # to 1-indexed
    col = col_0 + 1

    # Check struct defs
    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            spans = _refs_collect_all(result, sd.name, "struct", include_declaration)
            return [(s.line_start, s.col_start) for s in spans]

    # Check functions
    for fn in result.program.functions:
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if name:
                spans = _refs_collect_all(result, name, kind,
                                          include_declaration, fn_scope=fn)
                return [(s.line_start, s.col_start) for s in spans]
    return []


class TestFindReferences:
    def test_find_function_references(self):
        source = "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) + helper(y) }"
        # Cursor on "helper" in fn main body — the first call at col 25
        # Line 1 (0-indexed): "fn main(y: f32) -> f32 { helper(y) + helper(y) }"
        #                       0123456789...           25
        refs = _find_refs(source, 1, 25, include_declaration=False)
        # Should find 2 call sites (both helper(...) calls)
        assert len(refs) == 2

    def test_find_function_references_include_declaration(self):
        source = "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) + helper(y) }"
        refs = _find_refs(source, 1, 25, include_declaration=True)
        # Should find declaration + 2 call sites = 3
        assert len(refs) == 3
        # The declaration is on line 1 (1-indexed)
        assert (1, 1) in refs

    def test_find_variable_references(self):
        source = "fn f(x: f32) -> f32 { x + x }"
        # Cursor on "x" in the body — "x + x" starts around col 22 (0-indexed)
        # "fn f(x: f32) -> f32 { x + x }"
        #  0123456789012345678901234567890
        #                       22   26
        refs = _find_refs(source, 0, 22, include_declaration=False)
        # Should find 2 Identifier uses of x
        assert len(refs) == 2

    def test_find_variable_references_include_declaration(self):
        source = "fn f(x: f32) -> f32 { x + x }"
        refs = _find_refs(source, 0, 22, include_declaration=True)
        # 2 uses + 1 param declaration = 3
        assert len(refs) == 3

    def test_find_struct_references(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { let q = Point { x: 1.0, y: 2.0 }; q.x }"
        # Cursor on "Point" in param type annotation — line 1, around col 8
        # "fn f(p: Point) -> f32 { let q = Point { x: 1.0, y: 2.0 }; q.x }"
        #  012345678
        refs = _find_refs(source, 1, 8, include_declaration=False)
        # Should find: TypeAnnotation "Point" in param + StructLiteral "Point"
        assert len(refs) >= 2

    def test_find_struct_references_include_declaration(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { let q = Point { x: 1.0, y: 2.0 }; q.x }"
        refs = _find_refs(source, 1, 8, include_declaration=True)
        # Should include struct def + param type + struct literal = 3+
        assert len(refs) >= 3

    def test_variable_scoped_to_function(self):
        source = "fn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }"
        # Cursor on "x" in f's body — line 0, col 22
        refs_f = _find_refs(source, 0, 22, include_declaration=False)
        # Should only find uses in f, not in g
        assert len(refs_f) == 1
        # Cursor on "x" in g's body — line 1, col 22
        refs_g = _find_refs(source, 1, 22, include_declaration=False)
        assert len(refs_g) == 1

    def test_refs_span_to_range_conversion(self):
        span = Span(2, 5, 2, 10)
        r = _span_to_range(span)
        # 1-indexed span -> 0-indexed LSP range
        assert r.start.line == 1
        assert r.start.character == 4
        assert r.end.line == 1
        assert r.end.character == 9

    def test_find_function_from_definition(self):
        source = "fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        # Cursor on "helper" at the definition site — line 0, col 3
        # "fn helper(x: f32) -> f32 { x }"
        #  0123
        refs = _find_refs(source, 0, 3, include_declaration=False)
        # Should find the call site in main
        assert len(refs) == 1

    def test_find_struct_from_definition(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        # Cursor on "Point" at struct def — line 0, col 7
        refs = _find_refs(source, 0, 7, include_declaration=True)
        # struct def + param type annotation
        assert len(refs) >= 2

    def test_no_references_for_literal(self):
        source = "fn f(x: f32) -> f32 { 1.0 }"
        # Cursor on "1.0" — a FloatLiteral, not a referenceable symbol
        refs = _find_refs(source, 0, 22, include_declaration=False)
        assert refs == []

    def test_grad_wrt_reference(self):
        source = "fn loss(w: f32) -> f32 { w * w }\nfn main(w: f32) -> f32 { grad(loss(w), w) }"
        # Cursor on the second "w" in main (the one used in computation)
        # Actually, let's find refs for "w" in main's body
        # "fn main(w: f32) -> f32 { grad(loss(w), w) }"
        #  0         1         2         3         4
        #  0123456789012345678901234567890123456789012345
        # The "w" arg inside loss(w) is at col 36, and grad's wrt "w" is at col 39
        # Let's click on w param at col 8
        refs = _find_refs(source, 1, 8, include_declaration=False)
        # Should find: Identifier w in loss(w) arg, grad wrt w
        assert len(refs) >= 2


# ---------------------------------------------------------------------------
# Document Symbol tests
# ---------------------------------------------------------------------------

class TestDocumentSymbol:
    def test_functions_listed(self):
        source = """
fn foo(x: f32) -> f32 { x }
fn bar(y: i32) -> i32 { y }
"""
        _, result = validate(source, "<test>")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        fn_names = [s.name for s in symbols if s.kind == types.SymbolKind.Function]
        assert "foo" in fn_names
        assert "bar" in fn_names

    def test_structs_listed(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        struct_names = [s.name for s in symbols if s.kind == types.SymbolKind.Struct]
        assert "Point" in struct_names

    def test_params_as_children(self):
        source = "fn f(x: f32, y: i32) -> f32 { x }"
        _, result = validate(source, "<test>")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        fn_sym = [s for s in symbols if s.name == "f"][0]
        assert fn_sym.children is not None
        child_names = [c.name for c in fn_sym.children]
        assert "x" in child_names
        assert "y" in child_names
        for child in fn_sym.children:
            assert child.kind == types.SymbolKind.Variable

    def test_struct_fields_as_children(self):
        source = """
struct Vec2 { x: f32, y: f32 }
fn f(v: Vec2) -> f32 { v.x }
"""
        _, result = validate(source, "<test>")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        struct_sym = [s for s in symbols if s.name == "Vec2"][0]
        assert struct_sym.children is not None
        child_names = [c.name for c in struct_sym.children]
        assert "x" in child_names
        assert "y" in child_names
        for child in struct_sym.children:
            assert child.kind == types.SymbolKind.Property

    def test_no_program_returns_none(self):
        result = AnalysisResult(program=None, type_map={}, fn_table={}, struct_defs={})
        symbols = _build_document_symbols(result)
        assert symbols is None

    def test_none_result_returns_none(self):
        symbols = _build_document_symbols(None)
        assert symbols is None

    def test_function_symbol_range(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        fn_sym = [s for s in symbols if s.name == "f"][0]
        # Span is 1-indexed, LSP Range is 0-indexed
        assert fn_sym.range.start.line == 0
        assert fn_sym.range.start.character == 0


# ---------------------------------------------------------------------------
# Rename tests
# ---------------------------------------------------------------------------


def _edit_ranges(edits):
    """Extract (line, start_char, end_char) tuples from TextEdit list for easy comparison."""
    return sorted(
        (e.range.start.line, e.range.start.character, e.range.end.character)
        for e in edits
    )


class TestRename:
    def test_rename_function(self):
        # Line 0: ""
        # Line 1: "fn helper(x: f32) -> f32 { x }"
        # Line 2: "fn main(y: f32) -> f32 { helper(y) }"
        source = "\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")
        assert result.program is not None

        # Cursor on "helper" in the FnDef (line 1, char 3 — 0-indexed)
        edits = rename_at(source, result, 1, 3, "util")
        assert edits is not None
        assert len(edits) == 2
        for e in edits:
            assert e.new_text == "util"
        ranges = _edit_ranges(edits)
        # FnDef on line 1, "helper" at chars 3..9
        assert (1, 3, 9) in ranges
        # CallExpr on line 2, "helper" at chars 25..31
        assert (2, 25, 31) in ranges

    def test_rename_function_from_callsite(self):
        source = "\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { helper(y) }"
        _, result = validate(source, "<test>")

        # Cursor on "helper" at the call site (line 2, char 25)
        edits = rename_at(source, result, 2, 25, "util")
        assert edits is not None
        assert len(edits) == 2

    def test_rename_variable(self):
        source = "fn f(x: f32) -> f32 { x + x }"
        _, result = validate(source, "<test>")

        # Cursor on first "x" in body (line 0, char 22 — 0-indexed)
        edits = rename_at(source, result, 0, 22, "a")
        assert edits is not None
        # Should rename: param "x" + two Identifier "x" uses = 3 edits
        assert len(edits) == 3
        for e in edits:
            assert e.new_text == "a"

    def test_rename_param(self):
        source = "fn f(x: f32, y: f32) -> f32 { x + y }"
        _, result = validate(source, "<test>")

        # Cursor on "x" in param position (line 0, char 5)
        edits = rename_at(source, result, 0, 5, "a")
        assert edits is not None
        # Should rename param "x" and Identifier "x", but NOT "y"
        for e in edits:
            assert e.new_text == "a"
        ranges = _edit_ranges(edits)
        # Check that no edit touches "y" positions
        for line, start, end in ranges:
            text_at = source.splitlines()[line][start:end]
            assert text_at == "x"

    def test_rename_respects_scoping(self):
        source = "\nfn f(x: f32) -> f32 { x }\nfn g(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")

        # Rename "x" in function f (line 1, char 5 — the param "x")
        edits = rename_at(source, result, 1, 5, "a")
        assert edits is not None
        # All edits should be on line 1 (function f), not line 2 (function g)
        for e in edits:
            assert e.range.start.line == 1

    def test_rename_let_binding(self):
        source = "fn f(x: f32) -> f32 { let y = x; y + y }"
        _, result = validate(source, "<test>")

        # Cursor on "y" in "let y" (line 0, char 26)
        edits = rename_at(source, result, 0, 26, "z")
        assert edits is not None
        # let binding "y" + two Identifier "y" uses = 3
        assert len(edits) == 3
        for e in edits:
            assert e.new_text == "z"

    def test_prepare_rename_returns_range(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")

        # Cursor on "x" in body (line 0, char 22)
        rng = prepare_rename_at(source, result, 0, 22)
        assert rng is not None
        assert rng.start.line == 0
        assert rng.start.character == 22
        assert rng.end.character == 23  # "x" is 1 char

    def test_prepare_rename_rejects_literals(self):
        source = "fn f(x: f32) -> f32 { 1.0 }"
        _, result = validate(source, "<test>")

        # Cursor on "1.0" (line 0, char 22)
        rng = prepare_rename_at(source, result, 0, 22)
        assert rng is None

    def test_prepare_rename_rejects_builtins(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        _, result = validate(source, "<test>")

        # Cursor on "exp" (line 0, char 22)
        rng = prepare_rename_at(source, result, 0, 22)
        assert rng is None

    def test_rename_struct(self):
        source = "\nstruct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point { p }"
        _, result = validate(source, "<test>")

        # Cursor on "Point" in struct def (line 1, char 7)
        edits = rename_at(source, result, 1, 7, "Vec2")
        assert edits is not None
        # StructDef name + param type annotation + return type annotation = 3
        assert len(edits) == 3
        for e in edits:
            assert e.new_text == "Vec2"

    def test_rename_returns_none_for_empty_program(self):
        source = "fn f( -> f32 { x }"  # parse error
        _, result = validate(source, "<test>")
        assert rename_at(source, result, 0, 3, "g") is None

    def test_rename_returns_none_outside_code(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        # Line 99 is way outside
        assert rename_at(source, result, 99, 0, "a") is None


# ---------------------------------------------------------------------------
# Signature Help tests
# ---------------------------------------------------------------------------

class TestSignatureHelp:
    def test_parse_call_context_at_open_paren(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        # Cursor right after "exp(" — character 27 is right after the '('
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=27))
        assert name == "exp"
        assert idx == 0

    def test_parse_call_context_after_comma(self):
        source = "fn f(a: f32, b: f32) -> f32 { helper(a, b) }"
        # Cursor after the comma+space, on 'b' — should be param index 1
        # "helper(a, b)" — 'h' starts at col 30, '(' at 36, 'a' at 37, ',' at 38, ' ' at 39, 'b' at 40
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=40))
        assert name == "helper"
        assert idx == 1

    def test_parse_call_context_third_param(self):
        source = "fn f(a: f32) -> f32 { random.uniform(k, 0.0, 1.0, 4) }"
        # Cursor after second comma — should be param index 2
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=48))
        assert name == "random.uniform"
        assert idx == 2

    def test_builtin_signatures_contains_expected_entries(self):
        expected = [
            "exp", "log", "tanh", "sqrt", "abs", "mean", "sum",
            "reshape", "concat", "iota", "random.key", "random.split",
            "random.uniform", "random.normal", "conv2d", "max_pool", "avg_pool",
        ]
        for name in expected:
            assert name in _BUILTIN_SIGNATURES, f"Missing builtin: {name}"
            pnames, ptypes, ret = _BUILTIN_SIGNATURES[name]
            assert len(pnames) == len(ptypes), f"Param name/type mismatch for {name}"
            assert isinstance(ret, str)

    def test_no_signature_outside_call(self):
        source = "fn f(x: f32) -> f32 { x }"
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=23))
        assert name is None

    def test_nested_calls_inner(self):
        source = "fn f(x: f32) -> f32 { exp(log(x)) }"
        # Cursor inside log( — after the inner '(' — should return "log", not "exp"
        # "exp(log(x))" — 'e' at 22, '(' at 25, 'l' at 26, 'o' at 27, 'g' at 28, '(' at 29, 'x' at 30
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=30))
        assert name == "log"
        assert idx == 0

    def test_nested_calls_outer(self):
        source = "fn f(x: f32) -> f32 { exp(log(x), y) }"
        # Cursor after the comma in exp(..., y) — between log(x) and y
        # exp(log(x), y) — '(' at 25, 'log(x)' fills 26-30, ')' at 31, ',' at 32, ' ' at 33, 'y' at 34
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=34))
        assert name == "exp"
        assert idx == 1

    def test_multiline_call(self):
        source = "fn f(a: f32, b: f32) -> f32 {\n    helper(\n        a,\n        b\n    )\n}"
        # Cursor on line 3 (the 'b' line), character 8 — after the comma on prev line
        name, idx = _sig_parse_call_context(source, types.Position(line=3, character=9))
        assert name == "helper"
        assert idx == 1

    def test_empty_args(self):
        source = "fn f() -> f32 { helper() }"
        # Cursor right after "helper(" — inside empty parens
        name, idx = _sig_parse_call_context(source, types.Position(line=0, character=23))
        assert name == "helper"
        assert idx == 0


# ---------------------------------------------------------------------------
# Inlay Hints tests
# ---------------------------------------------------------------------------

class TestInlayHints:
    def test_let_binding_shows_type(self):
        source = "fn f(x: f32) -> f32 { let a = x; a }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        assert len(hints) == 1
        assert hints[0].label == ": f32"
        assert hints[0].kind == types.InlayHintKind.Type
        # 'a' is right after 'let ' in the source
        # "fn f(x: f32) -> f32 { let a = x; a }"
        #  position 0..                   ^23='l' 27='a' 28=after 'a'
        assert hints[0].position.line == 0
        assert hints[0].position.character == 27

    def test_no_hint_when_explicitly_annotated(self):
        source = "fn f(x: f32) -> f32 { let a: f32 = x; a }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        assert len(hints) == 0

    def test_multiple_let_bindings(self):
        source = """fn f(x: f32) -> f32 {
    let a = x;
    let b = a;
    b
}"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 5, source)
        assert len(hints) == 2
        labels = [h.label for h in hints]
        assert labels[0] == ": f32"
        assert labels[1] == ": f32"

    def test_range_filtering(self):
        source = """fn f(x: f32) -> f32 {
    let a = x;
    let b = a;
    b
}"""
        _, result = validate(source, "<test>")
        # Only request hints for line 2 (1-indexed) — should match "let a = x;"
        hints = _build_inlay_hints(result, 2, 2, source)
        assert len(hints) == 1
        assert hints[0].label == ": f32"

    def test_hint_position_is_after_name(self):
        source = """fn f(x: f32) -> f32 {
    let abc = x;
    abc
}"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 4, source)
        assert len(hints) == 1
        # "    let abc = x;"  — 'a' of 'abc' is at char 8 (0-indexed), len('abc')=3
        # so hint position should be at character 11 (right after 'abc')
        assert hints[0].position.line == 1
        assert hints[0].position.character == 11

    def test_no_program_returns_empty(self):
        source = "fn f( -> f32 { x }"  # parse error
        _, result = validate(source, "<test>")
        assert result.program is None
        hints = _build_inlay_hints(result, 1, 1, source)
        assert hints == []

    def test_hint_padding(self):
        source = "fn f(x: f32) -> f32 { let a = x; a }"
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 1, source)
        assert len(hints) == 1
        assert hints[0].padding_left is False
        assert hints[0].padding_right is True

    def test_nested_if_let_binding(self):
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        let pos = x;
        pos
    } else {
        let neg = 0.0 - x;
        neg
    }
}"""
        _, result = validate(source, "<test>")
        hints = _build_inlay_hints(result, 1, 9, source)
        assert len(hints) == 2
        labels = [h.label for h in hints]
        assert ": f32" in labels[0]
        assert ": f32" in labels[1]


# ---------------------------------------------------------------------------
# Semantic Tokens tests
# ---------------------------------------------------------------------------

class TestSemanticTokens:
    def test_function_definition_token(self):
        source = "fn foo(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        # Find the function name token
        fn_tokens = [t for t in tokens if t[3] == _ST_FUNCTION and t[4] & _MOD_DECLARATION]
        assert len(fn_tokens) >= 1
        # Verify it has the right length for "foo"
        assert fn_tokens[0][2] == 3
        # Verify position: "fn foo" — name starts at col 3 (0-indexed)
        assert fn_tokens[0][0] == 0  # line 0
        assert fn_tokens[0][1] == 3  # col 3

    def test_parameter_token(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        param_tokens = [t for t in tokens if t[3] == _ST_PARAMETER]
        assert len(param_tokens) >= 1
        # Should include both the param declaration and the usage of x in the body
        param_decl = [t for t in param_tokens if t[4] & _MOD_DECLARATION]
        param_usage = [t for t in param_tokens if t[4] == 0]
        assert len(param_decl) >= 1
        assert len(param_usage) >= 1

    def test_variable_vs_parameter(self):
        source = """fn f(x: f32) -> f32 {
    let a = x;
    a
}"""
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        # 'a' should be variable, 'x' in body should be parameter
        var_tokens = [t for t in tokens if t[3] == _ST_VARIABLE]
        param_tokens = [t for t in tokens if t[3] == _ST_PARAMETER and t[4] == 0]
        # 'a' appears as variable (declaration + usage)
        assert len(var_tokens) >= 2  # let decl + trailing expr
        # 'x' usage in "let a = x" should be parameter
        assert len(param_tokens) >= 1

    def test_delta_encoding(self):
        raw = [(0, 5, 3, 0, 0), (0, 10, 4, 1, 0), (1, 2, 5, 2, 0)]
        data = _sem_delta_encode(raw)
        assert data == [0, 5, 3, 0, 0, 0, 5, 4, 1, 0, 1, 2, 5, 2, 0]

    def test_delta_encoding_same_line(self):
        raw = [(2, 0, 3, 0, 0), (2, 5, 4, 1, 0)]
        data = _sem_delta_encode(raw)
        # First token: delta_line=2, delta_start=0
        # Second token: delta_line=0, delta_start=5 (5 - 0 = 5)
        assert data == [2, 0, 3, 0, 0, 0, 5, 4, 1, 0]

    def test_struct_definition_token(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        tokens = []
        for sd in result.program.struct_defs:
            _sem_collect_tokens(sd, tokens, set())
        struct_tokens = [t for t in tokens if t[3] == _ST_STRUCT]
        assert len(struct_tokens) >= 1
        # Struct name should have declaration modifier
        struct_decl = [t for t in struct_tokens if t[4] & _MOD_DECLARATION]
        assert len(struct_decl) >= 1
        # Verify length matches "Point"
        assert struct_decl[0][2] == 5

    def test_struct_type_annotations(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        tokens = []
        for sd in result.program.struct_defs:
            _sem_collect_tokens(sd, tokens, set())
        type_tokens = [t for t in tokens if t[3] == _ST_TYPE]
        # "f32" appears twice in struct fields
        assert len(type_tokens) >= 2

    def test_number_literal_token(self):
        source = "fn f(x: f32) -> f32 { 1.5 }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        num_tokens = [t for t in tokens if t[3] == _ST_NUMBER]
        assert len(num_tokens) >= 1
        # "1.5" has length 3
        assert num_tokens[0][2] == 3

    def test_int_literal_token(self):
        source = "fn f(x: i32) -> i32 { 42 }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        num_tokens = [t for t in tokens if t[3] == _ST_NUMBER]
        assert len(num_tokens) >= 1
        # "42" has length 2
        assert num_tokens[0][2] == 2

    def test_call_expression_token(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        call_tokens = [t for t in tokens if t[3] == _ST_FUNCTION and t[4] == 0]
        assert len(call_tokens) >= 1
        # "exp" has length 3
        assert call_tokens[0][2] == 3

    def test_field_access_token(self):
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }
"""
        _, result = validate(source, "<test>")
        tokens = []
        for fn in result.program.functions:
            _sem_collect_tokens(fn, tokens, set())
        prop_tokens = [t for t in tokens if t[3] == _ST_PROPERTY]
        assert len(prop_tokens) >= 1
        # "x" has length 1
        assert prop_tokens[0][2] == 1

    def test_let_keyword_token(self):
        source = """fn f(x: f32) -> f32 {
    let a = 1.0;
    a
}"""
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        kw_tokens = [t for t in tokens if t[3] == _ST_KEYWORD]
        assert len(kw_tokens) >= 1
        # "let" has length 3
        assert kw_tokens[0][2] == 3

    def test_let_variable_declaration(self):
        source = """fn f(x: f32) -> f32 {
    let a = 1.0;
    a
}"""
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        var_decl = [t for t in tokens if t[3] == _ST_VARIABLE and t[4] & _MOD_DECLARATION]
        assert len(var_decl) >= 1
        # "a" has length 1
        assert var_decl[0][2] == 1

    def test_type_annotation_token(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        tokens = []
        _sem_collect_tokens(result.program.functions[0], tokens, set())
        type_tokens = [t for t in tokens if t[3] == _ST_TYPE]
        # "f32" appears in param type annotation
        assert len(type_tokens) >= 1
        assert type_tokens[0][2] == 3  # "f32"

    def test_empty_program(self):
        """Delta-encode with no tokens produces empty list."""
        data = _sem_delta_encode([])
        assert data == []

    def test_full_program_collects_all_node_types(self):
        """A program with structs, functions, params, calls, literals, etc."""
        source = """
struct Vec2 { x: f32, y: f32 }
fn dot(a: Vec2, b: Vec2) -> f32 {
    let px = a.x;
    px
}
"""
        _, result = validate(source, "<test>")
        tokens = []
        for sd in result.program.struct_defs:
            _sem_collect_tokens(sd, tokens, set())
        for fn in result.program.functions:
            _sem_collect_tokens(fn, tokens, set())
        # Should have tokens of multiple types
        token_types = {t[3] for t in tokens}
        assert _ST_FUNCTION in token_types    # fn dot
        assert _ST_PARAMETER in token_types   # a, b
        assert _ST_VARIABLE in token_types    # px
        assert _ST_STRUCT in token_types      # Vec2
        assert _ST_PROPERTY in token_types    # .x
        assert _ST_TYPE in token_types        # f32
        assert _ST_KEYWORD in token_types     # let


# ---------------------------------------------------------------------------
# Code Action tests
# ---------------------------------------------------------------------------

class TestCodeActions:
    def test_edit_distance_basic(self):
        assert _ca_edit_distance("kitten", "sitting") == 3
        assert _ca_edit_distance("abc", "abc") == 0
        assert _ca_edit_distance("", "abc") == 3
        assert _ca_edit_distance("abc", "") == 3

    def test_edit_distance_transposition(self):
        # Swapping two adjacent chars is distance 2 (delete + insert)
        assert _ca_edit_distance("exp", "epx") == 2

    def test_edit_distance_single_change(self):
        assert _ca_edit_distance("cat", "bat") == 1
        assert _ca_edit_distance("cat", "ca") == 1
        assert _ca_edit_distance("cat", "cats") == 1

    def test_find_similar_finds_close_match(self):
        candidates = ["exp", "log", "tanh", "sqrt", "mean"]
        similar = _ca_find_similar("epx", candidates)
        assert "exp" in similar

    def test_find_similar_no_match(self):
        candidates = ["exp", "log"]
        similar = _ca_find_similar("zzzzzzz", candidates, max_distance=2)
        assert similar == []

    def test_find_similar_excludes_self(self):
        candidates = ["exp", "log", "tanh"]
        similar = _ca_find_similar("exp", candidates)
        assert "exp" not in similar

    def test_find_similar_sorted_by_distance(self):
        candidates = ["ab", "abc", "abcd", "xyz"]
        similar = _ca_find_similar("ab", candidates, max_distance=2)
        # "abc" (dist=1) should come before "abcd" (dist=2)
        assert similar.index("abc") < similar.index("abcd")

    def test_find_similar_limits_results(self):
        # Generate many candidates within distance
        candidates = [f"a{chr(ord('a') + i)}" for i in range(10)]
        similar = _ca_find_similar("aa", candidates, max_distance=2)
        assert len(similar) <= 5

    def test_code_action_unknown_function(self):
        # 'epx' is not defined — should produce a diagnostic
        source = "fn f(x: f32) -> f32 { epx(x) }"
        diags, result = validate(source, "<test>")
        # There should be a diagnostic about undefined function 'epx'
        assert len(diags) >= 1
        fn_diag = [d for d in diags if "epx" in d.message]
        assert len(fn_diag) >= 1

        # Simulate code action request
        uri = "file:///test.mao"
        _cache[uri] = result
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=fn_diag[0].range,
            context=types.CodeActionContext(diagnostics=fn_diag),
        )
        actions = code_actions(None, params)
        assert actions is not None
        titles = [a.title for a in actions]
        # Should suggest 'exp' as a fix for 'epx'
        assert any("exp" in t for t in titles)

    def test_code_action_unknown_struct(self):
        source = """
struct Point { x: f32, y: f32 }
fn f() -> Point { Pint { x: 1.0, y: 2.0 } }
"""
        diags, result = validate(source, "<test>")
        struct_diag = [d for d in diags if "Pint" in d.message]
        assert len(struct_diag) >= 1

        uri = "file:///test_struct.mao"
        _cache[uri] = result
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=struct_diag[0].range,
            context=types.CodeActionContext(diagnostics=struct_diag),
        )
        actions = code_actions(None, params)
        assert actions is not None
        titles = [a.title for a in actions]
        assert any("Point" in t for t in titles)

    def test_code_action_undefined_variable(self):
        source = "fn f(value: f32) -> f32 { valeu }"
        diags, result = validate(source, "<test>")
        var_diag = [d for d in diags if "valeu" in d.message]
        assert len(var_diag) >= 1

        uri = "file:///test_var.mao"
        _cache[uri] = result
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=var_diag[0].range,
            context=types.CodeActionContext(diagnostics=var_diag),
        )
        actions = code_actions(None, params)
        # 'valeu' vs 'value' is distance 2, should be suggested
        assert actions is not None
        titles = [a.title for a in actions]
        assert any("value" in t for t in titles)

    def test_code_action_no_suggestions_for_distant_name(self):
        source = "fn f(x: f32) -> f32 { zzzzzzzzz(x) }"
        diags, result = validate(source, "<test>")
        bad_diag = [d for d in diags if "zzzzzzzzz" in d.message]
        assert len(bad_diag) >= 1

        uri = "file:///test_none.mao"
        _cache[uri] = result
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=bad_diag[0].range,
            context=types.CodeActionContext(diagnostics=bad_diag),
        )
        actions = code_actions(None, params)
        # No close matches, should return None
        assert actions is None

    def test_code_action_returns_none_without_cache(self):
        uri = "file:///nonexistent.mao"
        _cache.pop(uri, None)  # Ensure not cached
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=1),
            ),
            context=types.CodeActionContext(diagnostics=[]),
        )
        actions = code_actions(None, params)
        assert actions is None

    def test_code_action_quickfix_has_edit(self):
        source = "fn f(x: f32) -> f32 { epx(x) }"
        diags, result = validate(source, "<test>")
        fn_diag = [d for d in diags if "epx" in d.message]
        assert len(fn_diag) >= 1

        uri = "file:///test_edit.mao"
        _cache[uri] = result
        params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=fn_diag[0].range,
            context=types.CodeActionContext(diagnostics=fn_diag),
        )
        actions = code_actions(None, params)
        assert actions is not None
        # Find the 'exp' suggestion
        exp_action = [a for a in actions if "exp" in a.title][0]
        assert exp_action.kind == types.CodeActionKind.QuickFix
        assert exp_action.edit is not None
        assert uri in exp_action.edit.changes
        edits = exp_action.edit.changes[uri]
        assert len(edits) == 1
        assert edits[0].new_text == "exp"


# ---------------------------------------------------------------------------
# Folding range tests
# ---------------------------------------------------------------------------

class TestFoldingRanges:
    def test_multiline_function_foldable(self):
        source = """fn f(x: f32) -> f32 {
    let a = x;
    a
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        assert len(ranges) >= 1
        # Should have a range starting at line 0 and ending at line 3
        fn_ranges = [r for r in ranges if r.start_line == 0 and r.end_line == 3]
        assert len(fn_ranges) == 1

    def test_struct_foldable(self):
        source = """struct Point {
    x: f32,
    y: f32
}
fn f(p: Point) -> f32 { p.x }"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Should include a range for the struct definition
        struct_ranges = [r for r in ranges if r.start_line == 0]
        assert len(struct_ranges) >= 1

    def test_single_line_not_foldable(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Single-line function should NOT produce a folding range
        fn_ranges = [r for r in ranges if r.start_line == 0]
        assert len(fn_ranges) == 0

    def test_nested_blocks(self):
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0 - x
    }
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Should have ranges for the function AND the if/else blocks
        assert len(ranges) >= 2

    def test_scan_body_foldable(self):
        source = """fn f(xs: f32[10]) -> f32 {
    scan (carry, x) in (0.0, xs) {
        carry + x
    }
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        assert len(ranges) >= 2  # function + scan

    def test_map_body_foldable(self):
        source = """fn f(xs: f32[10]) -> f32[10] {
    map x in xs {
        x + 1.0
    }
}"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        assert len(ranges) >= 2  # function + map

    def test_empty_result_returns_empty(self):
        ranges = _build_folding_ranges(AnalysisResult(None, {}, {}, {}))
        assert ranges == []

    def test_single_line_struct_not_foldable(self):
        source = """struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { p.x }"""
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        # Single-line struct should NOT produce a folding range
        struct_ranges = [r for r in ranges if r.start_line == 0]
        assert len(struct_ranges) == 0


# ---------------------------------------------------------------------------
# Selection Range tests
# ---------------------------------------------------------------------------


class TestSelectionRange:
    def test_identifier_has_parent_chain(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        # Identifier "x" in body — should have ancestors: FnDef -> Block -> Identifier
        body_expr = fn.body.expr  # Identifier("x")
        ancestors = []
        _sel_collect_ancestors(fn, body_expr.span.line_start, body_expr.span.col_start, ancestors)
        assert len(ancestors) >= 2  # at least FnDef and Identifier
        # Innermost should be the Identifier
        chain = _sel_build_chain(ancestors)
        assert chain is not None
        # chain.range should be the Identifier's range
        # chain.parent should exist (Block or FnDef)
        assert chain.parent is not None

    def test_binop_expands_through_chain(self):
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        body = fn.body.expr  # BinOp
        left = body.left     # Identifier("a")
        ancestors = []
        _sel_collect_ancestors(fn, left.span.line_start, left.span.col_start, ancestors)
        # Should have: FnDef -> Block -> BinOp -> Identifier
        assert len(ancestors) >= 3
        chain = _sel_build_chain(ancestors)
        # Innermost is Identifier, parent is BinOp, grandparent is Block or FnDef
        assert chain.parent is not None
        assert chain.parent.parent is not None

    def test_outside_function_returns_fallback(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        # Position way outside any function
        ancestors = []
        for fn in result.program.functions:
            _sel_collect_ancestors(fn, 99, 1, ancestors)
        assert len(ancestors) == 0

    def test_multiple_positions(self):
        source = "fn f(a: f32, b: f32) -> f32 { a + b }"
        _, result = validate(source, "<test>")
        fn = result.program.functions[0]
        body = fn.body.expr  # BinOp

        # Two positions: on "a" and on "b"
        ancestors_a = []
        _sel_collect_ancestors(fn, body.left.span.line_start, body.left.span.col_start, ancestors_a)
        ancestors_b = []
        _sel_collect_ancestors(fn, body.right.span.line_start, body.right.span.col_start, ancestors_b)

        chain_a = _sel_build_chain(ancestors_a)
        chain_b = _sel_build_chain(ancestors_b)

        assert chain_a is not None
        assert chain_b is not None
        # Both should have the same outermost parent (FnDef)


# ---------------------------------------------------------------------------
# Doc comments and builtin docs
# ---------------------------------------------------------------------------

class TestDocComments:
    def test_builtin_docs_exist_for_all_builtins(self):
        from maomi.lsp import _BUILTINS
        for b in _BUILTINS:
            assert b in _BUILTIN_DOCS, f"Missing doc for builtin '{b}'"

    def test_hover_builtin_call_shows_docs(self):
        source = "fn f(x: f32) -> f32 { exp(x) }"
        diags, result = validate(source, "<test>")
        fn = result.program.functions[0]
        call_node = fn.body.expr  # exp(x)
        assert isinstance(call_node, CallExpr)
        hover = _get_hover_text(call_node, fn, result)
        assert hover is not None
        assert "exp" in hover
        assert "e^x" in hover  # from _BUILTIN_DOCS

    def test_hover_user_fn_call_shows_docs(self):
        source = "/// Double the input.\nfn double(x: f32) -> f32 { x + x }\nfn main(x: f32) -> f32 { double(x) }"
        diags, result = validate(source, "<test>")
        main_fn = result.program.functions[1]
        call_node = main_fn.body.expr  # double(x)
        assert isinstance(call_node, CallExpr)
        hover = _get_hover_text(call_node, main_fn, result)
        assert hover is not None
        assert "double" in hover
        assert "Double the input." in hover

    def test_hover_fndef_shows_doc(self):
        source = "/// Compute sum.\nfn mysum(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        fn = result.program.functions[0]
        hover = _get_hover_text(fn, fn, result)
        assert hover is not None
        assert "Compute sum." in hover

    def test_hover_fndef_without_doc(self):
        source = "fn mysum(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        fn = result.program.functions[0]
        hover = _get_hover_text(fn, fn, result)
        assert hover is not None
        assert "mysum" in hover
        # No doc section
        assert hover.count("\n\n") == 0 or "```" in hover

    def test_completion_builtin_has_docs(self):
        result_list = _complete_general(None, types.Position(line=0, character=0))
        exp_item = [i for i in result_list.items if i.label == "exp"][0]
        assert exp_item.documentation is not None
        assert "e^x" in exp_item.documentation.value

    def test_completion_user_fn_has_docs(self):
        source = "/// My func.\nfn myfunc(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        result_list = _complete_general(result, types.Position(line=1, character=0))
        myfunc_item = [i for i in result_list.items if i.label == "myfunc"][0]
        assert myfunc_item.documentation is not None
        assert "My func." in myfunc_item.documentation.value

    def test_completion_user_fn_no_doc(self):
        source = "fn myfunc(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        result_list = _complete_general(result, types.Position(line=0, character=0))
        myfunc_item = [i for i in result_list.items if i.label == "myfunc"][0]
        assert myfunc_item.documentation is None

    def test_completion_struct_has_docs(self):
        source = "/// A point.\nstruct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        diags, result = validate(source, "<test>")
        result_list = _complete_general(result, types.Position(line=2, character=0))
        pt_item = [i for i in result_list.items if i.label == "Point"][0]
        assert pt_item.documentation is not None
        assert "A point." in pt_item.documentation.value

    def test_signature_help_builtin_has_doc(self):
        sh = _build_signature_help("exp", ["x"], ["f32"], "f32", 0, doc="Compute e^x.")
        assert sh.signatures[0].documentation is not None
        assert "e^x" in sh.signatures[0].documentation.value

    def test_signature_help_no_doc(self):
        sh = _build_signature_help("myfn", ["x"], ["f32"], "f32", 0)
        assert sh.signatures[0].documentation is None


# ---------------------------------------------------------------------------
# Workspace Symbols
# ---------------------------------------------------------------------------

class TestWorkspaceSymbols:
    def test_search_function_name(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source1 = "fn add(a: f32, b: f32) -> f32 { a + b }"
            _, result1 = validate(source1, "<test1>")
            _cache["file:///test1.mao"] = result1

            source2 = "struct Point { x: f32, y: f32 }\nfn distance(p: Point) -> f32 { p.x }"
            _, result2 = validate(source2, "<test2>")
            _cache["file:///test2.mao"] = result2

            symbols = _workspace_symbols("add")
            assert len(symbols) == 1
            assert symbols[0].name == "add"
            assert symbols[0].kind == types.SymbolKind.Function
            assert symbols[0].location.uri == "file:///test1.mao"
        finally:
            _cache.clear()
            _cache.update(old_cache)

    def test_search_struct_name(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
            _, result = validate(source, "<test>")
            _cache["file:///test.mao"] = result

            symbols = _workspace_symbols("Point")
            assert len(symbols) == 1
            assert symbols[0].name == "Point"
            assert symbols[0].kind == types.SymbolKind.Struct
        finally:
            _cache.clear()
            _cache.update(old_cache)

    def test_case_insensitive(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
            _, result = validate(source, "<test>")
            _cache["file:///test.mao"] = result

            symbols = _workspace_symbols("point")
            assert len(symbols) == 1
            assert symbols[0].name == "Point"
        finally:
            _cache.clear()
            _cache.update(old_cache)

    def test_empty_query_returns_all(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source1 = "fn add(a: f32, b: f32) -> f32 { a + b }"
            _, result1 = validate(source1, "<test1>")
            _cache["file:///test1.mao"] = result1

            source2 = "struct Point { x: f32, y: f32 }\nfn distance(p: Point) -> f32 { p.x }"
            _, result2 = validate(source2, "<test2>")
            _cache["file:///test2.mao"] = result2

            symbols = _workspace_symbols("")
            names = {s.name for s in symbols}
            assert "add" in names
            assert "Point" in names
            assert "distance" in names
            assert len(symbols) == 3
        finally:
            _cache.clear()
            _cache.update(old_cache)

    def test_no_matches(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source = "fn add(a: f32, b: f32) -> f32 { a + b }"
            _, result = validate(source, "<test>")
            _cache["file:///test.mao"] = result

            symbols = _workspace_symbols("nonexistent")
            assert symbols == []
        finally:
            _cache.clear()
            _cache.update(old_cache)

    def test_multiple_files_matching(self):
        old_cache = dict(_cache)
        _cache.clear()
        try:
            source1 = "fn compute_add(a: f32, b: f32) -> f32 { a + b }"
            _, result1 = validate(source1, "<test1>")
            _cache["file:///test1.mao"] = result1

            source2 = "fn compute_mul(a: f32, b: f32) -> f32 { a * b }"
            _, result2 = validate(source2, "<test2>")
            _cache["file:///test2.mao"] = result2

            symbols = _workspace_symbols("compute")
            assert len(symbols) == 2
            names = {s.name for s in symbols}
            assert "compute_add" in names
            assert "compute_mul" in names
        finally:
            _cache.clear()
            _cache.update(old_cache)
