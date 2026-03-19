"""Tests for reference, highlight, and classify_symbol fixes (B9, B10, G9, G11, E4)."""

from lsprotocol import types

from maomi.lsp import (
    validate, _find_node_at, _span_contains,
    _refs_classify_node, _refs_collect_all,
    classify_symbol,
    _build_document_highlights,
    _local_functions,
)
from maomi.ast_nodes import (
    Span, Param, TypeAnnotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_refs(source, line_0, col_0, include_declaration=False, kind_override=None):
    """Parse source + collect references. Returns list of (line, col)
    pairs (1-indexed) for easy assertion."""
    _, result = validate(source, "<test>")
    assert result.program is not None
    source_lines = source.splitlines()

    line = line_0 + 1  # to 1-indexed
    col = col_0 + 1

    # Check struct defs
    for sd in result.program.struct_defs:
        if _span_contains(sd.span, line, col):
            spans = _refs_collect_all(result, sd.name, kind_override or "struct",
                                      include_declaration, source_lines=source_lines)
            return spans

    # Check functions
    for fn in result.program.functions:
        node = _find_node_at(fn, line, col)
        if node is not None:
            name, kind = _refs_classify_node(node, line, col)
            if kind_override:
                kind = kind_override
            if name:
                spans = _refs_collect_all(result, name, kind,
                                          include_declaration, fn_scope=fn,
                                          source_lines=source_lines)
                return spans
    return []


# ---------------------------------------------------------------------------
# B9: GradExpr/ValueAndGradExpr wrt variable has narrow span
# ---------------------------------------------------------------------------

class TestGradWrtNarrowSpan:
    def test_grad_wrt_narrow_span(self):
        """grad(expr, var) references to var should have a narrow span, not the full grad expression."""
        source = "fn f(x: f32) -> f32 { grad(x * x, x) }"
        # Cursor on 'x' in body (first x in expression) — col 22 (0-indexed)
        spans = _find_refs(source, 0, 22, include_declaration=False)
        # Should find Identifier uses + the wrt reference
        for s in spans:
            # No span should cover the entire grad expression
            span_len = s.col_end - s.col_start
            assert span_len <= 2, f"Expected narrow span for 'x', got span length {span_len}"

    def test_grad_wrt_span_not_full_expression(self):
        """The wrt span should not be the full grad(...) expression span."""
        source = "fn f(x: f32) -> f32 { grad(x * x, x) }"
        spans = _find_refs(source, 0, 22, include_declaration=False)
        # The grad expression itself spans from col 23 to 39 (roughly)
        # The wrt 'x' should only span 1 character
        for s in spans:
            assert s.col_end - s.col_start <= 2


# ---------------------------------------------------------------------------
# B10: classify_symbol with struct_names parameter
# ---------------------------------------------------------------------------

class TestClassifySymbolStructNames:
    def test_param_type_annotation_returns_type_when_not_struct(self):
        """classify_symbol on f32 type annotation returns 'type' when struct_names provided."""
        ta = TypeAnnotation("f32", None, Span(1, 6, 1, 9))
        param = Param("x", ta, Span(1, 1, 1, 9))
        name, kind = classify_symbol(param, line=1, col=6, struct_names={"Point"})
        assert name == "f32"
        assert kind == "type"

    def test_param_type_annotation_returns_struct_when_in_struct_names(self):
        """classify_symbol on Point type annotation returns 'struct' when in struct_names."""
        ta = TypeAnnotation("Point", None, Span(1, 6, 1, 11))
        param = Param("x", ta, Span(1, 1, 1, 11))
        name, kind = classify_symbol(param, line=1, col=6, struct_names={"Point"})
        assert name == "Point"
        assert kind == "struct"

    def test_backward_compat_without_struct_names(self):
        """Without struct_names, classify_symbol still returns 'struct' for type annotations."""
        ta = TypeAnnotation("f32", None, Span(1, 6, 1, 9))
        param = Param("x", ta, Span(1, 1, 1, 9))
        name, kind = classify_symbol(param, line=1, col=6)
        assert name == "f32"
        assert kind == "struct"

    def test_backward_compat_without_line_col(self):
        """Without line/col, classify_symbol returns 'variable' for Param."""
        ta = TypeAnnotation("f32", None, Span(1, 6, 1, 9))
        param = Param("x", ta, Span(1, 1, 1, 9))
        name, kind = classify_symbol(param)
        assert name == "x"
        assert kind == "variable"


# ---------------------------------------------------------------------------
# G9: Field references across FieldAccess, StructLiteral, WithExpr
# ---------------------------------------------------------------------------

class TestFieldReferences:
    def test_field_access_reference(self):
        """FieldAccess nodes with matching field name are found."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        from maomi.lsp._references import _refs_walk_node
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans)
        assert len(spans) >= 1

    def test_struct_literal_field_reference(self):
        """StructLiteral field names matching are found."""
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        from maomi.lsp._references import _refs_walk_node
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans)
        assert len(spans) >= 1

    def test_with_expr_field_reference(self):
        """WithExpr update paths containing field name are found."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point { p with { x = 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        from maomi.lsp._references import _refs_walk_node
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans)
        assert len(spans) >= 1

    def test_field_refs_collect_all(self):
        """_refs_collect_all with kind='field' finds references across the program."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }\nfn g(p: Point) -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        spans = _refs_collect_all(result, "x", "field", include_declaration=False)
        # Should find FieldAccess p.x + StructLiteral x field
        assert len(spans) >= 2


# ---------------------------------------------------------------------------
# G11: struct-kind refs include LetStmt and struct-field type annotations
# ---------------------------------------------------------------------------

class TestStructRefsExtended:
    def test_struct_refs_include_let_type_annotation(self):
        """Struct refs should include LetStmt type annotations."""
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { let p: Point = Point { x: 1.0, y: 2.0 }; p }"
        spans = _find_refs(source, 0, 7, include_declaration=False)
        # Should include: LetStmt type annotation + return type + StructLiteral
        # Count spans that are on line 2 (where the function is)
        assert len(spans) >= 3

    def test_struct_refs_include_field_type_annotations(self):
        """Struct refs should include type annotations inside other struct definitions."""
        source = "struct Inner { v: f32 }\nstruct Outer { i: Inner }\nfn f(o: Inner) -> f32 { o.v }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        spans = _refs_collect_all(result, "Inner", "struct",
                                  include_declaration=False, source_lines=source_lines)
        # Should find: Param type annotation in f + field type annotation in Outer
        assert len(spans) >= 2


# ---------------------------------------------------------------------------
# E4: Highlight span comparison uses value equality
# ---------------------------------------------------------------------------

class TestHighlightSpanComparison:
    def test_highlights_use_value_comparison(self):
        """DocumentHighlight should use value comparison, not id(), for span matching."""
        source = "fn f(x: f32) -> f32 { x + x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        # line=1, col=23 (1-indexed) -> x in the body
        highlights = _build_document_highlights(result, 1, 23)
        assert highlights is not None
        # Should have Write (declaration) and Read (usage) highlights
        kinds = {h.kind for h in highlights}
        assert types.DocumentHighlightKind.Write in kinds
        assert types.DocumentHighlightKind.Read in kinds

    def test_declaration_is_write_highlight(self):
        """The param declaration should be highlighted as Write."""
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        highlights = _build_document_highlights(result, 1, 23)
        assert highlights is not None
        # Find the declaration highlight (param at beginning of line)
        write_highlights = [h for h in highlights
                            if h.kind == types.DocumentHighlightKind.Write]
        assert len(write_highlights) >= 1

    def test_identical_spans_match_correctly(self):
        """Two Span objects with same values should be equal (frozen dataclass)."""
        s1 = Span(1, 5, 1, 6)
        s2 = Span(1, 5, 1, 6)
        assert s1 == s2
        # They can be used in sets
        assert s1 in {s2}
