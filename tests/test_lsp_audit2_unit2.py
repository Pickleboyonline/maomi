"""Tests for LSP audit round 2, Unit 2: References + Call Hierarchy fixes."""

from maomi.lsp import (
    validate,
    _refs_collect_all,
    _call_hierarchy_prepare,
    _call_hierarchy_incoming,
    _call_hierarchy_outgoing,
    _span_to_range,
    _local_functions,
)
from maomi.lsp._references import _refs_walk_node
from maomi.ast_nodes import Span


# ---------------------------------------------------------------------------
# B19: Field reference spans cover only the field name
# ---------------------------------------------------------------------------

class TestFieldReferenceNarrowSpan:
    def test_field_access_span_covers_field_name_only(self):
        """FieldAccess 'p.x' should produce a span covering 'x', not 'p.x'."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans, source_lines=source_lines)
        assert len(spans) >= 1
        for s in spans:
            span_len = s.col_end - s.col_start
            assert span_len == 1, f"Expected field name span of length 1 for 'x', got {span_len}"

    def test_struct_literal_field_span_covers_field_name_only(self):
        """StructLiteral 'Point { x: 1.0, y: 2.0 }' field 'x' span covers 'x' only."""
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans, source_lines=source_lines)
        assert len(spans) >= 1
        for s in spans:
            span_len = s.col_end - s.col_start
            assert span_len == 1, f"Expected field name span of length 1 for 'x', got {span_len}"

    def test_with_expr_field_span_covers_field_name_only(self):
        """WithExpr 'p with { x = 2.0 }' field 'x' span covers 'x' only."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point { p with { x = 2.0 } }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        spans = []
        for fn in _local_functions(result.program):
            _refs_walk_node(fn, "x", "field", spans, source_lines=source_lines)
        assert len(spans) >= 1
        for s in spans:
            span_len = s.col_end - s.col_start
            assert span_len == 1, f"Expected field name span of length 1 for 'x', got {span_len}"


# ---------------------------------------------------------------------------
# G8: Scan/map/fold/while variable declarations in refs
# ---------------------------------------------------------------------------

class TestLoopVarDeclarations:
    def test_scan_carry_var_declaration(self):
        """scan carry_var should be found as declaration in refs."""
        source = "fn f(xs: f32[10], init: f32) -> f32 {\n    scan (carry, elem) in (init, xs) {\n        carry + elem\n    }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        fn = result.program.functions[0]
        spans = _refs_collect_all(result, "carry", "variable",
                                  include_declaration=True, fn_scope=fn,
                                  source_lines=source_lines)
        # Should include declaration + body usage
        assert len(spans) >= 2
        # First span should be the declaration
        decl = spans[0]
        assert decl.col_end - decl.col_start == len("carry")

    def test_scan_elem_var_declaration(self):
        """scan elem_var should be found as declaration in refs."""
        source = "fn f(xs: f32[10], init: f32) -> f32 {\n    scan (carry, elem) in (init, xs) {\n        carry + elem\n    }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        fn = result.program.functions[0]
        spans = _refs_collect_all(result, "elem", "variable",
                                  include_declaration=True, fn_scope=fn,
                                  source_lines=source_lines)
        assert len(spans) >= 2
        decl = spans[0]
        assert decl.col_end - decl.col_start == len("elem")

    def test_map_elem_var_declaration(self):
        """map elem_var should be found as declaration in refs."""
        source = "fn f(xs: f32[5]) -> f32[5] {\n    map elem in xs {\n        elem * 2.0\n    }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        fn = result.program.functions[0]
        spans = _refs_collect_all(result, "elem", "variable",
                                  include_declaration=True, fn_scope=fn,
                                  source_lines=source_lines)
        assert len(spans) >= 2
        decl = spans[0]
        assert decl.col_end - decl.col_start == len("elem")

    def test_fold_carry_var_declaration(self):
        """fold carry_var should be found as declaration in refs."""
        source = "fn f(xs: f32[10], init: f32) -> f32 {\n    fold (carry, elem) in (init, xs) {\n        carry + elem\n    }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        fn = result.program.functions[0]
        spans = _refs_collect_all(result, "carry", "variable",
                                  include_declaration=True, fn_scope=fn,
                                  source_lines=source_lines)
        assert len(spans) >= 2
        decl = spans[0]
        assert decl.col_end - decl.col_start == len("carry")

    def test_fold_elem_var_declaration(self):
        """fold elem_var should be found as declaration in refs."""
        source = "fn f(xs: f32[10], init: f32) -> f32 {\n    fold (carry, elem) in (init, xs) {\n        carry + elem\n    }\n}"
        _, result = validate(source, "<test>")
        assert result.program is not None
        source_lines = source.splitlines()
        fn = result.program.functions[0]
        spans = _refs_collect_all(result, "elem", "variable",
                                  include_declaration=True, fn_scope=fn,
                                  source_lines=source_lines)
        assert len(spans) >= 2
        decl = spans[0]
        assert decl.col_end - decl.col_start == len("elem")


# ---------------------------------------------------------------------------
# G10: Call hierarchy selection_range is narrower than range
# ---------------------------------------------------------------------------

class TestCallHierarchySelectionRange:
    def test_selection_range_narrower_than_range(self):
        """selection_range should cover just the function name, not the full span."""
        source = "fn foo(x: f32) -> f32 { x }\nfn bar(x: f32) -> f32 { foo(x) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        items = _call_hierarchy_prepare(result, "file:///test.mao", 0, 3)
        assert items is not None
        assert len(items) == 1
        item = items[0]
        # selection_range should be narrower than range
        sr = item.selection_range
        r = item.range
        # Full range spans the entire function def
        full_width = (r.end.character - r.start.character)
        # Selection range should just cover "foo" (3 chars)
        sel_width = (sr.end.character - sr.start.character)
        assert sel_width < full_width, (
            f"selection_range width ({sel_width}) should be less than range width ({full_width})"
        )
        assert sel_width == len("foo")

    def test_incoming_hierarchy_item_has_narrow_selection(self):
        """Incoming call hierarchy items should also have narrow selection_range."""
        source = "fn foo(x: f32) -> f32 { x }\nfn bar(x: f32) -> f32 { foo(x) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        incoming = _call_hierarchy_incoming(result, "file:///test.mao", "foo")
        assert len(incoming) == 1
        item = incoming[0].from_
        sr = item.selection_range
        sel_width = sr.end.character - sr.start.character
        assert sel_width == len("bar")

    def test_outgoing_hierarchy_item_has_narrow_selection(self):
        """Outgoing call hierarchy items should also have narrow selection_range."""
        source = "fn foo(x: f32) -> f32 { x }\nfn bar(x: f32) -> f32 { foo(x) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        outgoing = _call_hierarchy_outgoing(result, "file:///test.mao", "bar")
        assert len(outgoing) == 1
        item = outgoing[0].to
        sr = item.selection_range
        sel_width = sr.end.character - sr.start.character
        assert sel_width == len("foo")


# ---------------------------------------------------------------------------
# E4: Variable with same name as function doesn't trigger call hierarchy
# ---------------------------------------------------------------------------

class TestVariableNotFunction:
    def test_variable_same_name_as_function_no_call_hierarchy(self):
        """An Identifier that matches a function name but is just a variable
        should not trigger call hierarchy."""
        source = "fn x(a: f32) -> f32 { a }\nfn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        # In function 'f', the identifier 'x' is a parameter, not a call to fn 'x'.
        # Cursor on the trailing 'x' in f's body (line 2, col ~22 0-indexed).
        line2 = source.split("\n")[1]
        # Find the last 'x' in line2 (the body return expression)
        col_0 = line2.rindex("x")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 1, col_0)
        # Should be None because 'x' is a variable, not a function call
        assert items is None

    def test_function_call_still_works(self):
        """Actual function calls should still trigger call hierarchy."""
        source = "fn x(a: f32) -> f32 { a }\nfn f(y: f32) -> f32 { x(y) }"
        _, result = validate(source, "<test>")
        assert result.program is not None
        # Cursor on 'x(' call in f's body
        line2 = source.split("\n")[1]
        col_0 = line2.index("x(")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 1, col_0)
        assert items is not None
        assert items[0].name == "x"
