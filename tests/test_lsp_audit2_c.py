"""Tests for LSP audit round 2, Unit C: Crash Fixes + Correctness."""

from maomi.lsp import (
    validate,
    _call_hierarchy_prepare,
    _build_code_lenses,
    _find_matching_brace,
    _goto_find_definition,
    _build_folding_ranges,
    _goto_type_definition,
    _find_node_at,
)
from maomi.lsp._core import _EMPTY_RESULT
from maomi.ast_nodes import Identifier, Param


# ---------------------------------------------------------------------------
# C2: Call hierarchy double coordinate conversion
# ---------------------------------------------------------------------------

class TestCallHierarchyCoordinates:
    def test_prepare_with_0_indexed_input(self):
        """_call_hierarchy_prepare receives 0-indexed coords and converts once."""
        source = "fn foo(x: f32) -> f32 { x }\nfn bar(x: f32) -> f32 { foo(x) }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        # "foo" is defined at line 1, col 4 (1-indexed).
        # 0-indexed: line=0, col=3
        items = _call_hierarchy_prepare(result, "file:///test.mao", 0, 3)
        assert items is not None
        assert len(items) == 1
        assert items[0].name == "foo"

    def test_prepare_on_call_site(self):
        """Prepare on a call site returns the called function."""
        source = "fn foo(x: f32) -> f32 { x }\nfn bar(x: f32) -> f32 { foo(x) }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        # "foo(" call is on line 2, somewhere around col 25-28 (1-indexed).
        # 0-indexed: line=1. Find position of "foo" in second line.
        line2 = source.split("\n")[1]
        col_0 = line2.index("foo(")
        items = _call_hierarchy_prepare(result, "file:///test.mao", 1, col_0)
        assert items is not None
        assert items[0].name == "foo"


# ---------------------------------------------------------------------------
# C3: Code lens crash on None program
# ---------------------------------------------------------------------------

class TestCodeLensNoneProgram:
    def test_none_program_returns_empty(self):
        """_build_code_lenses with None program returns [] instead of crashing."""
        result = _EMPTY_RESULT  # program is None
        lenses = _build_code_lenses(result, "file:///test.mao")
        assert lenses == []


# ---------------------------------------------------------------------------
# C4: _find_matching_brace ignores strings/comments
# ---------------------------------------------------------------------------

class TestFindMatchingBraceStrings:
    def test_brace_in_string_ignored(self):
        """Braces inside string literals should not count for matching."""
        # The real { is at col 0, the } in the string should be ignored,
        # and the matching } is at the end.
        source = '{ "}" }'
        pos = _find_matching_brace(source, 0, 0)
        assert pos is not None
        assert pos.character == 6  # The closing } at position 6

    def test_brace_in_comment_ignored(self):
        """Braces inside comments should not count for matching."""
        source = "{\n// }\n}"
        pos = _find_matching_brace(source, 0, 0)
        assert pos is not None
        assert pos.line == 2
        assert pos.character == 0

    def test_normal_matching_still_works(self):
        """Normal brace matching without strings/comments still works."""
        source = "fn f() { if true { x } }"
        # Find the first {
        col = source.index("{")
        pos = _find_matching_brace(source, 0, col)
        assert pos is not None
        assert pos.character == len(source) - 1


# ---------------------------------------------------------------------------
# B4: Shadowed variable goto-def
# ---------------------------------------------------------------------------

class TestShadowedVariableGotoDef:
    def test_let_shadows_param(self):
        """When let shadows a param, goto-def should go to the let stmt."""
        source = "fn f(x: f32) -> f32 { let x = 2.0; x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        fn = result.program.functions[0]
        # The trailing x is at line 1, col 37 (1-indexed)
        trailing_x_col = source.rindex("x") + 1  # 1-indexed
        node = _find_node_at(fn, 1, trailing_x_col)
        assert node is not None
        assert isinstance(node, Identifier)
        assert node.name == "x"

        found = _goto_find_definition(node, fn, result)
        assert found is not None
        defn_span, _ = found

        # Should point to the let stmt, not the param.
        # The param "x" is at col 5 (1-indexed), the let stmt starts later.
        param_col = source.index("(x:") + 2  # 1-indexed col of param "x"
        let_col = source.index("let x") + 1  # 1-indexed col of "let"
        assert defn_span.col_start >= let_col, (
            f"Expected goto-def to point to let (col>={let_col}), "
            f"got col={defn_span.col_start} (param is at col {param_col})"
        )


# ---------------------------------------------------------------------------
# B6: Goto-def for Param type annotations
# ---------------------------------------------------------------------------

class TestParamTypeAnnotationGotoDef:
    def test_param_struct_type_goto_def(self):
        """Goto-def on a Param with struct type annotation navigates to struct def."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        fn = result.program.functions[0]
        param_node = fn.params[0]
        assert isinstance(param_node, Param)
        assert param_node.type_annotation.base == "Point"

        found = _goto_find_definition(param_node, fn, result)
        assert found is not None
        defn_span, source_file = found
        # Should point to the struct def on line 1
        assert defn_span.line_start == 1


# ---------------------------------------------------------------------------
# B10/B11: Duplicate error reporting
# ---------------------------------------------------------------------------

class TestDuplicateErrors:
    def test_unknown_param_type_single_error(self):
        """Unknown type in param should produce exactly 1 error, not 2."""
        source = "fn f(x: Foo) -> f32 { 1.0 }"
        diags, result = validate(source, "<test>")
        # Filter for the "unknown type: 'Foo'" diagnostic
        foo_diags = [d for d in diags if "Foo" in d.message and "unknown" in d.message.lower()]
        assert len(foo_diags) == 1

    def test_unknown_struct_field_type_single_error(self):
        """Unknown type in struct field should produce exactly 1 error."""
        source = "struct S { x: Foo }\nfn f(s: S) -> f32 { 1.0 }"
        diags, result = validate(source, "<test>")
        foo_diags = [d for d in diags if "Foo" in d.message.lower() or "unknown" in d.message.lower()]
        # Should not have duplicate errors for the same location
        seen = set()
        for d in foo_diags:
            key = (d.range.start.line, d.range.start.character, d.message)
            assert key not in seen, f"Duplicate diagnostic: {d.message}"
            seen.add(key)


# ---------------------------------------------------------------------------
# B12: Duplicate struct names
# ---------------------------------------------------------------------------

class TestDuplicateStructNames:
    def test_duplicate_struct_name_error(self):
        """Two structs with the same name should produce an error."""
        source = "struct S { x: f32 }\nstruct S { y: i32 }\nfn f(a: f32) -> f32 { a }"
        diags, result = validate(source, "<test>")
        dup_diags = [d for d in diags if "duplicate struct" in d.message.lower()]
        assert len(dup_diags) >= 1
        assert "S" in dup_diags[0].message


# ---------------------------------------------------------------------------
# B13: Self-referential type alias
# ---------------------------------------------------------------------------

class TestSelfReferentialTypeAlias:
    def test_self_referential_alias_error(self):
        """type A = A; should produce an error."""
        source = "type A = A;\nfn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        self_ref_diags = [d for d in diags if "self-referential" in d.message.lower()]
        assert len(self_ref_diags) >= 1
        assert "A" in self_ref_diags[0].message


# ---------------------------------------------------------------------------
# B16: Duplicate folding ranges
# ---------------------------------------------------------------------------

class TestDuplicateFoldingRanges:
    def test_scan_no_duplicate_ranges(self):
        """Scan/fold/map body blocks should not produce duplicate folding ranges."""
        source = """fn f(x: f32[10], init: f32) -> f32 {
    scan (carry, elem) in (init, x) {
        carry + elem
    }
}"""
        diags, result = validate(source, "<test>")
        assert result.program is not None
        ranges = _build_folding_ranges(result)
        # Check no exact duplicates
        range_tuples = [(r.start_line, r.end_line) for r in ranges]
        assert len(range_tuples) == len(set(range_tuples)), \
            f"Duplicate folding ranges found: {range_tuples}"

    def test_map_no_duplicate_ranges(self):
        """Map body blocks should not produce duplicate folding ranges."""
        source = """fn f(x: f32[5]) -> f32[5] {
    map elem in x {
        elem * 2.0
    }
}"""
        diags, result = validate(source, "<test>")
        assert result.program is not None
        ranges = _build_folding_ranges(result)
        range_tuples = [(r.start_line, r.end_line) for r in ranges]
        assert len(range_tuples) == len(set(range_tuples)), \
            f"Duplicate folding ranges found: {range_tuples}"

    def test_fold_no_duplicate_ranges(self):
        """Fold body blocks should not produce duplicate folding ranges."""
        source = """fn f(x: f32[10], init: f32) -> f32 {
    fold (carry, elem) in (init, x) {
        carry + elem
    }
}"""
        diags, result = validate(source, "<test>")
        assert result.program is not None
        ranges = _build_folding_ranges(result)
        range_tuples = [(r.start_line, r.end_line) for r in ranges]
        assert len(range_tuples) == len(set(range_tuples)), \
            f"Duplicate folding ranges found: {range_tuples}"


# ---------------------------------------------------------------------------
# B17: StructArrayType in goto-type-def
# ---------------------------------------------------------------------------

class TestStructArrayTypeTypeDef:
    def test_struct_array_type_goto_type_def(self):
        """Goto-type-def on a StructArrayType variable should work."""
        source = "struct Point { x: f32, y: f32 }\nfn f(ps: Point[4]) -> f32 { ps.x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        fn = result.program.functions[0]
        param_node = fn.params[0]
        assert isinstance(param_node, Param)

        span = _goto_type_definition(param_node, fn, result)
        assert span is not None
        assert span.line_start == 1  # struct Point is on line 1

    def test_struct_array_identifier_goto_type_def(self):
        """Goto-type-def on an identifier with StructArrayType should work."""
        source = "struct Point { x: f32, y: f32 }\nfn f(ps: Point[4]) -> f32 { ps.x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None

        fn = result.program.functions[0]
        # Find the "ps" identifier in the body (ps.x)
        line2 = source.split("\n")[1]
        ps_col = line2.index("ps.x") + 1  # 1-indexed
        node = _find_node_at(fn, 2, ps_col)
        assert node is not None
        # Should not crash on StructArrayType; may return struct def or None
        # depending on exact node found (FieldAccess vs Identifier)
        _goto_type_definition(node, fn, result)
