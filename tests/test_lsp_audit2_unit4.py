"""Tests for LSP audit Unit 4: folding, code lens, goto-def field span."""

from lsprotocol import types

from maomi.lsp import (
    validate,
    _build_folding_ranges,
    _build_code_lenses,
    _goto_find_definition,
    _children_of,
    _local_functions,
    AnalysisResult,
)
from maomi.ast_nodes import FieldAccess


def _find_node_by_type_and_attr(root, node_type, attr_name, attr_value):
    """Recursively find a node of given type with matching attribute."""
    if isinstance(root, node_type):
        if getattr(root, attr_name, None) == attr_value:
            return root
    for child in _children_of(root):
        result = _find_node_by_type_and_attr(child, node_type, attr_name, attr_value)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# G3: Doc comment folding
# ---------------------------------------------------------------------------

class TestDocCommentFolding:
    def test_multiline_doc_comment_produces_comment_range(self):
        source = (
            "/// This function adds two numbers.\n"
            "/// It returns their sum.\n"
            "/// Pure and simple.\n"
            "fn add(x: f32, y: f32) -> f32 { x + y }"
        )
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        comment_ranges = [
            r for r in ranges if r.kind == types.FoldingRangeKind.Comment
        ]
        assert len(comment_ranges) == 1
        assert comment_ranges[0].start_line == 0
        assert comment_ranges[0].end_line == 2

    def test_single_doc_comment_no_folding(self):
        source = (
            "/// Just one line.\n"
            "fn f(x: f32) -> f32 { x }"
        )
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        comment_ranges = [
            r for r in ranges if r.kind == types.FoldingRangeKind.Comment
        ]
        assert len(comment_ranges) == 0

    def test_two_separate_doc_comment_blocks(self):
        source = (
            "/// Doc for f.\n"
            "/// More about f.\n"
            "fn f(x: f32) -> f32 { x }\n"
            "/// Doc for g.\n"
            "/// More about g.\n"
            "fn g(x: f32) -> f32 { x }"
        )
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        comment_ranges = [
            r for r in ranges if r.kind == types.FoldingRangeKind.Comment
        ]
        assert len(comment_ranges) == 2


# ---------------------------------------------------------------------------
# G4: Import block folding
# ---------------------------------------------------------------------------

class TestImportBlockFolding:
    def test_two_imports_produce_imports_range(self):
        source = (
            "import math;\n"
            "from math import { normalize };\n"
            "fn f(x: f32) -> f32 { normalize(x, 0.0, 1.0) }"
        )
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        import_ranges = [
            r for r in ranges if r.kind == types.FoldingRangeKind.Imports
        ]
        assert len(import_ranges) == 1
        assert import_ranges[0].start_line == 0

    def test_single_import_no_folding(self):
        source = (
            "import math;\n"
            "fn f(x: f32) -> f32 { math.normalize(x, 0.0, 1.0) }"
        )
        _, result = validate(source, "<test>")
        ranges = _build_folding_ranges(result)
        import_ranges = [
            r for r in ranges if r.kind == types.FoldingRangeKind.Imports
        ]
        assert len(import_ranges) == 0


# ---------------------------------------------------------------------------
# G6: FieldAccess goto-def jumps to field span
# ---------------------------------------------------------------------------

class TestFieldAccessGotoDef:
    def test_field_access_jumps_to_field_not_struct(self):
        source = (
            "struct Point {\n"
            "    x: f32,\n"
            "    y: f32\n"
            "}\n"
            "fn f(p: Point) -> f32 { p.x }"
        )
        _, result = validate(source, "<test>")
        fn = _local_functions(result.program)[0]
        node = _find_node_by_type_and_attr(fn, FieldAccess, "field", "x")
        assert node is not None
        found = _goto_find_definition(node, fn, result)
        assert found is not None
        span, _ = found
        sdef = result.program.struct_defs[0]
        assert span != sdef.span, "Should jump to field span, not struct span"
        assert span == sdef.field_name_spans[0], "Should jump to field 'x' span"

    def test_field_access_second_field(self):
        source = (
            "struct Point {\n"
            "    x: f32,\n"
            "    y: f32\n"
            "}\n"
            "fn f(p: Point) -> f32 { p.y }"
        )
        _, result = validate(source, "<test>")
        fn = _local_functions(result.program)[0]
        node = _find_node_by_type_and_attr(fn, FieldAccess, "field", "y")
        assert node is not None
        found = _goto_find_definition(node, fn, result)
        assert found is not None
        span, _ = found
        sdef = result.program.struct_defs[0]
        assert span == sdef.field_name_spans[1], "Should jump to field 'y' span"


# ---------------------------------------------------------------------------
# E8: Run lens excluded for struct-param functions
# ---------------------------------------------------------------------------

class TestCodeLensStructParams:
    def test_struct_param_no_run_lens(self):
        source = (
            "struct Point { x: f32, y: f32 }\n"
            "fn f(p: Point) -> f32 { p.x }"
        )
        _, result = validate(source, "<test>")
        lenses = _build_code_lenses(result, "file:///test.mao")
        run_lenses = [l for l in lenses if l.command.title == "\u25b6 Run"]
        assert len(run_lenses) == 0, "Functions with StructType params should not get Run lens"

    def test_concrete_params_still_get_run_lens(self):
        source = "fn f(x: f32, y: f32) -> f32 { x + y }"
        _, result = validate(source, "<test>")
        lenses = _build_code_lenses(result, "file:///test.mao")
        run_lenses = [l for l in lenses if l.command.title == "\u25b6 Run"]
        assert len(run_lenses) == 1, "Functions with concrete params should get Run lens"

    def test_no_params_still_gets_run_lens(self):
        source = "fn f() -> f32 { 1.0 }"
        _, result = validate(source, "<test>")
        lenses = _build_code_lenses(result, "file:///test.mao")
        run_lenses = [l for l in lenses if l.command.title == "\u25b6 Run"]
        assert len(run_lenses) == 1
