"""Tests for document symbol and selection range fixes (B5, B6, struct field selection)."""

from types import SimpleNamespace

from lsprotocol import types

from maomi.lsp import (
    validate,
    _build_document_symbols,
    _sel_build_chain,
    _span_contains,
)


def _validate(source: str):
    return validate(source, "<test>")


# ---------------------------------------------------------------------------
# B5: Struct field children should have their own ranges, not parent's
# ---------------------------------------------------------------------------

class TestStructFieldRanges:
    def test_field_children_have_own_ranges(self):
        source = "struct Point { x: f32, y: f32 }"
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        assert symbols is not None

        struct_sym = [s for s in symbols if s.name == "Point"][0]
        assert struct_sym.children is not None
        assert len(struct_sym.children) == 2

        x_child = struct_sym.children[0]
        y_child = struct_sym.children[1]

        # Field children should have different ranges from the parent struct
        assert x_child.range != struct_sym.range
        assert y_child.range != struct_sym.range

        # The two fields should have different ranges from each other
        assert x_child.range != y_child.range

    def test_field_range_covers_field_name(self):
        source = "struct Vec2 { x: f32, y: f32 }"
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        struct_sym = [s for s in symbols if s.name == "Vec2"][0]

        x_child = struct_sym.children[0]
        # "x" is 1 character wide
        assert x_child.range.end.character - x_child.range.start.character == 1

        y_child = struct_sym.children[1]
        # "y" is 1 character wide
        assert y_child.range.end.character - y_child.range.start.character == 1

    def test_multiline_struct_field_ranges(self):
        source = """\
struct Data {
    alpha: f32,
    beta: i32
}
fn f(d: Data) -> f32 { d.alpha }"""
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        struct_sym = [s for s in symbols if s.name == "Data"][0]
        assert len(struct_sym.children) == 2

        alpha_child = struct_sym.children[0]
        beta_child = struct_sym.children[1]

        # Fields on different lines should have different line numbers
        assert alpha_child.range.start.line != beta_child.range.start.line


# ---------------------------------------------------------------------------
# B6: selection_range should be narrower than range for functions and structs
# ---------------------------------------------------------------------------

class TestSelectionRangeNarrower:
    def test_function_selection_range_narrower(self):
        source = "fn my_func(x: f32) -> f32 { x }"
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        fn_sym = [s for s in symbols if s.name == "my_func"][0]

        # selection_range should cover just "my_func" (7 chars)
        sel = fn_sym.selection_range
        sel_width = sel.end.character - sel.start.character
        assert sel_width == len("my_func")

        # range should cover the entire function definition (wider)
        r = fn_sym.range
        r_width = r.end.character - r.start.character
        assert r_width > sel_width

    def test_struct_selection_range_narrower(self):
        source = "struct MyStruct { x: f32 }\nfn f(s: MyStruct) -> f32 { s.x }"
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        struct_sym = [s for s in symbols if s.name == "MyStruct"][0]

        # selection_range should cover just "MyStruct" (8 chars)
        sel = struct_sym.selection_range
        sel_width = sel.end.character - sel.start.character
        assert sel_width == len("MyStruct")

        # range should cover the entire struct definition (wider)
        r = struct_sym.range
        r_width = r.end.character - r.start.character
        assert r_width > sel_width

    def test_selection_range_points_to_name(self):
        source = "fn add(x: f32, y: f32) -> f32 { x + y }"
        _, result = _validate(source)
        symbols = _build_document_symbols(result)
        fn_sym = [s for s in symbols if s.name == "add"][0]

        sel = fn_sym.selection_range
        # "fn add" -> "add" starts at character 3
        assert sel.start.character == 3
        assert sel.end.character == 6


# ---------------------------------------------------------------------------
# Selection ranges for struct fields
# ---------------------------------------------------------------------------

class TestSelectionRangeStructFields:
    def test_selection_range_inside_struct_field(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        _, result = _validate(source)

        # "x:" first appears at character index 15 (0-indexed)
        x_col = source.index("x:")

        ancestors: list = []
        for sd in result.program.struct_defs:
            if _span_contains(sd.span, 1, x_col + 1):
                ancestors.append(sd)
                for fspan in sd.field_name_spans:
                    if _span_contains(fspan, 1, x_col + 1):
                        ancestors.append(SimpleNamespace(span=fspan))
                        break
                break

        chain = _sel_build_chain(ancestors)
        assert chain is not None
        # The innermost (last) selection range should be the field name
        assert chain.parent is not None  # should have a parent (the struct)

        # The innermost range should be narrower than the parent
        inner_width = chain.range.end.character - chain.range.start.character
        outer_width = chain.parent.range.end.character - chain.parent.range.start.character
        assert inner_width < outer_width

    def test_selection_range_y_field(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        _, result = _validate(source)

        # Find position of "y" in "y: f32"
        y_col = source.index("y:")

        ancestors: list = []
        for sd in result.program.struct_defs:
            if _span_contains(sd.span, 1, y_col + 1):
                ancestors.append(sd)
                for fspan in sd.field_name_spans:
                    if _span_contains(fspan, 1, y_col + 1):
                        ancestors.append(SimpleNamespace(span=fspan))
                        break
                break

        chain = _sel_build_chain(ancestors)
        assert chain is not None
        assert chain.parent is not None
        # Inner range covers "y" (1 char)
        assert chain.range.end.character - chain.range.start.character == 1
