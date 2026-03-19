"""Tests for LSP audit round 2 — Unit B: Rename Safety + Hover."""

from maomi.lsp import (
    validate, prepare_rename_at, rename_at, _get_hover_text,
    classify_symbol,
)
from maomi.ast_nodes import FieldAccess, Identifier, Span


# ---------------------------------------------------------------------------
# B2: rename_at rejects keywords and invalid identifiers
# ---------------------------------------------------------------------------

class TestRenameRejectsKeywords:
    def _validate(self, source):
        diags, result = validate(source, "<test>")
        return result

    def test_rename_to_keyword_fn_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        # cursor on 'x' param (line 0, col 5 in 0-indexed)
        edits = rename_at(source, result, 0, 5, "fn")
        assert edits is None

    def test_rename_to_keyword_let_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "let")
        assert edits is None

    def test_rename_to_keyword_if_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "if")
        assert edits is None

    def test_rename_to_type_name_f32_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "f32")
        assert edits is None

    def test_rename_to_type_name_bool_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "bool")
        assert edits is None

    def test_rename_to_empty_string_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "")
        assert edits is None

    def test_rename_to_invalid_identifier_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "123abc")
        assert edits is None

    def test_rename_to_identifier_with_spaces_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "a b")
        assert edits is None

    def test_rename_to_valid_identifier_succeeds(self):
        source = "fn f(x: f32) -> f32 { x }"
        result = self._validate(source)
        edits = rename_at(source, result, 0, 5, "y")
        assert edits is not None
        assert len(edits) >= 2  # param + body usage


# ---------------------------------------------------------------------------
# B3: Rename shadowed variable doesn't cross scopes
# ---------------------------------------------------------------------------

class TestRenameShadowedVariable:
    def _validate(self, source):
        diags, result = validate(source, "<test>")
        return result

    def test_rename_param_does_not_affect_shadowed_let(self):
        source = "fn f(x: f32) -> f32 {\n  let x = 2.0;\n  x\n}"
        result = self._validate(source)
        # Cursor on 'x' param (line 0, col 5 in 0-indexed)
        edits = rename_at(source, result, 0, 5, "y")
        assert edits is not None
        # Should only rename the param 'x', not the let 'x' or the final 'x'
        # The param is at line 0 col 5, and nothing else
        renamed_positions = [(e.range.start.line, e.range.start.character) for e in edits]
        assert (0, 5) in renamed_positions  # param
        # The let x on line 1 and the use on line 2 should NOT be renamed
        assert (1, 6) not in renamed_positions  # let x = ...
        assert (2, 2) not in renamed_positions  # trailing x

    def test_rename_let_binding_when_param_has_same_name(self):
        """Renaming the inner let x should not affect the param x."""
        source = "fn f(x: f32) -> f32 {\n  let x = 2.0;\n  x\n}"
        result = self._validate(source)
        # Cursor on the inner 'x' in let x = 2.0 (line 1, col 6 in 0-indexed)
        edits = rename_at(source, result, 1, 6, "y")
        assert edits is not None
        renamed_positions = [(e.range.start.line, e.range.start.character) for e in edits]
        # Should rename the let binding and the usage after it, but NOT the param
        assert (0, 5) not in renamed_positions  # param x should not be touched

    def test_rename_param_used_before_shadow(self):
        """Param x used before shadow should be renamed; usage after shadow should not."""
        source = "fn f(x: f32) -> f32 {\n  let a = x;\n  let x = 2.0;\n  x\n}"
        result = self._validate(source)
        # Cursor on 'x' param (line 0, col 5)
        edits = rename_at(source, result, 0, 5, "y")
        assert edits is not None
        renamed_positions = [(e.range.start.line, e.range.start.character) for e in edits]
        assert (0, 5) in renamed_positions  # param decl
        assert (1, 10) in renamed_positions  # let a = x (usage before shadow)
        assert (2, 6) not in renamed_positions  # let x = 2.0 (shadow point)
        assert (3, 2) not in renamed_positions  # trailing x (after shadow)


# ---------------------------------------------------------------------------
# B5: classify_symbol on FieldAccess returns (field_name, "field")
# ---------------------------------------------------------------------------

class TestClassifyFieldAccess:
    def test_field_access_returns_field_kind(self):
        fa = FieldAccess(
            object=Identifier(name="p", span=Span(1, 1, 1, 1)),
            field="x",
            span=Span(1, 1, 1, 3),
        )
        name, kind = classify_symbol(fa)
        assert name == "x"
        assert kind == "field"

    def test_field_access_nested(self):
        inner = FieldAccess(
            object=Identifier(name="p", span=Span(1, 1, 1, 1)),
            field="inner",
            span=Span(1, 1, 1, 7),
        )
        outer = FieldAccess(
            object=inner,
            field="x",
            span=Span(1, 1, 1, 9),
        )
        name, kind = classify_symbol(outer)
        assert name == "x"
        assert kind == "field"


# ---------------------------------------------------------------------------
# B7: Param hover shows wildcard f32[..]
# ---------------------------------------------------------------------------

class TestParamHoverWildcard:
    def test_param_hover_wildcard_type(self):
        source = "fn f(x: f32[..]) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        param = fn.params[0]
        hover_text = _get_hover_text(param, fn, result)
        assert hover_text is not None
        assert "f32[..]" in hover_text

    def test_param_hover_normal_type_still_works(self):
        source = "fn f(x: f32[3, 4]) -> f32 { sum(x) }"
        diags, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        param = fn.params[0]
        hover_text = _get_hover_text(param, fn, result)
        assert hover_text is not None
        assert "f32[3, 4]" in hover_text

    def test_param_hover_scalar_type(self):
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        param = fn.params[0]
        hover_text = _get_hover_text(param, fn, result)
        assert hover_text is not None
        assert "f32" in hover_text
        assert "[" not in hover_text  # no dims for scalar


# ---------------------------------------------------------------------------
# B8: Param hover shows comptime prefix
# ---------------------------------------------------------------------------

class TestParamHoverComptime:
    def test_param_hover_comptime_prefix(self):
        source = "fn f(x: f32[3, 4], comptime axis: i32) -> f32[3] { sum(x, axis=axis) }"
        diags, result = validate(source, "<test>")
        assert result.program is not None
        fn = result.program.functions[0]
        # Find the comptime param
        comptime_param = None
        for p in fn.params:
            if p.name == "axis":
                comptime_param = p
                break
        assert comptime_param is not None
        assert comptime_param.comptime  # truthy (Token or True)
        hover_text = _get_hover_text(comptime_param, fn, result)
        assert hover_text is not None
        assert "comptime " in hover_text
        assert "axis: i32" in hover_text

    def test_param_hover_non_comptime_no_prefix(self):
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        fn = result.program.functions[0]
        param = fn.params[0]
        hover_text = _get_hover_text(param, fn, result)
        assert hover_text is not None
        assert "comptime" not in hover_text


# ---------------------------------------------------------------------------
# B9: prepare_rename_at on struct type annotation returns struct name range
# ---------------------------------------------------------------------------

class TestPrepareRenameStructAnnotation:
    def test_cursor_on_struct_type_in_param_returns_struct_name(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }"
        diags, result = validate(source, "<test>")
        assert result.program is not None
        # Cursor on "Point" in the param type annotation (line 1, col 8 in 0-indexed)
        rng = prepare_rename_at(source, result, 1, 8)
        assert rng is not None
        # The range should cover "Point" (5 chars), not "p" (1 char)
        length = rng.end.character - rng.start.character
        assert length == 5  # len("Point")
