"""Tests for diagnostic infrastructure (MaomiError extensions + diagnostic.py)."""

from maomi.errors import MaomiError, LexerError, ParseError, MaomiTypeError
from maomi.diagnostic import Diagnostic, Label, Level, from_error, enrich, _extract_title


# ---------------------------------------------------------------------------
# MaomiError backward compatibility
# ---------------------------------------------------------------------------


class TestMaomiErrorBackwardCompat:
    """Existing call sites with (message, filename, line, col) still work."""

    def test_basic_construction(self):
        err = MaomiError("bad thing", "test.mao", 1, 5)
        assert err.message == "bad thing"
        assert err.filename == "test.mao"
        assert err.line == 1
        assert err.col == 5
        assert err.col_end == 6  # default: col + 1
        assert err.hint is None
        assert err.secondary_labels == []
        assert err.severity == "error"

    def test_with_col_end(self):
        err = MaomiError("bad thing", "test.mao", 1, 5, col_end=10)
        assert err.col_end == 10
        assert err.hint is None
        assert err.secondary_labels == []
        assert err.severity == "error"

    def test_str_representation_unchanged(self):
        err = MaomiError("unexpected token", "foo.mao", 3, 7)
        assert str(err) == "foo.mao:3:7: unexpected token"

    def test_subclass_lexer_error(self):
        err = LexerError("invalid char", "test.mao", 1, 1)
        assert err.hint is None
        assert err.secondary_labels == []
        assert err.severity == "error"

    def test_subclass_parse_error(self):
        err = ParseError("expected }", "test.mao", 5, 1)
        assert err.hint is None
        assert err.severity == "error"

    def test_subclass_type_error(self):
        err = MaomiTypeError("type mismatch", "test.mao", 2, 3)
        assert err.hint is None
        assert err.severity == "error"


# ---------------------------------------------------------------------------
# MaomiError new fields
# ---------------------------------------------------------------------------


class TestMaomiErrorNewFields:
    def test_hint(self):
        err = MaomiError("bad types", "test.mao", 1, 1, hint="try cast()")
        assert err.hint == "try cast()"

    def test_secondary_labels_with_label_objects(self):
        lab = Label("defined here", "test.mao", 1, 1, 5)
        err = MaomiError("duplicate", "test.mao", 5, 1, secondary_labels=[lab])
        assert len(err.secondary_labels) == 1
        assert err.secondary_labels[0] is lab

    def test_secondary_labels_default_empty(self):
        err = MaomiError("msg", "f.mao", 1, 1)
        assert err.secondary_labels == []
        # Ensure it's a new list each time (not shared mutable default)
        err2 = MaomiError("msg2", "f.mao", 2, 1)
        assert err.secondary_labels is not err2.secondary_labels

    def test_severity_default_error(self):
        err = MaomiError("msg", "f.mao", 1, 1)
        assert err.severity == "error"

    def test_severity_warning(self):
        err = MaomiError("unused var", "f.mao", 1, 1, severity="warning")
        assert err.severity == "warning"

    def test_all_new_fields_together(self):
        lab = Label("first defined here", "a.mao", 1, 1, 3)
        err = MaomiError(
            "duplicate function 'foo'",
            "b.mao", 5, 1, col_end=4,
            hint="rename one of the definitions",
            secondary_labels=[lab],
            severity="error",
        )
        assert err.hint == "rename one of the definitions"
        assert err.secondary_labels == [lab]
        assert err.severity == "error"
        assert err.col_end == 4


# ---------------------------------------------------------------------------
# Label dataclass
# ---------------------------------------------------------------------------


class TestLabel:
    def test_fields(self):
        lab = Label("defined here", "test.mao", 10, 3, 8)
        assert lab.text == "defined here"
        assert lab.filename == "test.mao"
        assert lab.line == 10
        assert lab.col == 3
        assert lab.col_end == 8

    def test_equality(self):
        a = Label("x", "f.mao", 1, 1, 2)
        b = Label("x", "f.mao", 1, 1, 2)
        assert a == b

    def test_inequality(self):
        a = Label("x", "f.mao", 1, 1, 2)
        b = Label("y", "f.mao", 1, 1, 2)
        assert a != b


# ---------------------------------------------------------------------------
# Level enum
# ---------------------------------------------------------------------------


class TestLevel:
    def test_values(self):
        assert Level.ERROR.value == "error"
        assert Level.WARNING.value == "warning"
        assert Level.HINT.value == "hint"

    def test_members(self):
        assert set(Level) == {Level.ERROR, Level.WARNING, Level.HINT}


# ---------------------------------------------------------------------------
# from_error conversion
# ---------------------------------------------------------------------------


class TestFromError:
    def test_basic_conversion(self):
        err = MaomiError("undefined variable 'x'", "test.mao", 3, 5, col_end=6)
        diag = from_error(err)
        assert diag.title == "Undefined variable"
        assert diag.text == "undefined variable 'x'"
        assert diag.level == Level.ERROR
        assert diag.filename == "test.mao"
        assert diag.line == 3
        assert diag.col == 5
        assert diag.col_end == 6
        assert diag.hint is None
        assert diag.secondary_labels == []

    def test_warning_level(self):
        err = MaomiError("unused import", "test.mao", 1, 1, severity="warning")
        diag = from_error(err)
        assert diag.level == Level.WARNING

    def test_hint_preserved(self):
        err = MaomiError("type mismatch", "t.mao", 1, 1, hint="use cast()")
        diag = from_error(err)
        assert diag.hint == "use cast()"

    def test_secondary_labels_from_label_objects(self):
        lab = Label("defined here", "a.mao", 1, 1, 5)
        err = MaomiError("duplicate", "b.mao", 5, 1, secondary_labels=[lab])
        diag = from_error(err)
        assert len(diag.secondary_labels) == 1
        assert diag.secondary_labels[0] is lab

    def test_secondary_labels_from_tuples(self):
        err = MaomiError("duplicate", "b.mao", 5, 1,
                         secondary_labels=[("defined here", "a.mao", 1, 1, 5)])
        diag = from_error(err)
        assert len(diag.secondary_labels) == 1
        lab = diag.secondary_labels[0]
        assert isinstance(lab, Label)
        assert lab.text == "defined here"
        assert lab.filename == "a.mao"
        assert lab.line == 1
        assert lab.col == 1
        assert lab.col_end == 5

    def test_secondary_labels_from_lists(self):
        err = MaomiError("duplicate", "b.mao", 5, 1,
                         secondary_labels=[["defined here", "a.mao", 1, 1, 5]])
        diag = from_error(err)
        assert len(diag.secondary_labels) == 1
        assert isinstance(diag.secondary_labels[0], Label)

    def test_with_source_enrichment(self):
        err = MaomiError("expected f32, got i32", "t.mao", 1, 1)
        diag = from_error(err, source="let x: f32 = 42;")
        assert diag.hint is not None
        assert "cast" in diag.hint

    def test_without_source_no_enrichment(self):
        err = MaomiError("expected f32, got i32", "t.mao", 1, 1)
        diag = from_error(err)
        assert diag.hint is None


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------


class TestExtractTitle:
    def test_undefined_variable(self):
        assert _extract_title("undefined variable 'x'") == "Undefined variable"

    def test_undefined_function(self):
        assert _extract_title("undefined function 'foo'") == "Unknown function"

    def test_unknown_function(self):
        assert _extract_title("call to unknown function 'bar'") == "Unknown function"

    def test_type_mismatch(self):
        assert _extract_title("type mismatch: expected f32, got i32") == "Type mismatch"

    def test_mismatched_types(self):
        assert _extract_title("mismatched types in binary op") == "Type mismatch"

    def test_unknown_struct(self):
        assert _extract_title("unknown struct 'Foo'") == "Unknown struct"

    def test_no_struct(self):
        assert _extract_title("no struct named 'Bar'") == "Unknown struct"

    def test_argument_count(self):
        assert _extract_title("function 'f' expects 2 args, got 3") == "Incorrect argument count"

    def test_return_type(self):
        assert _extract_title("return type mismatch") == "Return type mismatch"

    def test_duplicate(self):
        assert _extract_title("duplicate function 'foo'") == "Duplicate definition"

    def test_shape_error(self):
        assert _extract_title("cannot add: shape [3,4] vs [5,6]") == "Shape error"

    def test_expected_syntax(self):
        assert _extract_title("expected ';' after expression") == "Syntax error"

    def test_import_error(self):
        assert _extract_title("failed to import module 'math'") == "Import error"

    def test_unknown_field(self):
        assert _extract_title("unknown field 'x' on struct S") == "Unknown field"

    def test_no_field(self):
        assert _extract_title("no field 'y' in struct T") == "Unknown field"

    def test_colon_fallback(self):
        assert _extract_title("some issue: details here") == "Some issue"

    def test_long_message_fallback(self):
        long_msg = "a" * 80
        assert _extract_title(long_msg) == "Error"

    def test_short_message_fallback(self):
        assert _extract_title("oops") == "oops"


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------


class TestEnrich:
    def _make_diag(self, text, hint=None):
        return Diagnostic(
            title="Test",
            text=text,
            level=Level.ERROR,
            filename="test.mao",
            line=1,
            col=1,
            col_end=2,
            hint=hint,
        )

    def test_shape_rank_mismatch_hint(self):
        diag = self._make_diag("cannot add f32[3,4] and f32[3]")
        result = enrich(diag, "let x = a + b;")
        assert result.hint is not None
        assert "different ranks" in result.hint
        assert "Broadcasting" in result.hint

    def test_shape_same_rank_no_hint(self):
        diag = self._make_diag("cannot add f32[3,4] and f32[5,6]")
        result = enrich(diag, "let x = a + b;")
        # Same rank (2 vs 2), so the rank-mismatch hint should not fire
        assert result.hint is None or "different ranks" not in result.hint

    def test_scalar_array_hint(self):
        diag = self._make_diag("cannot multiply f32 by f32[3,4]")
        result = enrich(diag, "let x = a * b;")
        assert result.hint is not None
        assert "broadcast" in result.hint.lower()

    def test_type_mismatch_cast_hint(self):
        diag = self._make_diag("expected f32, got i32")
        result = enrich(diag, "let x: f32 = y;")
        assert result.hint is not None
        assert "cast" in result.hint
        assert "f32" in result.hint
        assert "i32" in result.hint

    def test_type_mismatch_same_type_no_hint(self):
        diag = self._make_diag("expected f32, got f32")
        result = enrich(diag, "let x = y;")
        # Same types: cast hint should not fire
        assert result.hint is None

    def test_no_enrichment_for_unrelated_message(self):
        diag = self._make_diag("undefined variable 'x'")
        result = enrich(diag, "let y = x;")
        assert result.hint is None

    def test_existing_hint_not_overwritten(self):
        diag = self._make_diag(
            "cannot add f32[3,4] and f32[3]",
            hint="user-provided hint",
        )
        result = enrich(diag, "let x = a + b;")
        assert result.hint == "user-provided hint"

    def test_existing_hint_blocks_scalar_array(self):
        diag = self._make_diag(
            "cannot multiply f32 by f32[3,4]",
            hint="already set",
        )
        result = enrich(diag, "let x = a * b;")
        assert result.hint == "already set"

    def test_existing_hint_blocks_cast(self):
        diag = self._make_diag(
            "expected f32, got i32",
            hint="already set",
        )
        result = enrich(diag, "let x = y;")
        assert result.hint == "already set"


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------


class TestDiagnostic:
    def test_defaults(self):
        diag = Diagnostic(
            title="Test",
            text="msg",
            level=Level.ERROR,
            filename="f.mao",
            line=1,
            col=1,
            col_end=2,
        )
        assert diag.secondary_labels == []
        assert diag.hint is None

    def test_with_all_fields(self):
        lab = Label("here", "a.mao", 1, 1, 3)
        diag = Diagnostic(
            title="Dup",
            text="duplicate def",
            level=Level.WARNING,
            filename="b.mao",
            line=5,
            col=1,
            col_end=4,
            secondary_labels=[lab],
            hint="rename it",
        )
        assert diag.title == "Dup"
        assert diag.level == Level.WARNING
        assert diag.secondary_labels == [lab]
        assert diag.hint == "rename it"
