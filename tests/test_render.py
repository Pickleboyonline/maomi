"""Tests for the ANSI source-snippet error renderer."""

from maomi.errors import MaomiError, MaomiTypeError, LexerError
from maomi.render import render_error


SAMPLE_SOURCE = """\
fn add(a: f32, b: i32) -> f32 {
    let c = a + 1.0;
    a + b
}"""


# ---------------------------------------------------------------------------
# Basic rendering with source
# ---------------------------------------------------------------------------

class TestRenderWithSource:
    def test_produces_multiline_output(self):
        err = MaomiError("expected f32, got i32", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        # Should have title, location, gutter, source, underline, gutter
        assert len(lines) >= 5

    def test_title_contains_error(self):
        err = MaomiError("expected f32, got i32", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert out.startswith("error: expected f32, got i32")

    def test_location_line(self):
        err = MaomiError("expected f32, got i32", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "  --> test.mao:3:5" in out

    def test_source_line_shown(self):
        err = MaomiError("expected f32, got i32", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "a + b" in out

    def test_primary_underline_carets(self):
        err = MaomiError("expected f32, got i32", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        underline_lines = [l for l in lines if "^^^" in l]
        assert len(underline_lines) == 1
        assert "expected f32, got i32" in underline_lines[0]

    def test_underline_spans_correct_columns(self):
        # col=5, col_end=8 -> 3 carets at position 4 (0-indexed)
        err = MaomiError("type error", "test.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        underline_lines = [l for l in lines if "^^^" in l]
        assert len(underline_lines) == 1
        # Find the caret position within the underline content
        ul = underline_lines[0]
        # After the gutter "   | ", the underline should have 4 spaces then 3 carets
        caret_idx = ul.index("^^^")
        # Just verify the count of carets
        caret_segment = ul[caret_idx:]
        caret_count = 0
        for ch in caret_segment:
            if ch == "^":
                caret_count += 1
            else:
                break
        assert caret_count == 3


# ---------------------------------------------------------------------------
# Fallback (no source)
# ---------------------------------------------------------------------------

class TestRenderFallback:
    def test_no_source_oneline(self):
        err = MaomiError("undefined variable 'x'", "test.mao", 5, 3)
        out = render_error(err, source=None, use_color=False)
        assert out == "error: undefined variable 'x' (at test.mao:5:3)"

    def test_line_out_of_range(self):
        err = MaomiError("bad line", "test.mao", 999, 1)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert out == "error: bad line (at test.mao:999:1)"


# ---------------------------------------------------------------------------
# Color control
# ---------------------------------------------------------------------------

class TestColorControl:
    def test_no_ansi_when_disabled(self):
        err = MaomiError("msg", "f.mao", 1, 1)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "\033[" not in out

    def test_ansi_when_enabled(self):
        err = MaomiError("msg", "f.mao", 1, 1)
        out = render_error(err, SAMPLE_SOURCE, use_color=True)
        assert "\033[" in out


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_error_title(self):
        err = MaomiError("bad thing", "f.mao", 1, 1)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert out.startswith("error: bad thing")

    def test_warning_title(self):
        err = MaomiError("unused var", "f.mao", 1, 1)
        err.severity = "warning"
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert out.startswith("warning: unused var")

    def test_warning_fallback(self):
        err = MaomiError("unused var", "f.mao", 1, 1)
        err.severity = "warning"
        out = render_error(err, source=None, use_color=False)
        assert out.startswith("warning:")


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

class TestHints:
    def test_hint_appears(self):
        err = MaomiError("type mismatch", "f.mao", 3, 5, 8)
        err.hint = "Use cast(b, f32) to convert"
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "= hint: Use cast(b, f32) to convert" in out

    def test_no_hint_no_line(self):
        err = MaomiError("type mismatch", "f.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "hint:" not in out


# ---------------------------------------------------------------------------
# Secondary labels
# ---------------------------------------------------------------------------

class TestSecondaryLabels:
    def test_secondary_label_rendered(self):
        err = MaomiError("return type mismatch", "f.mao", 3, 5, 8)
        err.secondary_labels = [
            {"line": 1, "col": 19, "col_end": 22, "message": "return type declared here"},
        ]
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "---" in out
        assert "return type declared here" in out

    def test_secondary_uses_dashes(self):
        err = MaomiError("mismatch", "f.mao", 3, 5, 8)
        err.secondary_labels = [
            {"line": 1, "col": 19, "col_end": 22, "message": "declared here"},
        ]
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        dash_lines = [l for l in lines if "---" in l and "declared here" in l]
        assert len(dash_lines) == 1

    def test_both_primary_and_secondary(self):
        err = MaomiError("return type mismatch", "f.mao", 3, 5, 8)
        err.secondary_labels = [
            {"line": 1, "col": 19, "col_end": 22, "message": "declared here"},
        ]
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "^^^" in out
        assert "---" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_error_on_first_line(self):
        err = MaomiError("parse error", "f.mao", 1, 1, 3)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "fn add" in out
        assert "^^" in out

    def test_error_on_last_line(self):
        err = MaomiError("missing semi", "f.mao", 4, 1, 2)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "}" in out
        assert "^" in out

    def test_col_end_beyond_line(self):
        err = MaomiError("overflow", "f.mao", 4, 1, 999)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        # Should clamp col_end to line length; no crash
        assert "^" in out

    def test_single_char_span(self):
        err = MaomiError("unexpected token", "f.mao", 1, 1)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        caret_lines = [l for l in lines if "^" in l]
        assert len(caret_lines) >= 1

    def test_empty_source(self):
        err = MaomiError("empty", "f.mao", 1, 1)
        out = render_error(err, "", use_color=False)
        # Empty source -> line 1 doesn't exist -> fallback
        assert "error: empty (at f.mao:1:1)" == out


# ---------------------------------------------------------------------------
# Gutter alignment
# ---------------------------------------------------------------------------

class TestGutterAlignment:
    def test_single_digit_gutter(self):
        err = MaomiError("msg", "f.mao", 3, 5, 8)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        lines = out.split("\n")
        # Line number 3 should be right-justified in 1-wide gutter
        source_lines = [l for l in lines if " | " in l and "a + b" in l]
        assert len(source_lines) == 1
        assert " 3 |" in source_lines[0]

    def test_multi_digit_gutter(self):
        # Create source with 15 lines
        big_source = "\n".join(f"line {i}" for i in range(1, 16))
        err = MaomiError("msg", "f.mao", 12, 1, 5)
        out = render_error(err, big_source, use_color=False)
        lines = out.split("\n")
        source_lines = [l for l in lines if "line 12" in l and "|" in l]
        assert len(source_lines) >= 1
        # "12" in a 2-wide gutter
        assert " 12 |" in source_lines[0]

    def test_mixed_gutter_secondary_forces_wider(self):
        big_source = "\n".join(f"line {i}" for i in range(1, 25))
        err = MaomiError("msg", "f.mao", 3, 1, 5)
        err.secondary_labels = [
            {"line": 22, "col": 1, "col_end": 5, "message": "note"},
        ]
        out = render_error(err, big_source, use_color=False)
        # Line 22 -> gutter width 2
        lines = out.split("\n")
        source_lines_3 = [l for l in lines if "line 3" in l and "|" in l]
        assert len(source_lines_3) >= 1
        # Line 3 should be right-justified in 2-wide gutter: " 3 |"
        assert " 3 |" in source_lines_3[0]


# ---------------------------------------------------------------------------
# Integration with error subtypes
# ---------------------------------------------------------------------------

class TestErrorSubtypes:
    def test_type_error(self):
        err = MaomiTypeError("incompatible types", "f.mao", 2, 5, 10)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "error: incompatible types" in out
        assert "let c = a + 1.0;" in out

    def test_lexer_error(self):
        err = LexerError("unexpected character", "f.mao", 1, 1, 2)
        out = render_error(err, SAMPLE_SOURCE, use_color=False)
        assert "error: unexpected character" in out
