"""Tests for brace counting that ignores braces in comments and strings."""

from maomi.lsp._formatting import (
    _compute_brace_depth,
    _effective_braces,
    _format_document,
)


class TestEffectiveBraces:
    def test_no_comment_no_string(self):
        assert _effective_braces("fn f() {") == "fn f() {"

    def test_comment_strips_trailing(self):
        assert _effective_braces("x = 1; // { brace in comment }") == "x = 1; "

    def test_string_neutralizes_braces(self):
        result = _effective_braces('let s = "{}";')
        assert "{" not in result
        assert "}" not in result

    def test_mixed_real_and_comment_braces(self):
        result = _effective_braces("fn f() { // { extra }")
        assert result.count("{") == 1
        assert result.count("}") == 0

    def test_empty_line(self):
        assert _effective_braces("") == ""

    def test_only_comment(self):
        assert _effective_braces("// { } { }") == ""


class TestComputeBraceDepthWithComments:
    def test_comment_braces_ignored(self):
        lines = [
            "fn f() {",
            "    // { this has braces }",
            "    let x = 1;",
        ]
        # Only the real { on line 0 counts; comment braces on line 1 are ignored
        assert _compute_brace_depth(lines, 2) == 1

    def test_string_braces_ignored(self):
        lines = [
            'let s = "{}";',
            "let x = 1;",
        ]
        assert _compute_brace_depth(lines, 1) == 0

    def test_normal_braces_still_counted(self):
        lines = [
            "fn f() {",
            "    if true {",
            "        x;",
        ]
        assert _compute_brace_depth(lines, 2) == 2

    def test_mixed_real_and_comment_braces(self):
        lines = [
            "fn f() { // opening {",
            "    let x = 1;",
            "} // closing }",
        ]
        # Line 0: 1 real open brace (comment { ignored)
        # Line 1: 0 braces
        # Target is line 2 so we count lines 0 and 1
        assert _compute_brace_depth(lines, 2) == 1


class TestFormatDocumentWithComments:
    def test_comment_braces_dont_affect_indent(self):
        source = "fn f() {\n    // { extra brace\n    let x = 1;\n}\n"
        edits = _format_document(source)
        # Already correctly formatted, so no edits needed
        assert edits == []

    def test_comment_braces_would_have_broken_indent(self):
        # If comment braces were counted, line 3 would be double-indented
        source = "fn f() {\n// { extra\nlet x = 1;\n}\n"
        edits = _format_document(source)
        assert len(edits) == 1
        result = edits[0].new_text
        result_lines = result.splitlines()
        # "// { extra" should be at depth 1 (inside fn body)
        assert result_lines[1] == "    // { extra"
        # "let x = 1;" should also be at depth 1 (not 2)
        assert result_lines[2] == "    let x = 1;"

    def test_string_braces_dont_affect_indent(self):
        source = 'fn f() {\nlet s = "{}";\nlet x = 1;\n}\n'
        edits = _format_document(source)
        assert len(edits) == 1
        result = edits[0].new_text
        result_lines = result.splitlines()
        assert result_lines[1] == '    let s = "{}";'
        assert result_lines[2] == "    let x = 1;"
