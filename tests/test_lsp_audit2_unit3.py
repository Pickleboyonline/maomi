"""Tests for LSP audit round 2, Unit 3: Signature Help + Inlay Hints fixes."""

import inspect

import pytest
from lsprotocol import types

from maomi.lsp import validate, _build_inlay_hints
from maomi.lsp._signature import _sig_parse_call_context
from maomi.lsp._inlay_hints import _inlay_collect_from_expr


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hints(source, start_line_1=1, end_line_1=100):
    _, result = validate(source, "<test>")
    return _build_inlay_hints(result, start_line_1, end_line_1, source)


# ---------------------------------------------------------------------------
# B14: Signature help works when cursor is inside a string argument
# ---------------------------------------------------------------------------

class TestSigHelpInsideString:
    def test_cursor_inside_string_arg(self):
        """Cursor inside a string arg like config("ke|y") should still find the call."""
        source = 'fn f(x: f32) -> f32 { x }\nfn g() -> f32 { f(1.0) }'
        # Simulate: config("ke|y")  -- but use a simpler example
        # The key test: backward scan from inside a string should still find function
        line_text = 'config("key")'
        source_with_call = line_text
        pos = types.Position(line=0, character=10)  # inside "key", on 'e'
        name, count, named = _sig_parse_call_context(source_with_call, pos)
        assert name == "config", f"Expected 'config', got {name!r}"
        assert count == 0

    def test_cursor_at_start_of_string(self):
        """Cursor right after opening quote."""
        source = 'foo("bar", 42)'
        pos = types.Position(line=0, character=5)  # on 'b' inside string
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "foo"
        assert count == 0

    def test_cursor_inside_second_string_arg(self):
        """Cursor inside second string argument."""
        source = 'foo(1, "hello")'
        pos = types.Position(line=0, character=10)  # inside "hello"
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "foo"
        assert count == 1


# ---------------------------------------------------------------------------
# E1: Named arg detected when cursor is on name text or '=' sign
# ---------------------------------------------------------------------------

class TestNamedArgDetection:
    def test_named_arg_on_value(self):
        """Baseline: cursor on value after '=' works."""
        source = 'f(x, axis=1)'
        pos = types.Position(line=0, character=11)  # on '1'
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "f"
        assert named == "axis"

    def test_named_arg_on_name_text(self):
        """Cursor on the name text 'axis' before '=' should detect named arg."""
        source = 'f(x, axis=1)'
        pos = types.Position(line=0, character=6)  # on 'x' in 'axis'
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "f"
        assert named == "axis"

    def test_named_arg_on_equals(self):
        """Cursor on '=' sign should detect named arg."""
        source = 'f(x, axis=1)'
        pos = types.Position(line=0, character=9)  # on '='
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "f"
        assert named == "axis"

    def test_named_arg_first_param(self):
        """Named arg as first parameter."""
        source = 'f(axis=1)'
        pos = types.Position(line=0, character=3)  # on 'x' in 'axis'
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "f"
        assert named == "axis"


# ---------------------------------------------------------------------------
# E2: Cursor on opening paren returns function name
# ---------------------------------------------------------------------------

class TestCursorOnOpenParen:
    def test_cursor_on_opening_paren(self):
        """Cursor positioned right at '(' should still find the function name."""
        source = 'foo(1, 2)'
        # col=3 is the '(' character
        pos = types.Position(line=0, character=3)
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "foo", f"Expected 'foo', got {name!r}"
        assert count == 0

    def test_cursor_on_paren_no_args(self):
        """Cursor at '(' of a no-arg call."""
        source = 'bar()'
        pos = types.Position(line=0, character=3)
        name, count, named = _sig_parse_call_context(source, pos)
        assert name == "bar"


# ---------------------------------------------------------------------------
# G9: No duplicate WhileExpr/FoldExpr branches in inlay hints
# ---------------------------------------------------------------------------

class TestNoDuplicateBranches:
    def test_no_duplicate_while_branch(self):
        """_inlay_collect_from_expr should not have duplicate WhileExpr branches."""
        source = inspect.getsource(_inlay_collect_from_expr)
        count = source.count("WhileExpr")
        assert count == 1, f"Expected 1 WhileExpr check, found {count}"

    def test_no_duplicate_fold_branch(self):
        """_inlay_collect_from_expr should not have duplicate FoldExpr branches."""
        source = inspect.getsource(_inlay_collect_from_expr)
        count = source.count("FoldExpr")
        assert count == 1, f"Expected 1 FoldExpr check, found {count}"


# ---------------------------------------------------------------------------
# E3: Destructured let hint shows struct name, not full expansion
# ---------------------------------------------------------------------------

class TestStructTypeHintDisplay:
    def test_struct_let_shows_name_only(self):
        """let p = Point { x: 1.0, y: 2.0 } should show ': Point', not full expansion."""
        source = """\
struct Point { x: f32, y: f32 }

fn f() -> f32 {
    let p = Point { x: 1.0, y: 2.0 };
    p.x
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        assert len(type_hints) >= 1
        # The type hint for 'p' should be ': Point', not ': Point { x: f32, y: f32 }'
        p_hint = type_hints[0]
        assert p_hint.label == ": Point", f"Expected ': Point', got {p_hint.label!r}"

    def test_non_struct_type_unchanged(self):
        """Non-struct types like f32 should display normally."""
        source = """\
fn f(x: f32) -> f32 {
    let y = x + 1.0;
    y
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        assert len(type_hints) >= 1
        assert type_hints[0].label == ": f32"
