"""Tests for inlay hint fixes: nested LetStmt recursion and parameter name hints."""

import pytest
from lsprotocol import types

from maomi.lsp import validate, _build_inlay_hints


def _hints(source, start_line_1=1, end_line_1=100):
    """Helper: validate source and return inlay hints for given 1-indexed line range."""
    _, result = validate(source, "<test>")
    return _build_inlay_hints(result, start_line_1, end_line_1, source)


# ---------------------------------------------------------------------------
# B3: Nested let inside block expressions gets type hints
# ---------------------------------------------------------------------------

class TestNestedLetRecursion:
    def test_let_inside_if_then_branch(self):
        source = """\
fn f(x: f32) -> f32 {
    let result = if x > 0.0 {
        let pos = x + 1.0;
        pos
    } else {
        x
    };
    result
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        labels = [h.label for h in type_hints]
        # 'result' and 'pos' should both get type hints
        assert ": f32" in labels[0]
        assert ": f32" in labels[1]
        assert len(type_hints) == 2

    def test_let_inside_if_else_branch(self):
        source = """\
fn f(x: f32) -> f32 {
    let result = if x > 0.0 {
        x
    } else {
        let neg = 0.0 - x;
        neg
    };
    result
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        labels = [h.label for h in type_hints]
        # 'result' and 'neg' should both get type hints
        assert len(type_hints) == 2
        assert all(": f32" in lbl for lbl in labels)

    def test_let_inside_scan_body(self):
        source = """\
fn f(xs: f32[5]) -> f32 {
    let result = scan (carry, elem) in (0.0, xs) {
        let step = carry + elem;
        step
    };
    result[4]
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        # 'result' and 'step' should both get type hints
        assert len(type_hints) == 2

    def test_let_inside_map_body(self):
        source = """\
fn f(xs: f32[5]) -> f32[5] {
    let result = map elem in xs {
        let doubled = elem + elem;
        doubled
    };
    result
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        # 'result' and 'doubled' should both get type hints
        assert len(type_hints) == 2

    def test_deeply_nested_let(self):
        """Let inside if inside let value."""
        source = """\
fn f(x: f32) -> f32 {
    let outer = if x > 0.0 {
        let mid = if x > 1.0 {
            let inner = x + 2.0;
            inner
        } else {
            x
        };
        mid
    } else {
        x
    };
    outer
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        # 'outer', 'mid', 'inner' should all get type hints
        assert len(type_hints) == 3


# ---------------------------------------------------------------------------
# G17: Parameter name hints at call sites
# ---------------------------------------------------------------------------

class TestParameterHints:
    def test_user_function_call_gets_param_hints(self):
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(x: f32, y: f32) -> f32 { add(x, y) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        # 'x' matches param 'a' (different name) => hint
        # 'y' matches param 'b' (different name) => hint
        assert len(param_hints) == 2
        labels = [h.label for h in param_hints]
        assert "a:" in labels
        assert "b:" in labels

    def test_param_hint_position_before_arg(self):
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(x: f32, y: f32) -> f32 { add(x, y) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints) == 2
        # All parameter hints should have padding_right=True
        for h in param_hints:
            assert h.padding_right is True

    def test_param_hint_skipped_when_name_matches(self):
        """If the argument name matches the param name, no hint."""
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(a: f32, b: f32) -> f32 { add(a, b) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        # Both args match param names => no hints
        assert len(param_hints) == 0

    def test_param_hint_partial_match(self):
        """Only matching names are skipped, others get hints."""
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(a: f32, y: f32) -> f32 { add(a, y) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        # 'a' matches => no hint; 'y' doesn't match 'b' => hint
        assert len(param_hints) == 1
        assert param_hints[0].label == "b:"

    def test_single_param_builtin_no_hint(self):
        """Single-parameter builtins (like exp, sqrt) should not get hints."""
        source = """\
fn f(x: f32) -> f32 { exp(x) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints) == 0

    def test_multi_param_builtin_gets_hints(self):
        """Multi-parameter builtins should get parameter hints."""
        source = """\
fn f(a: f32[3], b: f32[3]) -> f32[3] { where(a > b, a, b) }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        # where(cond, x, y) has 3 params; args are (a>b, a, b) which don't match
        # param names (cond, x, y), so all 3 get hints
        assert len(param_hints) == 3
        labels = [h.label for h in param_hints]
        assert "cond:" in labels
        assert "x:" in labels
        assert "y:" in labels

    def test_hints_within_line_range(self):
        """Only hints within the requested line range are returned."""
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(x: f32, y: f32) -> f32 {
    add(x, y)
}"""
        # Only request line 3 (1-indexed) — where the call is
        hints_in_range = _hints(source, start_line_1=3, end_line_1=3)
        param_hints = [h for h in hints_in_range if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints) == 2

        # Request line 1 only — no call there
        hints_out = _hints(source, start_line_1=1, end_line_1=1)
        param_hints_out = [h for h in hints_out if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints_out) == 0

    def test_no_param_hints_for_no_args(self):
        """A call with no arguments should produce no parameter hints."""
        source = """\
fn zero() -> f32 { 0.0 }
fn main() -> f32 { zero() }"""
        hints = _hints(source)
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        assert len(param_hints) == 0

    def test_param_hints_coexist_with_type_hints(self):
        """Both type hints and parameter hints should be returned."""
        source = """\
fn add(a: f32, b: f32) -> f32 { a + b }
fn main(x: f32, y: f32) -> f32 {
    let result = add(x, y);
    result
}"""
        hints = _hints(source)
        type_hints = [h for h in hints if h.kind == types.InlayHintKind.Type]
        param_hints = [h for h in hints if h.kind == types.InlayHintKind.Parameter]
        assert len(type_hints) == 1  # 'result'
        assert len(param_hints) == 2  # 'a:' and 'b:'
