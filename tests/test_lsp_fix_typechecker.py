"""Tests for type checker fixes: empty body, duplicate names, wide squiggles."""

from maomi.lsp import validate


class TestEmptyFunctionBody:
    """B16: Empty function body with a declared return type should produce a diagnostic."""

    def test_empty_body_with_return_type(self):
        source = "fn f(x: f32) -> f32 { }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        msgs = [d.message for d in diags]
        assert any("body is empty" in m for m in msgs), f"Expected 'body is empty' error, got: {msgs}"

    def test_nonempty_body_no_diagnostic(self):
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert diags == []


class TestDuplicateNames:
    """B19: Duplicate function names, param names, and struct fields should be reported."""

    def test_duplicate_function_names(self):
        source = "fn f(x: f32) -> f32 { x }\nfn f(y: f32) -> f32 { y }"
        diags, result = validate(source, "<test>")
        msgs = [d.message for d in diags]
        assert any("duplicate function name" in m and "'f'" in m for m in msgs), \
            f"Expected duplicate function name error, got: {msgs}"

    def test_duplicate_param_names(self):
        source = "fn f(x: f32, x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        msgs = [d.message for d in diags]
        assert any("duplicate parameter name" in m and "'x'" in m for m in msgs), \
            f"Expected duplicate parameter name error, got: {msgs}"

    def test_duplicate_struct_fields(self):
        source = "struct P { x: f32, x: f32 }\nfn f(p: P) -> f32 { p.x }"
        diags, result = validate(source, "<test>")
        msgs = [d.message for d in diags]
        assert any("duplicate field" in m and "'x'" in m for m in msgs), \
            f"Expected duplicate struct field error, got: {msgs}"

    def test_no_duplicates_no_diagnostic(self):
        source = "fn f(x: f32, y: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        assert diags == []


class TestWideSquiggles:
    """G20: Type error diagnostics should span more than 1 character for identifiers."""

    def test_undefined_variable_squiggle_width(self):
        source = "fn f(x: f32) -> f32 { foobar }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        # Find the undefined variable diagnostic
        undef_diag = None
        for d in diags:
            if "undefined variable" in d.message and "foobar" in d.message:
                undef_diag = d
                break
        assert undef_diag is not None, f"Expected undefined variable error, got: {[d.message for d in diags]}"
        width = undef_diag.range.end.character - undef_diag.range.start.character
        assert width > 1, f"Expected squiggle wider than 1 char, got width {width}"
        # 'foobar' is 6 chars
        assert width == 6, f"Expected width 6 for 'foobar', got {width}"

    def test_undefined_function_squiggle_width(self):
        source = "fn f(x: f32) -> f32 { unknown_func(x) }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        undef_diag = None
        for d in diags:
            if "undefined function" in d.message and "unknown_func" in d.message:
                undef_diag = d
                break
        assert undef_diag is not None, f"Expected undefined function error, got: {[d.message for d in diags]}"
        width = undef_diag.range.end.character - undef_diag.range.start.character
        assert width > 1, f"Expected squiggle wider than 1 char, got width {width}"

    def test_wrong_arg_count_squiggle_width(self):
        source = "fn g(x: f32) -> f32 { x }\nfn f(x: f32) -> f32 { g(x, x) }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        arg_diag = None
        for d in diags:
            if "expects" in d.message and "arguments" in d.message:
                arg_diag = d
                break
        assert arg_diag is not None, f"Expected wrong arg count error, got: {[d.message for d in diags]}"
        width = arg_diag.range.end.character - arg_diag.range.start.character
        assert width > 1, f"Expected squiggle wider than 1 char, got width {width}"

    def test_type_mismatch_squiggle_width(self):
        # Use multi-character identifier 'val' so squiggle should be wider than 1
        source = "fn g(x: f32) -> f32 { x }\nfn f(val: i32) -> f32 { g(val) }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        mismatch_diag = None
        for d in diags:
            if "argument" in d.message and ("expected" in d.message or "mismatch" in d.message):
                mismatch_diag = d
                break
        assert mismatch_diag is not None, f"Expected type mismatch error, got: {[d.message for d in diags]}"
        width = mismatch_diag.range.end.character - mismatch_diag.range.start.character
        assert width > 1, f"Expected squiggle wider than 1 char, got width {width}"
        # 'val' is 3 chars
        assert width == 3, f"Expected width 3 for 'val', got {width}"
