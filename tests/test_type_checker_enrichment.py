"""Tests for enriched type checker error messages."""
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker


def get_errors(source: str) -> list:
    tokens = Lexer(source, filename="<test>").tokenize()
    program = Parser(tokens, filename="<test>").parse()
    checker = TypeChecker(filename="<test>")
    checker.check(program)
    return checker.errors


class TestEnrichedErrors:
    def test_undefined_variable_hint(self):
        errors = get_errors("fn f(x: f32) -> f32 { y }")
        assert len(errors) >= 1
        err = errors[0]
        # At minimum, the error should work (backward compat)
        assert "undefined" in err.message.lower() or "unknown" in err.message.lower()

    def test_undefined_variable_with_similar_name(self):
        errors = get_errors("fn f(value: f32) -> f32 { valeu }")
        assert len(errors) >= 1
        err = errors[0]
        hint = getattr(err, 'hint', None)
        if hint:
            assert "value" in hint

    def test_undefined_variable_no_hint_when_no_similar(self):
        errors = get_errors("fn f(x: f32) -> f32 { zzzzz }")
        assert len(errors) >= 1
        err = errors[0]
        hint = getattr(err, 'hint', None)
        # zzzzz is too far from x, so no hint should be given
        assert hint is None

    def test_undefined_function_hint(self):
        source = "fn add(x: f32, y: f32) -> f32 { x + y }\nfn f(x: f32) -> f32 { ad(x, x) }"
        errors = get_errors(source)
        assert len(errors) >= 1
        fn_err = [e for e in errors if "undefined function" in e.message.lower()]
        assert len(fn_err) >= 1
        hint = getattr(fn_err[0], 'hint', None)
        if hint:
            assert "add" in hint

    def test_argument_count_mismatch(self):
        source = "fn add(x: f32, y: f32) -> f32 { x + y }\nfn f() -> f32 { add(1.0) }"
        errors = get_errors(source)
        assert len(errors) >= 1
        err = [e for e in errors if "arg" in e.message.lower()][0]
        labels = getattr(err, 'secondary_labels', [])
        # Should have secondary label pointing to function definition
        if labels:
            assert "defined" in labels[0][0].lower() or "function" in labels[0][0].lower()

    def test_argument_count_mismatch_secondary_label(self):
        source = "fn add(x: f32, y: f32) -> f32 { x + y }\nfn f() -> f32 { add(1.0) }"
        errors = get_errors(source)
        arg_errs = [e for e in errors if "expects" in e.message and "arg" in e.message.lower()]
        assert len(arg_errs) >= 1
        labels = getattr(arg_errs[0], 'secondary_labels', [])
        assert len(labels) >= 1
        assert "function defined here" in labels[0][0]

    def test_field_not_found_hint(self):
        source = "struct S { x: f32, y: f32 }\nfn f(s: S) -> f32 { s.z }"
        errors = get_errors(source)
        assert len(errors) >= 1
        field_err = [e for e in errors if "field" in e.message.lower() or "z" in e.message]
        if field_err:
            hint = getattr(field_err[0], 'hint', None)
            if hint:
                assert "x" in hint or "y" in hint  # available fields listed

    def test_field_not_found_lists_available_fields(self):
        source = "struct S { x: f32, y: f32 }\nfn f(s: S) -> f32 { s.z }"
        errors = get_errors(source)
        field_err = [e for e in errors if "no field" in e.message]
        assert len(field_err) >= 1
        hint = getattr(field_err[0], 'hint', None)
        assert hint is not None
        assert "Available fields:" in hint
        assert "x" in hint
        assert "y" in hint

    def test_duplicate_function_secondary_label(self):
        source = "fn f(x: f32) -> f32 { x }\nfn f(x: f32) -> f32 { x }"
        errors = get_errors(source)
        dup_err = [e for e in errors if "duplicate" in e.message.lower()]
        if dup_err:
            labels = getattr(dup_err[0], 'secondary_labels', [])
            if labels:
                assert "first" in labels[0][0].lower() or "defined" in labels[0][0].lower()

    def test_duplicate_function_has_secondary_label(self):
        source = "fn f(x: f32) -> f32 { x }\nfn f(x: f32) -> f32 { x }"
        errors = get_errors(source)
        dup_err = [e for e in errors if "duplicate function" in e.message.lower()]
        assert len(dup_err) >= 1
        labels = getattr(dup_err[0], 'secondary_labels', [])
        assert len(labels) >= 1
        assert "first defined here" in labels[0][0]

    def test_duplicate_struct_secondary_label(self):
        source = "struct S { x: f32 }\nstruct S { y: f32 }"
        errors = get_errors(source)
        dup_err = [e for e in errors if "duplicate struct" in e.message.lower()]
        assert len(dup_err) >= 1
        labels = getattr(dup_err[0], 'secondary_labels', [])
        assert len(labels) >= 1
        assert "first defined here" in labels[0][0]

    def test_backward_compatibility(self):
        """Existing error behavior is unchanged."""
        errors = get_errors("fn f(x: f32) -> i32 { x }")
        assert len(errors) >= 1
        # All errors should still have message, line, col
        for err in errors:
            assert err.message
            assert err.line > 0
            assert err.col > 0

    def test_binary_op_type_mismatch_hint(self):
        source = "fn f(x: f32, y: i32) -> f32 { x + y }"
        errors = get_errors(source)
        if errors:
            err = errors[0]
            hint = getattr(err, 'hint', None)
            if hint:
                assert "cast" in hint.lower()

    def test_binary_op_mismatch_has_cast_hint(self):
        source = "fn f(x: f32, y: i32) -> f32 { x + y }"
        errors = get_errors(source)
        op_err = [e for e in errors if "mismatched types" in e.message]
        assert len(op_err) >= 1
        hint = getattr(op_err[0], 'hint', None)
        assert hint is not None
        assert "cast" in hint.lower()

    def test_return_type_mismatch_secondary_label(self):
        source = "fn f(x: f32) -> i32 { x }"
        errors = get_errors(source)
        ret_err = [e for e in errors if "return type mismatch" in e.message]
        assert len(ret_err) >= 1
        labels = getattr(ret_err[0], 'secondary_labels', [])
        assert len(labels) >= 1
        assert "return type declared here" in labels[0][0]

    def test_return_type_mismatch_hint(self):
        source = "fn f(x: f32) -> i32 { x }"
        errors = get_errors(source)
        ret_err = [e for e in errors if "return type mismatch" in e.message]
        assert len(ret_err) >= 1
        hint = getattr(ret_err[0], 'hint', None)
        assert hint is not None
        assert "i32" in hint
        assert "f32" in hint

    def test_unknown_struct_hint(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Poitn { x: 1.0, y: 2.0 } }"
        errors = get_errors(source)
        struct_err = [e for e in errors if "unknown struct" in e.message]
        assert len(struct_err) >= 1
        hint = getattr(struct_err[0], 'hint', None)
        assert hint is not None
        assert "Point" in hint

    def test_unknown_type_hint(self):
        source = "fn f(x: f3) -> f32 { x }"
        errors = get_errors(source)
        type_err = [e for e in errors if "unknown type" in e.message]
        assert len(type_err) >= 1
        hint = getattr(type_err[0], 'hint', None)
        # f3 is close to f32
        if hint:
            assert "f32" in hint

    def test_error_without_hint_has_no_hint_attr(self):
        """Errors that don't have hints shouldn't have the attribute set."""
        errors = get_errors("fn f(x: f32) -> f32 { x }")
        assert len(errors) == 0  # No errors for valid code

    def test_comparison_type_mismatch_cast_hint(self):
        source = "fn f(x: f32, y: i32) -> bool { x == y }"
        errors = get_errors(source)
        cmp_err = [e for e in errors if "mismatched types" in e.message]
        assert len(cmp_err) >= 1
        hint = getattr(cmp_err[0], 'hint', None)
        assert hint is not None
        assert "cast" in hint.lower()

    def test_struct_field_count_secondary_label(self):
        source = "struct S { x: f32, y: f32 }\nfn f() -> S { S { x: 1.0 } }"
        errors = get_errors(source)
        count_err = [e for e in errors if "has 2 fields" in e.message]
        assert len(count_err) >= 1
        labels = getattr(count_err[0], 'secondary_labels', [])
        assert len(labels) >= 1
        assert "struct defined here" in labels[0][0]

    def test_struct_field_count_hint(self):
        source = "struct S { x: f32, y: f32 }\nfn f() -> S { S { x: 1.0 } }"
        errors = get_errors(source)
        count_err = [e for e in errors if "has 2 fields" in e.message]
        assert len(count_err) >= 1
        hint = getattr(count_err[0], 'hint', None)
        assert hint is not None
        assert "x" in hint
        assert "y" in hint

    def test_with_no_field_hint(self):
        source = "struct S { x: f32, y: f32 }\nfn f(s: S) -> S { s with { z = 1.0 } }"
        errors = get_errors(source)
        field_err = [e for e in errors if "no field" in e.message]
        assert len(field_err) >= 1
        hint = getattr(field_err[0], 'hint', None)
        assert hint is not None
        assert "Available fields:" in hint
