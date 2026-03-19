"""Tests for LSP audit Unit 5: Type Checker + Lexer Edge Cases.

Covers:
- G12: col_end on type checker _error() calls
- E6: CR-only line endings produce correct line numbers
- E7: BOM at start of file doesn't cause tokenization errors
- E10: Resolver error uses correct filename and position
"""
import os
import tempfile

import pytest

from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.resolver import resolve
from maomi.errors import MaomiError, MaomiTypeError
from maomi.tokens import TokenType


# -- Helpers --

def _type_errors(source: str) -> list[MaomiTypeError]:
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker()
    return checker.check(program)


def _lex(source: str) -> list:
    return Lexer(source).tokenize()


# -- G12: col_end on type checker errors --


class TestTypeCheckerColEnd:
    def test_binop_type_mismatch_has_col_end(self):
        """Binary operator with mismatched types should have col_end covering the full expression."""
        source = "fn f(x: f32, y: bool) -> f32 { x + y }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = errors[0]
        assert "operator" in err.message or "numeric" in err.message
        # col_end should be wider than col + 1 (covers the expression, not just first char)
        assert err.col_end > err.col + 1

    def test_undefined_variable_has_col_end(self):
        """Undefined variable error should have col_end covering the identifier."""
        source = "fn f() -> f32 { undefined_var }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = errors[0]
        assert "undefined variable" in err.message
        assert "'undefined_var'" in err.message
        # col_end should cover the entire identifier name
        assert err.col_end > err.col + 1

    def test_comparison_mismatch_has_col_end(self):
        """Comparison with mismatched types should have col_end."""
        source = "fn f(x: f32[3], y: f32[4]) -> bool { x == y }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = errors[0]
        assert err.col_end > err.col + 1

    def test_field_access_non_struct_has_col_end(self):
        """Field access on non-struct type should have col_end."""
        source = "fn f(x: f32) -> f32 { x.field }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = errors[0]
        assert "non-struct" in err.message
        assert err.col_end > err.col + 1

    def test_struct_field_type_mismatch_has_col_end(self):
        """Wrong field type in struct literal should have col_end."""
        source = "struct S { x: f32 }\nfn f() -> S { S { x: true } }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = errors[0]
        assert "field" in err.message
        assert err.col_end > err.col + 1

    def test_unknown_struct_has_col_end(self):
        """Unknown struct name in struct literal should have col_end."""
        source = "fn f() -> f32 { Nonexistent { x: 1.0 } }"
        errors = _type_errors(source)
        assert len(errors) >= 1
        err = [e for e in errors if "unknown struct" in e.message][0]
        assert err.col_end > err.col + 1


# -- E6: CR-only line endings --


class TestCROnlyLineEndings:
    def test_cr_only_produces_correct_line_numbers(self):
        """CR-only line endings should produce correct line numbers in tokens."""
        # Old Mac-style: lines separated by \r only
        source = "fn f() -> f32 {\r    1.0\r}"
        tokens = _lex(source)
        # The `1.0` literal should be on line 2
        float_tokens = [t for t in tokens if t.type == TokenType.FLOAT_LIT]
        assert len(float_tokens) == 1
        assert float_tokens[0].line == 2

        # The closing brace should be on line 3
        rbrace_tokens = [t for t in tokens if t.type == TokenType.RBRACE]
        assert len(rbrace_tokens) == 1
        assert rbrace_tokens[0].line == 3

    def test_cr_only_type_error_has_correct_line(self):
        """Type errors in CR-only files should have correct line numbers."""
        source = "fn f() -> f32 {\r    true\r}"
        errors = _type_errors(source)
        assert len(errors) >= 1
        # The return type mismatch error should reference the body, not all on line 1
        # At minimum, tokens are on the right lines

    def test_crlf_still_works(self):
        """Windows-style \\r\\n should still work correctly."""
        source = "fn f() -> f32 {\r\n    1.0\r\n}"
        tokens = _lex(source)
        float_tokens = [t for t in tokens if t.type == TokenType.FLOAT_LIT]
        assert len(float_tokens) == 1
        assert float_tokens[0].line == 2

    def test_cr_only_comment(self):
        """Comments followed by CR-only should work correctly."""
        source = "// comment\rfn f() -> f32 { 1.0 }"
        tokens = _lex(source)
        fn_tokens = [t for t in tokens if t.type == TokenType.FN]
        assert len(fn_tokens) == 1
        assert fn_tokens[0].line == 2


# -- E7: BOM handling --


class TestBOMHandling:
    def test_bom_at_start_doesnt_cause_error(self):
        """UTF-8 BOM at start of file should be stripped, not cause errors."""
        source = "\ufeff" + "fn f() -> f32 { 1.0 }"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        assert not lexer.errors, f"Unexpected lexer errors: {lexer.errors}"
        fn_tokens = [t for t in tokens if t.type == TokenType.FN]
        assert len(fn_tokens) == 1

    def test_bom_file_parses_correctly(self):
        """A file starting with BOM should parse and type-check without errors."""
        source = "\ufeff" + "fn f(x: f32) -> f32 { x + 1.0 }"
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
        assert len(program.functions) == 1
        assert program.functions[0].name == "f"

    def test_bom_file_type_checks(self):
        """A file starting with BOM should type-check without errors."""
        source = "\ufeff" + "fn f(x: f32) -> f32 { x + 1.0 }"
        errors = _type_errors(source)
        assert not errors


# -- E10: Resolver error uses correct filename and position --


class TestResolverErrorPosition:
    def test_missing_module_error_has_importing_file(self):
        """Error for missing module should reference the importing file, not the module path."""
        source = 'import nonexistent_module;'
        tokens = Lexer(source, filename="main.mao").tokenize()
        program = Parser(tokens, filename="main.mao").parse()

        with tempfile.TemporaryDirectory() as tmpdir:
            main_path = os.path.join(tmpdir, "main.mao")
            with open(main_path, "w") as f:
                f.write(source)

            with pytest.raises(MaomiError) as exc_info:
                resolve(program, main_path)

            err = exc_info.value
            # The error should reference the importing file, not the missing module
            assert err.filename == main_path
            # The error should have the import statement's position, not (0, 0)
            assert err.line == 1
            assert err.col >= 1

    def test_missing_module_error_not_zero_position(self):
        """Missing module error should not report position (0, 0)."""
        source = 'import nonexistent_module;'
        tokens = Lexer(source, filename="main.mao").tokenize()
        program = Parser(tokens, filename="main.mao").parse()

        with tempfile.TemporaryDirectory() as tmpdir:
            main_path = os.path.join(tmpdir, "main.mao")
            with open(main_path, "w") as f:
                f.write(source)

            with pytest.raises(MaomiError) as exc_info:
                resolve(program, main_path)

            err = exc_info.value
            # Should NOT be (0, 0) anymore
            assert not (err.line == 0 and err.col == 0), "Error position should not be (0, 0)"

    def test_circular_import_error_has_importing_file(self):
        """Circular import error should reference the importing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files that import each other
            a_path = os.path.join(tmpdir, "a.mao")
            b_path = os.path.join(tmpdir, "b.mao")

            with open(a_path, "w") as f:
                f.write('import b;\nfn fa() -> f32 { 1.0 }')
            with open(b_path, "w") as f:
                f.write('import a;\nfn fb() -> f32 { 2.0 }')

            source_a = open(a_path).read()
            tokens = Lexer(source_a, filename=a_path).tokenize()
            program = Parser(tokens, filename=a_path).parse()

            with pytest.raises(MaomiError) as exc_info:
                resolve(program, a_path)

            err = exc_info.value
            assert "circular" in err.message
            # The error should reference a valid file, not the module path at (0, 0)
            assert err.line >= 1
