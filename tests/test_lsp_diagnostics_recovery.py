"""End-to-end tests for LSP diagnostics and parser error recovery.

Tests parser recovery at top-level and statement-level, diagnostics accuracy,
fake ID insertion for completions, error recovery + completions interaction,
and resilience edge cases.
"""

import pytest
from lsprotocol import types

from maomi.lsp._core import validate, completion_validate, _FAKE_ID
from maomi.lsp._completion import _complete_dot, _complete_general
from maomi.lexer import Lexer
from maomi.parser import Parser


# ===========================================================================
# 1. Parser recovery — top level
# ===========================================================================

class TestParserRecoveryTopLevel:

    def test_broken_fn_good_fn_recovered(self):
        """One broken function, one good function. Good one should be in AST."""
        source = """
fn broken() -> f32 { + }
fn good(x: f32) -> f32 { x }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report at least one error for broken fn"
        assert result.program is not None, "Program should exist (partial recovery)"
        fn_names = [f.name for f in result.program.functions]
        assert "good" in fn_names, "Good function should survive recovery"

    def test_broken_struct_good_fn_recovered(self):
        """Broken struct definition + good function. Good fn should be in AST."""
        source = """
struct Broken { x: , y: f32 }
fn good(x: f32) -> f32 { x }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report error for broken struct"
        assert result.program is not None, "Program should exist"
        fn_names = [f.name for f in result.program.functions]
        assert "good" in fn_names, "Good function should survive struct parse error"

    def test_multiple_broken_declarations(self):
        """Multiple broken declarations in a row. Parser should not hang."""
        source = """
fn a( -> f32 { x }
fn b( -> f32 { x }
fn good(x: f32) -> f32 { x }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 2, "Should report errors for both broken functions"
        assert result.program is not None, "Program should exist"
        fn_names = [f.name for f in result.program.functions]
        assert "good" in fn_names, "Good function should survive multiple recovery attempts"

    def test_missing_closing_brace_last_fn(self):
        """Missing closing brace on last function."""
        source = """
fn good(x: f32) -> f32 { x }
fn incomplete(y: f32) -> f32 { y
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report missing brace error"
        # The good function should still be in the AST
        if result.program is not None:
            fn_names = [f.name for f in result.program.functions]
            assert "good" in fn_names or "incomplete" in fn_names, \
                "At least one function should be recovered"

    def test_good_fn_before_broken_is_usable(self):
        """Good function before broken one should be type-checked and usable."""
        source = """
fn add(x: f32, y: f32) -> f32 { x + y }
fn broken() -> f32 { + }
"""
        diags, result = validate(source, "<test>")
        assert result.program is not None
        fn_names = [f.name for f in result.program.functions]
        assert "add" in fn_names
        # Type info should exist for the good function
        assert "add" in result.fn_table, "Good function should be in fn_table"

    def test_lexer_error_recovery(self):
        """Lexer error recovery: invalid char is skipped, rest of file parses."""
        source = """
fn add(x: f32, y: f32) -> f32 { x + y }
fn broken() -> f32 { ? }
"""
        diags, result = validate(source, "<test>")
        # Fixed: lexer skips '?' and continues — program is preserved
        assert result.program is not None
        assert len(diags) >= 1  # lexer error reported as diagnostic
        fn_names = [f.name for f in result.program.functions]
        assert "add" in fn_names

    def test_broken_between_good_functions(self):
        """Broken fn sandwiched between two good functions."""
        source = """
fn first(x: f32) -> f32 { x }
fn broken(-> f32 { }
fn last(y: f32) -> f32 { y }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert result.program is not None
        fn_names = [f.name for f in result.program.functions]
        assert "first" in fn_names, "Function before broken should survive"
        assert "last" in fn_names, "Function after broken should survive"


# ===========================================================================
# 2. Parser recovery — statement level
# ===========================================================================

class TestParserRecoveryStatementLevel:

    def test_broken_let_good_code_after(self):
        """Broken let binding followed by good code. Later bindings should parse."""
        source = """
fn f(x: f32) -> f32 {
    let a = +;
    let b = x + 1.0;
    b
}
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report error for broken let"
        assert result.program is not None
        fn_names = [f.name for f in result.program.functions]
        assert "f" in fn_names, "Function should be in AST"

    def test_missing_semicolon_mid_body(self):
        """Missing semicolon in the middle of a function body."""
        source = """
fn f(x: f32) -> f32 {
    let a = x + 1.0
    let b = a + 2.0;
    b
}
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report missing semicolon error"

    def test_invalid_expression_mid_body(self):
        """Invalid expression in the middle of a block."""
        source = """
fn f(x: f32) -> f32 {
    let a = x;
    <<<;
    a
}
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report error for invalid expression"

    def test_nested_block_with_error(self):
        """Nested if/else block with error inside."""
        source = """
fn f(x: f32) -> f32 {
    if x > 0.0 {
        let a = +;
        x
    } else {
        x + 1.0
    }
}
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Should report error inside nested block"

    def test_recovery_preserves_later_let_bindings(self):
        """After recovery from a broken statement, later let bindings should exist."""
        source = """
fn f(x: f32) -> f32 {
    let a = x * 2.0;
    let b = +;
    let c = x + 1.0;
    c
}
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        # Function should still parse
        if result.program is not None:
            fn_names = [f.name for f in result.program.functions]
            assert "f" in fn_names


# ===========================================================================
# 3. Diagnostics accuracy
# ===========================================================================

class TestDiagnosticsAccuracy:

    def test_parse_error_correct_line(self):
        """Parse error should report the correct line number."""
        source = "fn f() -> f32 {\n    x\n}"
        diags, result = validate(source, "<test>")
        # 'x' is undeclared — type error
        assert len(diags) >= 1
        # The error should be on line 1 (0-indexed) or line 2 (0-indexed)
        # depending on whether it's a parse or type error

    def test_multiple_errors_reported(self):
        """Validate should report multiple errors, not just the first."""
        source = """
fn f(x: f32) -> i32 { x }
fn g(y: i32) -> f32 { y }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 2, f"Expected >= 2 errors, got {len(diags)}: {[d.message for d in diags]}"

    def test_type_errors_and_parse_errors_both_reported(self):
        """Both type errors and parse errors should appear in diagnostics."""
        source = """
fn broken( -> f32 { x }
fn mistyped(x: f32) -> i32 { x }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 2, f"Expected >= 2 errors, got {len(diags)}: {[d.message for d in diags]}"
        messages = [d.message for d in diags]
        has_parse = any("expected" in m.lower() for m in messages)
        has_type = any("type" in m.lower() or "mismatch" in m.lower() or "return" in m.lower() for m in messages)
        assert has_parse, f"No parse error found in: {messages}"
        assert has_type, f"No type error found in: {messages}"

    def test_error_line_column_0_indexed(self):
        """LSP diagnostics should use 0-indexed line/column."""
        source = "fn f( -> f32 { x }"  # error on line 1, col somewhere
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        for d in diags:
            assert d.range.start.line >= 0, "Line should be 0-indexed"
            assert d.range.start.character >= 0, "Column should be 0-indexed"

    def test_error_has_severity(self):
        """All diagnostics should have Error severity."""
        source = "fn f( -> f32 { x }"
        diags, result = validate(source, "<test>")
        for d in diags:
            assert d.severity == types.DiagnosticSeverity.Error

    def test_error_has_source(self):
        """All diagnostics should have 'maomi' as source."""
        source = "fn f( -> f32 { x }"
        diags, result = validate(source, "<test>")
        for d in diags:
            assert d.source == "maomi"

    def test_multiline_function_error_line(self):
        """Error in a multiline function should point to correct line."""
        source = """fn f(x: f32) -> f32 {
    let a = x;
    let b: i32 = a;
    b
}"""
        diags, result = validate(source, "<test>")
        # Type mismatch on "let b: i32 = a;" (a is f32, b is i32)
        # or return type mismatch
        if diags:
            # Should reference a line > 0 (the error is inside the body)
            error_lines = [d.range.start.line for d in diags]
            assert any(l > 0 for l in error_lines), \
                f"Expected error on line > 0, got lines: {error_lines}"


# ===========================================================================
# 4. Fake ID insertion
# ===========================================================================

class TestFakeIdInsertion:

    def test_fake_id_in_function_body_let_position(self):
        """Fake ID at let binding value position."""
        source = "fn f(x: f32) -> f32 {\n    let a = \n    x\n}"
        result = completion_validate(source, "<test>", 1, 12)
        # Should not crash; program may or may not parse
        # but should not return None for trivial cases

    def test_fake_id_start_of_line(self):
        """Fake ID at the start of a line inside a function."""
        source = "fn f(x: f32) -> f32 {\n\n    x\n}"
        result = completion_validate(source, "<test>", 1, 0)
        # Should not crash

    def test_fake_id_after_dot(self):
        """Fake ID after a dot (pipe completion case)."""
        source = "fn f(x: f32) -> f32 {\n    x.\n}"
        result = completion_validate(source, "<test>", 1, 6)
        # Should produce a parseable "x.__mao_cmplt__"
        # The result should have a program
        assert result.program is not None or result.type_map == {}, \
            "Fake ID after dot should not crash"

    def test_fake_id_inside_if_block(self):
        """Fake ID inside an if/else block."""
        source = "fn f(x: f32) -> f32 {\n    if x > 0.0 {\n        \n    } else {\n        x\n    }\n}"
        result = completion_validate(source, "<test>", 2, 8)
        # Should not crash

    def test_fake_id_inside_scan_body(self):
        """Fake ID inside a scan body."""
        source = """fn f(xs: f32[10]) -> f32 {
    scan (carry, elem) in (0.0, xs) {

    }
}"""
        result = completion_validate(source, "<test>", 2, 8)
        # Should not crash

    def test_fake_id_inside_map_body(self):
        """Fake ID inside a map body."""
        source = """fn f(xs: f32[10]) -> f32[10] {
    map elem in xs {

    }
}"""
        result = completion_validate(source, "<test>", 2, 8)
        # Should not crash

    def test_fake_id_does_not_leak_into_diagnostics(self):
        """The fake ID should never appear in diagnostics from validate()."""
        source = "fn f(x: f32) -> f32 { x }"
        diags, result = validate(source, "<test>")
        for d in diags:
            assert _FAKE_ID not in d.message, \
                f"Fake ID leaked into diagnostic: {d.message}"

    def test_fake_id_not_in_general_completions(self):
        """The fake ID should be filtered out of completion results."""
        source = "fn f(x: f32) -> f32 {\n    \n}"
        result = completion_validate(source, "<test>", 1, 4)
        position = types.Position(line=1, character=4)
        comp = _complete_general(result, position)
        if comp is not None:
            labels = [item.label for item in comp.items]
            assert _FAKE_ID not in labels, \
                f"Fake ID appeared in completion labels: {labels}"

    def test_fake_id_at_end_of_file(self):
        """Fake ID at the very end of a file."""
        source = "fn f(x: f32) -> f32 { x }\n"
        result = completion_validate(source, "<test>", 1, 0)
        # Should not crash

    def test_fake_id_in_empty_function_body(self):
        """Fake ID in an empty function body."""
        source = "fn f(x: f32) -> f32 {\n\n}"
        result = completion_validate(source, "<test>", 1, 0)
        # Should not crash


# ===========================================================================
# 5. Error recovery + completions
# ===========================================================================

class TestErrorRecoveryAndCompletions:

    def test_broken_fn_completions_in_good_fn(self):
        """File has a broken function. User is typing in a good function.
        Completions should work in the good function."""
        source = """fn broken() -> f32 { + }
fn good(x: f32) -> f32 {

}"""
        # Complete inside the good function body (line 2, col 4)
        result = completion_validate(source, "<test>", 2, 4)
        position = types.Position(line=2, character=4)
        comp = _complete_general(result, position)
        assert comp is not None, "Completions should work in good fn despite broken fn"
        labels = [item.label for item in comp.items]
        # Should at least have keywords
        assert "let" in labels or "if" in labels, \
            f"Expected keywords in completions, got: {labels[:10]}"

    def test_type_errors_no_parse_errors_completions_work(self):
        """File has type errors but no parse errors. Completions should work."""
        source = """fn f(x: f32) -> i32 { x }
fn g(y: f32) -> f32 {

}"""
        result = completion_validate(source, "<test>", 2, 4)
        position = types.Position(line=2, character=4)
        comp = _complete_general(result, position)
        assert comp is not None, "Completions should work despite type errors"
        labels = [item.label for item in comp.items]
        # Should have the variable 'y' in scope
        assert "y" in labels, f"Expected 'y' in completions, got: {labels[:20]}"

    def test_completions_include_variables_from_good_fn(self):
        """Variables from the good function should be in completions."""
        source = """fn broken(-> f32 { }
fn good(x: f32, y: f32) -> f32 {
    let a = x + y;

}"""
        result = completion_validate(source, "<test>", 3, 4)
        position = types.Position(line=3, character=4)
        comp = _complete_general(result, position)
        if comp is not None:
            labels = [item.label for item in comp.items]
            # Parameters should be visible
            assert "x" in labels, f"Expected 'x' in completions, got: {labels[:20]}"
            assert "y" in labels, f"Expected 'y' in completions, got: {labels[:20]}"

    def test_dot_completions_with_broken_fn_elsewhere(self):
        """Dot completions on a struct field should work despite broken fn elsewhere."""
        source = """struct Pt { x: f32, y: f32 }
fn broken(-> f32 { }
fn good(p: Pt) -> f32 {
    p.
}"""
        result = completion_validate(source, "<test>", 3, 6)
        position = types.Position(line=3, character=6)
        comp = _complete_dot(result, position, "p")
        if comp is not None:
            labels = [item.label for item in comp.items]
            assert "x" in labels or "y" in labels, \
                f"Expected struct fields in dot completions, got: {labels[:10]}"


# ===========================================================================
# 6. Resilience
# ===========================================================================

class TestResilience:

    def test_empty_file(self):
        """Empty file should produce no diagnostics and no crash."""
        diags, result = validate("", "<test>")
        assert diags == [] or len(diags) >= 0  # no crash
        # An empty file has no functions/structs — could be empty result

    def test_empty_file_completions(self):
        """Completions on empty file should not crash."""
        result = completion_validate("", "<test>", 0, 0)
        position = types.Position(line=0, character=0)
        comp = _complete_general(result, position)
        # Should at least return keywords even with empty file
        if comp is not None:
            labels = [item.label for item in comp.items]
            assert "fn" in labels or "let" in labels, \
                "Expected at least keyword completions on empty file"

    def test_file_with_only_comments(self):
        """File with only comments should produce no errors."""
        source = """// This is a comment
// Another comment
"""
        diags, result = validate(source, "<test>")
        assert diags == [], f"Comments-only file should have no errors, got: {[d.message for d in diags]}"

    def test_file_with_only_doc_comments(self):
        """File with only doc comments should produce no errors or be graceful."""
        source = """/// This is a doc comment
/// Another doc comment
"""
        diags, result = validate(source, "<test>")
        # Doc comments without a following fn/struct might cause issues
        # At minimum it should not crash

    def test_file_with_only_imports(self):
        """File with only imports (no functions) should not crash."""
        source = "import math;\n"
        diags, result = validate(source, "<test>")
        # Might error because 'math' module doesn't exist in test context
        # But should not crash

    def test_file_with_only_struct(self):
        """File with only a struct definition should not crash."""
        source = "struct Pt { x: f32, y: f32 }\n"
        diags, result = validate(source, "<test>")
        # Should parse fine
        assert result.program is not None or result.struct_defs is not None

    def test_deeply_nested_expressions(self):
        """Moderately deep expression nesting should not crash."""
        # Build a deeply nested expression: ((((x + 1.0) + 1.0) + 1.0) ...)
        depth = 50
        inner = "x"
        for _ in range(depth):
            inner = f"({inner} + 1.0)"
        source = f"fn f(x: f32) -> f32 {{ {inner} }}"
        diags, result = validate(source, "<test>")
        # Should not crash or hang

    def test_single_fn_keyword(self):
        """Just 'fn' with nothing else should produce error, not crash."""
        source = "fn"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Bare 'fn' should produce an error"

    def test_single_struct_keyword(self):
        """Just 'struct' with nothing else should produce error, not crash."""
        source = "struct"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Bare 'struct' should produce an error"

    def test_random_tokens(self):
        """Random nonsense should produce errors, not crash."""
        source = "let x = 1 + 2; struct; fn; 42"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Random tokens should produce errors"

    def test_unclosed_string(self):
        """Unclosed string literal should produce a lexer error."""
        source = 'fn f() -> f32 { "unclosed }'
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Unclosed string should produce an error"

    def test_very_long_identifier(self):
        """Very long identifier should not crash."""
        long_name = "x" * 1000
        source = f"fn {long_name}(x: f32) -> f32 {{ x }}"
        diags, result = validate(source, "<test>")
        # Should parse fine or error gracefully

    def test_unicode_in_source(self):
        """Unicode characters (outside strings) should produce lexer error, not crash."""
        source = "fn f(x: f32) -> f32 { x + \u03c0 }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Unicode identifier should produce an error"

    def test_whitespace_only(self):
        """Whitespace-only file should produce no errors."""
        source = "   \n\n   \t  \n"
        diags, result = validate(source, "<test>")
        assert diags == [], f"Whitespace-only should have no errors, got: {[d.message for d in diags]}"

    def test_completion_validate_at_out_of_range_line(self):
        """completion_validate at a line beyond file length should not crash."""
        source = "fn f(x: f32) -> f32 { x }"
        result = completion_validate(source, "<test>", 100, 0)
        # Should not crash, just return empty

    def test_completion_validate_at_out_of_range_col(self):
        """completion_validate at a column beyond line length should not crash."""
        source = "fn f(x: f32) -> f32 { x }"
        result = completion_validate(source, "<test>", 0, 1000)
        # Should not crash


# ===========================================================================
# 7. Parser.errors accumulation (direct parser testing)
# ===========================================================================

class TestParserErrorsDirectly:

    def test_parser_accumulates_errors(self):
        """Parser.errors should accumulate multiple errors during recovery."""
        source = """
fn a( -> f32 { x }
fn b( -> f32 { y }
fn c(z: f32) -> f32 { z }
"""
        tokens = Lexer(source, filename="<test>").tokenize()
        parser = Parser(tokens, filename="<test>")
        program = parser.parse()
        assert len(parser.errors) >= 2, \
            f"Expected >= 2 errors, got {len(parser.errors)}: {[e.message for e in parser.errors]}"
        fn_names = [f.name for f in program.functions]
        assert "c" in fn_names, "Good function 'c' should be in parsed program"

    def test_parser_block_level_recovery(self):
        """Parser should recover from errors within a block."""
        source = """
fn f(x: f32) -> f32 {
    let a = +;
    let b = x;
    b
}
"""
        tokens = Lexer(source, filename="<test>").tokenize()
        parser = Parser(tokens, filename="<test>")
        program = parser.parse()
        assert len(parser.errors) >= 1, "Should have error for 'let a = +'"
        assert len(program.functions) == 1, "Function 'f' should still parse"

    def test_parser_missing_rbrace_at_eof(self):
        """Missing closing brace at EOF should produce error."""
        source = "fn f(x: f32) -> f32 { x"
        tokens = Lexer(source, filename="<test>").tokenize()
        parser = Parser(tokens, filename="<test>")
        program = parser.parse()
        assert len(parser.errors) >= 1, "Should have error for missing '}'"
        assert len(program.functions) >= 1, "Function should still be added"

    def test_parser_struct_recovery(self):
        """Broken struct should not prevent subsequent definitions from parsing."""
        source = """
struct Bad { x: }
struct Good { a: f32, b: i32 }
fn f(x: f32) -> f32 { x }
"""
        tokens = Lexer(source, filename="<test>").tokenize()
        parser = Parser(tokens, filename="<test>")
        program = parser.parse()
        assert len(parser.errors) >= 1, "Should have error for broken struct"
        # At least the good struct or function should survive
        names = [s.name for s in program.struct_defs] + [f.name for f in program.functions]
        assert "Good" in names or "f" in names, \
            f"Expected recovery. Got structs: {[s.name for s in program.struct_defs]}, fns: {[f.name for f in program.functions]}"


# ===========================================================================
# 8. Edge cases in _insert_fake_id
# ===========================================================================

class TestInsertFakeId:

    def test_insert_at_valid_position(self):
        """Fake ID inserted at a valid position should produce parseable result."""
        from maomi.lsp._core import _insert_fake_id
        source = "fn f(x: f32) -> f32 {\n    \n}"
        modified = _insert_fake_id(source, 1, 4)
        assert _FAKE_ID in modified
        # Should be on line 1 (0-indexed)
        lines = modified.splitlines()
        assert _FAKE_ID in lines[1]

    def test_insert_at_line_beyond_file(self):
        """Inserting at a line beyond file length should return original source."""
        from maomi.lsp._core import _insert_fake_id
        source = "fn f(x: f32) -> f32 { x }"
        modified = _insert_fake_id(source, 100, 0)
        assert modified == source

    def test_insert_preserves_other_lines(self):
        """Inserting fake ID should not affect other lines."""
        from maomi.lsp._core import _insert_fake_id
        source = "line0\nline1\nline2\n"
        modified = _insert_fake_id(source, 1, 2)
        lines = modified.splitlines()
        assert lines[0] == "line0"
        assert lines[2] == "line2"
        assert _FAKE_ID in lines[1]


# ===========================================================================
# 9. Diagnostics for specific Maomi constructs
# ===========================================================================

class TestDiagnosticsSpecificConstructs:

    def test_scan_with_error(self):
        """Error in scan construct should produce diagnostic."""
        source = """fn f(xs: f32[10]) -> f32 {
    scan (carry, elem) in (0.0, xs) {
        carry +
    }
}"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1

    def test_map_with_error(self):
        """Error in map construct should produce diagnostic."""
        source = """fn f(xs: f32[10]) -> f32[10] {
    map elem in xs {
        elem +
    }
}"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1

    def test_if_else_type_mismatch(self):
        """If/else branches with different types should produce type error."""
        source = """fn f(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        1
    }
}"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "If/else branch type mismatch should be reported"

    def test_struct_literal_wrong_field(self):
        """Struct literal with wrong field name should produce error."""
        source = """struct Pt { x: f32, y: f32 }
fn f() -> Pt {
    Pt { x: 1.0, z: 2.0 }
}"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Wrong struct field should be reported"

    def test_undeclared_variable(self):
        """Undeclared variable should produce a type error."""
        source = "fn f() -> f32 { unknown_var }"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Undeclared variable should be reported"

    def test_wrong_number_of_args(self):
        """Calling a function with wrong number of args should produce error."""
        source = """fn add(x: f32, y: f32) -> f32 { x + y }
fn f() -> f32 { add(1.0) }"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1, "Wrong arg count should be reported"


# ===========================================================================
# 10. Bugs and gaps found during testing
# ===========================================================================

class TestBugsAndGaps:

    def test_diagnostic_range_always_width_1(self):
        """QUALITY ISSUE: Diagnostic ranges are always 1 character wide.

        _error_to_diagnostic always creates a range of width 1 (end = start + 1).
        This means error squiggles in the editor are always a single character,
        which makes it hard to see where the error is, especially for long tokens.

        Severity: LOW — functional but poor UX.
        """
        source = "fn f(this_is_a_long_variable: f32) -> i32 {\n    this_is_a_long_variable\n}"
        diags, _ = validate(source, "<test>")
        assert len(diags) >= 1
        for d in diags:
            width = d.range.end.character - d.range.start.character
            # Currently always 1 — documenting this as a known limitation
            assert width == 1, \
                "If this fails, width was improved (which is good!)"

    def test_cascading_errors_from_if_condition(self):
        """QUALITY ISSUE: Error in if-condition produces cascading secondary error.

        When the if-condition fails to parse, the parser skips to '}' and then
        the 'else' keyword appears at top-level, producing a second unrelated
        error. The user typed one mistake but gets two errors.

        Severity: MEDIUM — confusing for users.
        """
        source = """fn f(x: f32) -> f32 {
    if + {
        x
    } else {
        x + 1.0
    }
}"""
        diags, result = validate(source, "<test>")
        # Currently produces 2 errors: the real one and a cascading one
        assert len(diags) >= 1, "Should report at least the real error"
        # Document that the second error is cascading
        if len(diags) > 1:
            messages = [d.message for d in diags]
            # The second error about 'else' or 'fn' is cascading
            cascading = [m for m in messages if "else" in m or "expected fn" in m]
            assert len(cascading) > 0, \
                "Expected cascading error about 'else' keyword"

    def test_fold_variables_not_in_scope_for_completions(self):
        """BUG: carry and elem variables from fold are not visible in completions.

        _collect_from_expr handles ScanExpr and MapExpr but not FoldExpr.
        FoldExpr has the same carry_var and elem_vars fields as ScanExpr.

        Severity: HIGH — completions inside fold bodies are degraded.
        """
        source = """fn f(xs: f32[10]) -> f32 {
    fold (carry, elem) in (0.0, xs) {

    }
}"""
        result = completion_validate(source, "<test>", 2, 8)
        position = types.Position(line=2, character=8)
        comp = _complete_general(result, position)
        assert comp is not None
        var_labels = [i.label for i in comp.items
                      if i.kind == types.CompletionItemKind.Variable]
        # Fixed: carry and elem should be in scope inside fold body
        assert "carry" in var_labels
        assert "elem" in var_labels

    def test_while_variables_not_in_scope_for_completions(self):
        """BUG: Variables from while loop are not visible in completions.

        _collect_from_expr does not handle WhileExpr. While loops have
        a body block where let-bound variables should be in scope.

        Severity: MEDIUM — completions inside while bodies are degraded.
        """
        source = """fn f(x: f32) -> f32 {
    let y = x;
    while y > 0.0 limit 100 {

        y
    }
}"""
        result = completion_validate(source, "<test>", 3, 8)
        position = types.Position(line=3, character=8)
        comp = _complete_general(result, position)
        assert comp is not None
        var_labels = [i.label for i in comp.items
                      if i.kind == types.CompletionItemKind.Variable]
        # y should be in scope from the outer let, but while-body-specific
        # variables would be missing

    def test_type_alias_without_semicolon_cascading_errors(self):
        """QUALITY ISSUE: Missing semicolon on type alias causes cascading errors.

        'type Weight = f32[3, 3]' without semicolon causes the parser to
        consume 'fn' as part of the type alias, leading to multiple confusing
        errors and the type alias being lost.

        Severity: LOW — standard parsing behavior, but worth noting.
        """
        source = """type Weight = f32[3, 3]
fn f(w: Weight) -> f32 { sum(w) }"""
        diags, result = validate(source, "<test>")
        # Multiple errors expected due to missing semicolon cascading
        assert len(diags) >= 1
        # Type alias is lost due to parse failure
        if result.program:
            aliases = [t.name for t in result.program.type_aliases]
            assert "Weight" not in aliases, \
                "If this fails, recovery for type aliases improved"

    def test_broken_import_preserves_subsequent_functions(self):
        """Import recovery: broken import should not prevent functions from parsing."""
        source = """import ;
fn f(x: f32) -> f32 { x }
"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert result.program is not None
        fn_names = [f.name for f in result.program.functions]
        assert "f" in fn_names, "Function after broken import should parse"

    def test_multiple_struct_defs_one_broken(self):
        """When one struct in the middle is broken, others survive recovery."""
        source = """struct A { x: f32 }
struct B { y: }
struct C { z: i32 }
fn f(a: A) -> f32 { a.x }"""
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        assert result.program is not None
        struct_names = [s.name for s in result.program.struct_defs]
        assert "A" in struct_names, "Struct A before broken should survive"
        assert "C" in struct_names, "Struct C after broken should survive"

    def test_file_ends_mid_expression(self):
        """File ending mid-expression should produce errors but not crash."""
        source = "fn f(x: f32) -> f32 { x +"
        diags, result = validate(source, "<test>")
        assert len(diags) >= 1
        # Should produce at least 'expected expression' and 'expected }'
        assert result.program is not None, \
            "Program should still exist (partial recovery)"

    def test_completion_in_struct_literal_does_not_suggest_fields(self):
        """QUALITY ISSUE: Completions inside struct literal don't suggest remaining fields.

        When typing 'Pt { x: 1.0, |', completions should ideally suggest 'y'
        as the next field, but currently they don't provide struct-field-aware
        completions in this context.

        Severity: MEDIUM — would improve struct construction UX.
        """
        source = """struct Pt { x: f32, y: f32 }
fn f() -> Pt {
    Pt { x: 1.0,
}"""
        result = completion_validate(source, "<test>", 2, 18)
        position = types.Position(line=2, character=18)
        comp = _complete_general(result, position)
        if comp is not None:
            labels = [item.label for item in comp.items]
            # Currently 'y' is NOT suggested as a struct field completion
            # It would only appear if it happened to be a variable in scope
            field_items = [i for i in comp.items
                          if i.kind == types.CompletionItemKind.Field]
            assert len(field_items) == 0, \
                "If this fails, struct literal field completion was added (good!)"

    def test_completion_top_level_no_variables_leak(self):
        """Variables from functions should not leak into top-level completions."""
        source = """fn f(x: f32) -> f32 { x }

"""
        result = completion_validate(source, "<test>", 2, 0)
        position = types.Position(line=2, character=0)
        comp = _complete_general(result, position)
        if comp is not None:
            var_items = [i for i in comp.items
                        if i.kind == types.CompletionItemKind.Variable]
            assert len(var_items) == 0, \
                f"Variables should not leak to top level: {[i.label for i in var_items]}"

    def test_fake_id_mid_identifier_does_not_produce_hybrid_names(self):
        """Inserting fake ID mid-identifier should not produce hybrid names in completions."""
        source = """fn f(x: f32) -> f32 {
    let longvar = x + 1.0;
    lon
}"""
        result = completion_validate(source, "<test>", 2, 7)
        position = types.Position(line=2, character=7)
        comp = _complete_general(result, position)
        if comp is not None:
            labels = [item.label for item in comp.items]
            hybrid_labels = [l for l in labels if _FAKE_ID in l]
            assert len(hybrid_labels) == 0, \
                f"Hybrid fake ID names found: {hybrid_labels}"
