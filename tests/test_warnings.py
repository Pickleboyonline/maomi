"""Tests for the warning analysis pass."""
from maomi.warnings import analyze, Warning
from maomi.lexer import Lexer
from maomi.parser import Parser


def parse(source: str):
    tokens = Lexer(source, filename="<test>").tokenize()
    return Parser(tokens, filename="<test>").parse()


class TestUnusedVariables:
    def test_unused_let_binding(self):
        prog = parse("fn f(x: f32) -> f32 { let y = 1.0; x }")
        warnings = analyze(prog, "<test>")
        msgs = [w.message for w in warnings]
        assert any("y" in m for m in msgs)

    def test_used_variable_no_warning(self):
        prog = parse("fn f(x: f32) -> f32 { let y = x; y }")
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_underscore_prefix_suppresses(self):
        prog = parse("fn f(x: f32) -> f32 { let _y = 1.0; x }")
        warnings = analyze(prog, "<test>")
        msgs = [w.message for w in warnings]
        assert not any("_y" in m for m in msgs)

    def test_unused_param_no_warning(self):
        """Function parameters are intentionally not warned about."""
        prog = parse("fn f(x: f32, y: f32) -> f32 { x }")
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_param_used_no_warning(self):
        prog = parse("fn f(x: f32, y: f32) -> f32 { x + y }")
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_scan_variables(self):
        source = "fn f(xs: f32[10]) -> f32 { scan (carry, elem) in (0.0, xs) { carry + elem } }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_warning_has_hint(self):
        prog = parse("fn f(x: f32) -> f32 { let y = 1.0; x }")
        warnings = analyze(prog, "<test>")
        y_warnings = [w for w in warnings if "y" in w.message]
        assert len(y_warnings) > 0
        assert y_warnings[0].hint is not None
        assert "_y" in y_warnings[0].hint

    def test_map_elem_var_used(self):
        source = "fn f(xs: f32[10]) -> f32[10] { map x in xs { x + 1.0 } }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_fold_variables_used(self):
        source = "fn f(xs: f32[10]) -> f32 { fold (carry, elem) in (0.0, xs) { carry + elem } }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_if_expr_references(self):
        source = "fn f(x: f32, flag: bool) -> f32 { if flag { x } else { 0.0 } }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_variable_used_in_nested_call(self):
        source = "fn f(x: f32) -> f32 { let y = x; sqrt(y) }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_var_warnings = [w for w in warnings if w.kind == "unused_variable"]
        assert len(unused_var_warnings) == 0

    def test_multiple_unused(self):
        source = "fn f(x: f32) -> f32 { let a = 1.0; let b = 2.0; x }"
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        unused_names = {w.message for w in warnings if w.kind == "unused_variable"}
        assert any("a" in m for m in unused_names)
        assert any("b" in m for m in unused_names)

    def test_compiler_generated_names_skipped(self):
        """Compiler-generated names (from destructuring) start with _ and are skipped."""
        # Destructuring desugars into __maomi_destruct_N which starts with _
        source = """
struct Point { x: f32, y: f32 }
fn f(p: Point) -> f32 { let { x, y } = p; x + y }
"""
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        # Should not warn about __maomi_destruct_ names
        for w in warnings:
            assert "__maomi_destruct" not in w.message


class TestUnusedImports:
    def test_unused_selective_import(self):
        source = 'from math import { relu };\nfn f(x: f32) -> f32 { x }'
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        import_warnings = [w for w in warnings if w.kind == "unused_import"]
        assert any("relu" in w.message for w in import_warnings)

    def test_used_import_no_warning(self):
        source = 'from math import { relu };\nfn f(x: f32) -> f32 { relu(x) }'
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        import_warnings = [w for w in warnings if w.kind == "unused_import"]
        assert not any("relu" in w.message for w in import_warnings)

    def test_multiple_imports_partial_use(self):
        source = 'from math import { relu, sigmoid };\nfn f(x: f32) -> f32 { relu(x) }'
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        import_warnings = [w for w in warnings if w.kind == "unused_import"]
        assert any("sigmoid" in w.message for w in import_warnings)
        assert not any("relu" in w.message for w in import_warnings)

    def test_qualified_import_no_warning(self):
        """Qualified imports (without .names) are not checked currently."""
        source = 'import math;\nfn f(x: f32) -> f32 { x }'
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        import_warnings = [w for w in warnings if w.kind == "unused_import"]
        assert len(import_warnings) == 0


class TestUnusedFunctions:
    def test_unused_function_with_symbolic_dim(self):
        source = """
fn helper(x: f32[N]) -> f32[N] { x }
fn main(x: f32) -> f32 { x }
"""
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        fn_warnings = [w for w in warnings if w.kind == "unused_function"]
        assert any("helper" in w.message for w in fn_warnings)

    def test_called_function_no_warning(self):
        source = """
fn helper(x: f32[N]) -> f32[N] { x }
fn main(xs: f32[10]) -> f32[10] { helper(xs) }
"""
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        fn_warnings = [w for w in warnings if w.kind == "unused_function"]
        assert not any("helper" in w.message for w in fn_warnings)

    def test_concrete_function_not_warned(self):
        """Functions with only concrete dims may be entry points -- no warning."""
        source = """
fn entry(x: f32) -> f32 { x }
fn other(y: f32[10]) -> f32[10] { y }
"""
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        fn_warnings = [w for w in warnings if w.kind == "unused_function"]
        assert len(fn_warnings) == 0

    def test_underscore_function_suppressed(self):
        source = """
fn _helper(x: f32[N]) -> f32[N] { x }
fn main(x: f32) -> f32 { x }
"""
        prog = parse(source)
        warnings = analyze(prog, "<test>")
        fn_warnings = [w for w in warnings if w.kind == "unused_function"]
        assert not any("_helper" in w.message for w in fn_warnings)


class TestLSPIntegration:
    def test_warnings_appear_in_validate(self):
        from maomi.lsp._core import validate
        source = "fn f(x: f32) -> f32 { let y = 1.0; x }"
        diags, result = validate(source, "<test>")
        warning_diags = [d for d in diags if d.severity == 2]  # Warning = 2
        # There should be at least one warning for unused 'y'
        assert any("y" in d.message for d in warning_diags)

    def test_warnings_have_correct_severity(self):
        from maomi.lsp._core import validate
        source = "fn f(x: f32) -> f32 { let y = 1.0; x }"
        diags, result = validate(source, "<test>")
        for d in diags:
            if "Unused" in d.message or "unused" in d.message:
                assert d.severity == 2  # DiagnosticSeverity.Warning

    def test_hint_diagnostics_emitted(self):
        from maomi.lsp._core import validate
        source = "fn f(x: f32) -> f32 { let y = 1.0; x }"
        diags, result = validate(source, "<test>")
        hint_diags = [d for d in diags if d.severity == 4]  # Hint = 4
        # At least one hint for the unused variable
        assert any("hint:" in d.message for d in hint_diags)

    def test_no_warnings_on_clean_code(self):
        from maomi.lsp._core import validate
        source = "fn f(x: f32) -> f32 { x + 1.0 }"
        diags, result = validate(source, "<test>")
        warning_diags = [d for d in diags if d.severity == 2]
        assert len(warning_diags) == 0

    def test_warning_to_diagnostics_conversion(self):
        from maomi.lsp._core import _warning_to_diagnostics
        w = Warning(
            message="Unused variable 'y'",
            filename="<test>",
            line=1, col=23, col_end=24,
            hint="If this is intentional, prefix with underscore: _y",
            kind="unused_variable",
        )
        diags = _warning_to_diagnostics(w)
        assert len(diags) == 2  # Warning + Hint
        diag = diags[0]
        assert diag.severity == 2  # Warning
        assert diag.source == "maomi"
        assert "y" in diag.message
        # Line/col should be 0-indexed
        assert diag.range.start.line == 0
        assert diag.range.start.character == 22
        # Hint diagnostic
        hint_diag = diags[1]
        assert hint_diag.severity == 4  # Hint
        assert "hint:" in hint_diag.message
