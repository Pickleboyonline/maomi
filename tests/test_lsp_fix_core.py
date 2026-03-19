"""Tests for LSP _core.py stability fixes (C2, C2b, G19, G22)."""

from unittest.mock import patch

from lsprotocol import types

from maomi.lsp import validate, _error_to_diagnostic
from maomi.errors import MaomiError


class TestCoreStability:
    """C2 / C2b: unexpected exceptions in type checker and resolver must not
    crash the LSP server."""

    def test_type_checker_crash_does_not_crash_validate(self):
        """C2: if checker.check() raises an unexpected exception, validate()
        should still return gracefully."""
        source = "fn f(x: f32) -> f32 { x }"
        with patch(
            "maomi.lsp._core.TypeChecker.check",
            side_effect=RecursionError("simulated crash"),
        ):
            diags, result = validate(source, "<test>")
        # Must not raise; diagnostics may be empty (crash swallowed)
        assert isinstance(diags, list)

    def test_resolver_oserror_does_not_crash_validate(self):
        """C2b: if the resolver raises an OSError (e.g. permission denied),
        validate() should still return gracefully."""
        source = 'import "missing_mod";'
        with patch(
            "maomi.lsp._core.resolve",
            side_effect=OSError("Permission denied"),
        ):
            diags, result = validate(source, "<test>")
        assert isinstance(diags, list)
        assert result.program is None  # _EMPTY_RESULT returned


class TestTypeAliasNotEmpty:
    """G22: a file containing only type aliases should not be treated as
    empty."""

    def test_type_alias_only_file_produces_result(self):
        source = "type W = f32[3, 3];"
        diags, result = validate(source, "<test>")
        assert diags == []
        assert result.program is not None


class TestWarningSeverity:
    """G19: _error_to_diagnostic should respect a 'severity' attribute on
    MaomiError when set to 'warning'."""

    def test_default_severity_is_error(self):
        err = MaomiError("some error", "<test>", line=1, col=1)
        diag = _error_to_diagnostic(err)
        assert diag.severity == types.DiagnosticSeverity.Error

    def test_warning_severity(self):
        err = MaomiError("some warning", "<test>", line=1, col=1)
        err.severity = "warning"  # type: ignore[attr-defined]
        diag = _error_to_diagnostic(err)
        assert diag.severity == types.DiagnosticSeverity.Warning

    def test_unknown_severity_falls_back_to_error(self):
        err = MaomiError("some info", "<test>", line=1, col=1)
        err.severity = "info"  # type: ignore[attr-defined]
        diag = _error_to_diagnostic(err)
        assert diag.severity == types.DiagnosticSeverity.Error
