"""Tests for LSP Audit Round 2 - Unit A: Resilience fixes."""
from __future__ import annotations

import pytest
from lsprotocol import types

from maomi.lsp import validate, _complete_general, _complete_module, AnalysisResult
from maomi.lsp._completion import (
    _is_pipe_compatible, _annotation_matches_type,
    _pipe_completions, _is_complex_builtin_pipe_compatible,
)
from maomi.lsp._builtin_data import _KEYWORDS, _BUILTIN_NAMESPACES
from maomi.types import ScalarType, StructType, StructArrayType


# ---------------------------------------------------------------------------
# C1: Resolver error fallback — broken import still yields a usable program
# ---------------------------------------------------------------------------

class TestC1ResolverFallback:
    def test_broken_import_returns_program(self):
        """When resolve() fails (e.g. missing module), validate should still
        return the pre-resolve program so IDE features keep working."""
        source = 'import nonexistent;\nfn f(x: f32) -> f32 { x }'
        diags, result = validate(source, "<test>")
        # Should have a diagnostic for the import error
        assert any("nonexistent" in d.message.lower() or "import" in d.message.lower()
                    for d in diags)
        # Program should NOT be None — pre-resolve AST is preserved
        assert result.program is not None
        fn_names = [fn.name for fn in result.program.functions]
        assert "f" in fn_names

    def test_broken_import_completions_still_work(self):
        """General completions should still include locally-defined functions
        even when imports fail."""
        source = 'import nonexistent;\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 { y }'
        diags, result = validate(source, "<test>")
        assert result.program is not None
        # fn_table will be empty (type checker never ran), but program.functions has them
        pos = types.Position(line=2, character=0)
        completions = _complete_general(result, pos)
        labels = [item.label for item in completions.items]
        # Local function should appear via AST scan
        assert "helper" in labels
        assert "main" in labels


# ---------------------------------------------------------------------------
# B1: Generic/wildcard/comptime functions visible in completions
# ---------------------------------------------------------------------------

class TestB1GenericFunctions:
    def test_wildcard_function_in_general_completions(self):
        """Functions with f32[..] wildcard params should appear in general completions
        even though only monomorphized copies exist in fn_table."""
        source = (
            'fn double(x: f32[..]) -> f32[..] { x + x }\n'
            'fn caller(a: f32[3]) -> f32[3] { double(a) }\n'
        )
        _, result = validate(source, "<test>")
        assert result.program is not None
        pos = types.Position(line=1, character=0)
        completions = _complete_general(result, pos)
        labels = [item.label for item in completions.items]
        assert "double" in labels

    def test_generic_function_in_general_completions(self):
        """Functions with generic T params should appear in general completions."""
        source = (
            'struct Point { x: f32, y: f32 }\n'
            'fn identity(x: T) -> T { x }\n'
            'fn caller(p: Point) -> Point { identity(p) }\n'
        )
        _, result = validate(source, "<test>")
        assert result.program is not None
        pos = types.Position(line=2, character=0)
        completions = _complete_general(result, pos)
        labels = [item.label for item in completions.items]
        assert "identity" in labels

    def test_comptime_function_in_general_completions(self):
        """Functions with comptime params should appear in general completions."""
        source = (
            'fn reduce(x: f32[3, 4], comptime axis: i32) -> f32[4] { sum(x, axis=0) }\n'
            'fn caller(a: f32[3, 4]) -> f32[4] { reduce(a, axis=0) }\n'
        )
        _, result = validate(source, "<test>")
        assert result.program is not None
        pos = types.Position(line=1, character=0)
        completions = _complete_general(result, pos)
        labels = [item.label for item in completions.items]
        assert "reduce" in labels

    def test_no_duplicate_for_concrete_functions(self):
        """Concrete functions in fn_table should NOT be duplicated by AST scan."""
        source = 'fn f(x: f32) -> f32 { x }\n'
        _, result = validate(source, "<test>")
        assert result.program is not None
        pos = types.Position(line=0, character=0)
        completions = _complete_general(result, pos)
        f_items = [item for item in completions.items if item.label == "f"]
        assert len(f_items) == 1


# ---------------------------------------------------------------------------
# B17 partial: StructArrayType in pipe compatibility
# ---------------------------------------------------------------------------

class TestB17StructArrayTypePipe:
    def test_is_pipe_compatible_ew_struct_array(self):
        """Elementwise builtins (exp, sqrt, etc.) should be pipe-compatible
        with StructArrayType."""
        st = StructType("Point", (("x", ScalarType("f32")), ("y", ScalarType("f32"))))
        sat = StructArrayType(st, (3,))
        fake_sig = type("Sig", (), {"param_types": [ScalarType("f32")]})()
        assert _is_pipe_compatible(sat, "exp", fake_sig)

    def test_is_pipe_compatible_struct_name_match(self):
        """StructArrayType should match a StructType first param with same name."""
        st = StructType("Point", (("x", ScalarType("f32")),))
        sat = StructArrayType(st, (3,))
        fake_sig = type("Sig", (), {"param_types": [st]})()
        assert _is_pipe_compatible(sat, "my_fn", fake_sig)

    def test_annotation_matches_struct_array(self):
        """StructArrayType should match annotation with struct's name."""
        from maomi.ast_nodes import TypeAnnotation, Span
        ann = TypeAnnotation(base="Point", dims=None, span=Span(1, 1, 1, 5))
        st = StructType("Point", (("x", ScalarType("f32")),))
        sat = StructArrayType(st, (3,))
        assert _annotation_matches_type(ann, sat, {})

    def test_complex_builtin_pipe_compat_struct_array(self):
        """StructArrayType should work with reduction category complex builtins."""
        st = StructType("Params", (("w", ScalarType("f32")),))
        sat = StructArrayType(st, (5,))
        assert _is_complex_builtin_pipe_compatible(sat, "reduction")
        assert _is_complex_builtin_pipe_compatible(sat, "stop_grad")


# ---------------------------------------------------------------------------
# B20: Missing random builtins in namespace
# ---------------------------------------------------------------------------

class TestB20RandomBuiltins:
    def test_exponential_in_random_namespace(self):
        """random.exponential should be in _BUILTIN_NAMESPACES['random']."""
        assert "exponential" in _BUILTIN_NAMESPACES["random"]

    def test_randint_in_random_namespace(self):
        """random.randint should be in _BUILTIN_NAMESPACES['random']."""
        assert "randint" in _BUILTIN_NAMESPACES["random"]


# ---------------------------------------------------------------------------
# G2: comptime keyword in completions
# ---------------------------------------------------------------------------

class TestG2ComptimeKeyword:
    def test_comptime_in_keywords(self):
        """comptime should be in _KEYWORDS."""
        assert "comptime" in _KEYWORDS

    def test_comptime_in_general_completions(self):
        """comptime should appear as a keyword in general completions."""
        source = 'fn f(x: f32) -> f32 { x }\n'
        _, result = validate(source, "<test>")
        pos = types.Position(line=0, character=0)
        completions = _complete_general(result, pos)
        labels = [item.label for item in completions.items]
        assert "comptime" in labels
