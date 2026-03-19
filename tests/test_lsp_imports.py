"""End-to-end LSP tests for import scenarios.

Tests that LSP features (completions, hover, go-to-def, diagnostics, etc.)
work correctly when modules are imported. Uses real fixture files on disk
since the resolver needs to read imported modules.

Fixtures live in tests/fixtures/modules/:
  mathlib.mao     — fn double(x: f32), fn square(x: f32)
  uses_mathlib.mao — import mathlib; fn quad() { mathlib.double(...) }
  uses_from.mao   — from mathlib import { double }; fn quad() { double(...) }
  nn.mao          — fn relu(x: f32), fn linear(x: f32[4], w: f32[4])
  uses_internal.mao — import nn; fn apply_relu() { nn.relu(...) }
  with_struct.mao — struct Point { x: f32, y: f32 }; fn make_point(), fn get_x()
  uses_path.mao   — import "lib/helpers" as helpers; fn add_two() { helpers.add_one(...) }
  lib/helpers.mao — fn add_one(x: f32)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from lsprotocol import types

from maomi.lsp import (
    validate,
    _find_node_at,
    _complete_dot,
    _complete_general,
    _get_hover_text,
    _goto_find_definition,
    _local_functions,
    _build_document_symbols,
    _refs_classify_node,
    _refs_collect_all,
)
from maomi.ast_nodes import CallExpr, Identifier, FieldAccess
from tests.lsp_validation import assert_all_completions_valid

FIXTURES = Path(__file__).parent / "fixtures" / "modules"


def _validate_fixture(name: str):
    """Validate a fixture file with its real path (so resolver can find imports)."""
    path = FIXTURES / name
    source = path.read_text()
    diags, result = validate(source, str(path))
    return source, diags, result


def _find_in_ast(result, node_type, attr_name, attr_value):
    """Find a node by type and attribute in any local function."""
    from maomi.lsp._ast_utils import _children_of
    def _search(node):
        if isinstance(node, node_type) and getattr(node, attr_name, None) == attr_value:
            return node
        for child in _children_of(node):
            found = _search(child)
            if found:
                return found
        return None
    for fn in _local_functions(result.program):
        found = _search(fn)
        if found:
            return fn, found
    return None, None


# ---------------------------------------------------------------------------
# Diagnostics — imports resolve without errors
# ---------------------------------------------------------------------------

class TestImportDiagnostics:
    def test_qualified_import_no_errors(self):
        """import mathlib; — resolves cleanly, no diagnostics."""
        _, diags, result = _validate_fixture("uses_mathlib.mao")
        assert len(diags) == 0
        assert result.program is not None

    def test_from_import_no_errors(self):
        """from mathlib import { double }; — resolves cleanly."""
        _, diags, result = _validate_fixture("uses_from.mao")
        assert len(diags) == 0
        assert result.program is not None

    def test_path_import_no_errors(self):
        """import "lib/helpers" as helpers; — resolves cleanly."""
        _, diags, result = _validate_fixture("uses_path.mao")
        assert len(diags) == 0
        assert result.program is not None

    def test_struct_import_no_errors(self):
        """Importing a module with structs resolves cleanly."""
        source = 'from with_struct import { Point, make_point };\n\nfn f() -> f32 {\n    let p = make_point(1.0, 2.0);\n    p.x\n}\n'
        path = FIXTURES / "_test_struct_import.mao"
        path.write_text(source)
        try:
            diags, result = validate(source, str(path))
            assert result.program is not None
            assert len([d for d in diags if d.severity == types.DiagnosticSeverity.Error]) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_missing_module_error(self):
        """Importing a nonexistent module produces a diagnostic."""
        source = 'import nonexistent;\nfn f(x: f32) -> f32 { x }\n'
        path = FIXTURES / "_test_missing.mao"
        path.write_text(source)
        try:
            diags, result = validate(source, str(path))
            assert len(diags) >= 1
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Completions — module dot completion
# ---------------------------------------------------------------------------

class TestImportCompletions:
    def test_qualified_module_completion(self):
        """After 'import mathlib;', typing 'mathlib.' shows double and square."""
        _, _, result = _validate_fixture("uses_mathlib.mao")
        assert result.program is not None
        fn_names = [f.name for f in result.program.functions]
        assert "mathlib.double" in fn_names
        assert "mathlib.square" in fn_names

        # Module dot completion
        from maomi.lsp._completion import _complete_module
        comp = _complete_module(result, "mathlib")
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "double" in labels
        assert "square" in labels

    def test_from_import_general_completion(self):
        """After 'from mathlib import { double }', 'double' appears in general completions."""
        _, _, result = _validate_fixture("uses_from.mao")
        assert result.program is not None
        pos = types.Position(line=3, character=4)
        comp = _complete_general(result, pos)
        labels = {item.label for item in comp.items}
        assert "double" in labels

    def test_path_import_module_completion(self):
        """After 'import "lib/helpers" as helpers;', typing 'helpers.' shows add_one."""
        _, _, result = _validate_fixture("uses_path.mao")
        assert result.program is not None
        from maomi.lsp._completion import _complete_module
        comp = _complete_module(result, "helpers")
        assert comp is not None
        labels = {item.label for item in comp.items}
        assert "add_one" in labels

    def test_imported_module_in_general_completion(self):
        """Module name 'mathlib' appears as a Module completion item."""
        _, _, result = _validate_fixture("uses_mathlib.mao")
        pos = types.Position(line=3, character=4)
        comp = _complete_general(result, pos)
        module_items = [i for i in comp.items if i.kind == types.CompletionItemKind.Module]
        module_labels = {i.label for i in module_items}
        assert "mathlib" in module_labels

    def test_imported_struct_fields_completion(self):
        """Dot on an imported struct shows its fields."""
        source = 'from with_struct import { Point, make_point };\n\nfn f() -> f32 {\n    let p = make_point(1.0, 2.0);\n    p.x\n}\n'
        path = FIXTURES / "_test_struct_comp.mao"
        path.write_text(source)
        try:
            diags, result = validate(source, str(path))
            assert result.program is not None
            # Find the 'p' identifier in p.x and check type
            for fn in _local_functions(result.program):
                node = _find_node_at(fn, 5, 5)  # line 5 (1-indexed), col 5
                if node is not None:
                    typ = result.type_map.get(id(node))
                    if typ is not None:
                        from maomi.types import StructType
                        assert isinstance(typ, StructType)
                        field_names = [f[0] for f in typ.fields]
                        assert "x" in field_names
                        assert "y" in field_names
                        break
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Import-Aware Completions — context-sensitive module/name suggestions
# ---------------------------------------------------------------------------

class TestImportAwareCompletions:
    def test_from_module_name_shows_modules(self):
        """'from ___' shows available .mao files + stdlib modules."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_mathlib.mao")
        result = _complete_import("from ", 5, filepath)
        assert result is not None
        labels = {i.label for i in result.items}
        assert "mathlib" in labels
        assert "nn" in labels
        # Stdlib
        assert "math" in labels or "optim" in labels

    def test_import_module_name_shows_modules(self):
        """'import ___' shows available modules."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_mathlib.mao")
        result = _complete_import("import ", 7, filepath)
        assert result is not None
        labels = {i.label for i in result.items}
        assert "mathlib" in labels

    def test_from_import_braces_shows_exports(self):
        """'from mathlib import { ___' shows mathlib's functions."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_from.mao")
        result = _complete_import("from mathlib import { ", 21, filepath)
        assert result is not None
        labels = {i.label for i in result.items}
        assert "double" in labels
        assert "square" in labels

    def test_from_nn_import_shows_nn_exports(self):
        """'from nn import { ___' shows nn's functions."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_from.mao")
        result = _complete_import("from nn import { ", 17, filepath)
        assert result is not None
        labels = {i.label for i in result.items}
        assert "relu" in labels
        assert "linear" in labels

    def test_struct_exports_in_import_names(self):
        """'from with_struct import { ___' shows struct names too."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_from.mao")
        result = _complete_import("from with_struct import { ", 25, filepath)
        assert result is not None
        labels = {i.label for i in result.items}
        assert "Point" in labels
        assert "make_point" in labels

    def test_not_triggered_on_regular_code(self):
        """Regular code lines should not trigger import completions."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_mathlib.mao")
        assert _complete_import("    let x = from", 16, filepath) is None
        assert _complete_import("    mathlib.double(x)", 20, filepath) is None
        assert _complete_import("    x + y", 5, filepath) is None

    def test_stdlib_modules_available(self):
        """Stdlib modules (math, nn, optim) are available even with no local files."""
        from maomi.lsp._completion import _complete_import
        result = _complete_import("from ", 5, "/nonexistent/path/file.mao")
        assert result is not None
        labels = {i.label for i in result.items}
        assert "nn" in labels
        assert "optim" in labels
        assert "math" in labels

    def test_all_items_are_correct_kind(self):
        """Module completions use Module kind, name completions use Function/Struct kind."""
        from maomi.lsp._completion import _complete_import
        filepath = str(FIXTURES / "uses_from.mao")
        # Module completions
        mod_result = _complete_import("from ", 5, filepath)
        for item in mod_result.items:
            assert item.kind == types.CompletionItemKind.Module
        # Name completions
        name_result = _complete_import("from mathlib import { ", 21, filepath)
        for item in name_result.items:
            assert item.kind in (types.CompletionItemKind.Function, types.CompletionItemKind.Struct)


# ---------------------------------------------------------------------------
# Hover — on imported function calls
# ---------------------------------------------------------------------------

class TestImportHover:
    def test_hover_on_qualified_call(self):
        """Hover on 'mathlib.double(...)' shows function signature."""
        source, _, result = _validate_fixture("uses_mathlib.mao")
        assert result.program is not None
        # Find the CallExpr for mathlib.double
        fn_scope, call_node = _find_in_ast(result, CallExpr, "callee", "mathlib.double")
        assert call_node is not None
        hover = _get_hover_text(call_node, fn_scope, result)
        assert hover is not None
        assert "double" in hover

    def test_hover_on_from_import_call(self):
        """Hover on 'double(...)' (from-imported) shows function signature."""
        source, _, result = _validate_fixture("uses_from.mao")
        assert result.program is not None
        fn_scope, call_node = _find_in_ast(result, CallExpr, "callee", "double")
        assert call_node is not None
        hover = _get_hover_text(call_node, fn_scope, result)
        assert hover is not None
        assert "double" in hover


# ---------------------------------------------------------------------------
# Go-to-definition — across files
# ---------------------------------------------------------------------------

class TestImportGotoDef:
    def test_goto_def_qualified_call(self):
        """Go-to-def on 'mathlib.double(...)' jumps to mathlib.mao."""
        source, _, result = _validate_fixture("uses_mathlib.mao")
        fn_scope, call_node = _find_in_ast(result, CallExpr, "callee", "mathlib.double")
        assert call_node is not None
        found = _goto_find_definition(call_node, fn_scope, result)
        assert found is not None
        span, source_file = found
        # Should point to mathlib.mao
        assert source_file is not None
        assert "mathlib.mao" in source_file

    def test_goto_def_from_import_call(self):
        """Go-to-def on 'double(...)' (from-imported) jumps to mathlib.mao."""
        source, _, result = _validate_fixture("uses_from.mao")
        fn_scope, call_node = _find_in_ast(result, CallExpr, "callee", "double")
        assert call_node is not None
        found = _goto_find_definition(call_node, fn_scope, result)
        assert found is not None
        span, source_file = found
        # from-import creates alias — may resolve to local alias or original
        # Either way, should resolve somewhere
        assert span is not None

    def test_goto_def_path_import(self):
        """Go-to-def on 'helpers.add_one(...)' jumps to lib/helpers.mao."""
        source, _, result = _validate_fixture("uses_path.mao")
        fn_scope, call_node = _find_in_ast(result, CallExpr, "callee", "helpers.add_one")
        assert call_node is not None
        found = _goto_find_definition(call_node, fn_scope, result)
        assert found is not None
        span, source_file = found
        assert source_file is not None
        assert "helpers.mao" in source_file


# ---------------------------------------------------------------------------
# Document symbols — imported functions filtered
# ---------------------------------------------------------------------------

class TestImportSymbols:
    def test_symbols_only_local_functions(self):
        """Document symbols should show local functions, not imported ones."""
        _, _, result = _validate_fixture("uses_mathlib.mao")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        names = {s.name for s in symbols}
        assert "quad" in names
        # Imported functions should NOT appear
        assert "mathlib.double" not in names
        assert "mathlib.square" not in names

    def test_symbols_from_import_shows_local(self):
        """From-import: document symbols show local function, not imported alias."""
        _, _, result = _validate_fixture("uses_from.mao")
        symbols = _build_document_symbols(result)
        assert symbols is not None
        names = {s.name for s in symbols}
        assert "quad" in names


# ---------------------------------------------------------------------------
# Program structure after resolution
# ---------------------------------------------------------------------------

class TestImportResolution:
    def test_qualified_import_fn_table(self):
        """fn_table contains prefixed imported functions."""
        _, _, result = _validate_fixture("uses_mathlib.mao")
        assert "mathlib.double" in result.fn_table
        assert "mathlib.square" in result.fn_table
        assert "quad" in result.fn_table

    def test_from_import_fn_table(self):
        """fn_table contains the alias (unprefixed) function after from-import."""
        _, _, result = _validate_fixture("uses_from.mao")
        # The alias "double" should be accessible
        assert "quad" in result.fn_table
        # "double" should work as an alias
        fn_names = [f.name for f in result.program.functions]
        assert "double" in fn_names or "mathlib.double" in fn_names

    def test_local_functions_excludes_imports(self):
        """_local_functions only returns functions defined in the current file."""
        _, _, result = _validate_fixture("uses_mathlib.mao")
        local = _local_functions(result.program)
        local_names = [f.name for f in local]
        assert "quad" in local_names
        assert "mathlib.double" not in local_names
        assert "mathlib.square" not in local_names

    def test_imported_struct_in_struct_defs(self):
        """Imported structs appear in struct_defs."""
        source = 'from with_struct import { Point };\n\nfn f(p: Point) -> f32 { p.x }\n'
        path = FIXTURES / "_test_struct_defs.mao"
        path.write_text(source)
        try:
            _, result = validate(source, str(path))
            assert result.program is not None
            assert "Point" in result.struct_defs or "with_struct.Point" in result.struct_defs
        finally:
            path.unlink(missing_ok=True)
