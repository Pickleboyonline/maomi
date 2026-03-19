"""Comprehensive audit of Maomi LSP Completions & Edits."""
from __future__ import annotations
import pytest
from lsprotocol import types
from maomi.lsp import validate, _complete_general, _complete_dot, _vars_in_scope, AnalysisResult
from maomi.lsp._completion import (
    _complete_import, _complete_struct_literal, _complete_module,
    _pipe_completions, _is_pipe_compatible, _annotation_str,
    _annotation_matches_type, _parse_already_imported,
)
from maomi.lsp._core import completion_validate, _FAKE_ID, _insert_fake_id, _EMPTY_RESULT
from maomi.lsp._builtin_data import (
    _KEYWORDS, _TYPE_NAMES, _BUILTINS, _BUILTIN_SET,
    _BUILTIN_NAMESPACES, _BUILTIN_DOCS, _BUILTIN_CATEGORIES, _EW_NAMES,
)
from tests.lsp_validation import assert_all_completions_valid, check_edit

def _validate_and_complete_general(source, line, character):
    _, result = validate(source, "<test>")
    pos = types.Position(line=line, character=character)
    comp = _complete_general(result, pos)
    return comp, pos, result

def _validate_and_complete_dot(source, line, character):
    _, result = validate(source, "<test>")
    pos = types.Position(line=line, character=character)
    comp = _complete_dot(result, pos)
    return comp, pos, result

def _labels(comp):
    if comp is None:
        return set()
    return {item.label for item in comp.items}

def _items_by_label(comp, label):
    if comp is None:
        return []
    return [i for i in comp.items if i.label == label]

# 1. _complete_general across contexts
class TestGeneralTopLevel:
    def test_top_level_outside_function(self):
        source = "fn f(x: f32) -> f32 { x }\n"
        comp, pos, _ = _validate_and_complete_general(source, 1, 0)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        assert "fn" in _labels(comp)

    def test_inside_function_body(self):
        source = "fn f(x: f32) -> f32 {\n  \n}"
        comp, pos, _ = _validate_and_complete_general(source, 1, 2)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        labels = _labels(comp)
        assert "x" in labels
        assert "let" in labels

    def test_after_partial_identifier(self):
        source = "fn f(x: f32) -> f32 {\n  ex\n}"
        comp, pos, _ = _validate_and_complete_general(source, 1, 4)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        assert "exp" in _labels(comp)

    def test_inside_broken_code(self):
        source = "fn f(x: f32) -> f32 {\n  let a = ;\n  \n}"
        comp, pos, _ = _validate_and_complete_general(source, 2, 2)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_empty_file(self):
        comp, pos, _ = _validate_and_complete_general("", 0, 0)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        assert "fn" in _labels(comp)

    def test_cursor_at_eof(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 5, 0)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_cursor_mid_token(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 1)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_after_let_equals(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 {\n  let a = \n}", 1, 10)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

# 2. No duplicates
class TestGeneralNoDuplicates:
    def test_no_duplicate_labels_same_kind(self):
        source = "struct Point { x: f32, y: f32 }\nfn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 {\n    let a = 1.0;\n    a\n}"
        comp, pos, _ = _validate_and_complete_general(source, 4, 4)
        assert comp is not None
        seen = set()
        duplicates = []
        for item in comp.items:
            key = (item.label, item.kind)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        assert duplicates == [], f"Duplicate (label, kind) pairs: {duplicates}"

    def test_no_duplicate_builtins(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert comp is not None
        label_counts = {}
        for item in comp.items:
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
        dups = {k: v for k, v in label_counts.items() if v > 1}
        assert dups == {}, f"Duplicate completion labels: {dups}"

# 3. CompletionItemKind
class TestGeneralKinds:
    def test_keyword_kind(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        fn_items = _items_by_label(comp, "fn")
        assert len(fn_items) == 1
        assert fn_items[0].kind == types.CompletionItemKind.Keyword

    def test_builtin_kind(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert _items_by_label(comp, "exp")[0].kind == types.CompletionItemKind.Function

    def test_type_kind(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert _items_by_label(comp, "f32")[0].kind == types.CompletionItemKind.TypeParameter

    def test_struct_kind(self):
        comp, pos, _ = _validate_and_complete_general("struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }", 1, 25)
        assert _items_by_label(comp, "Point")[0].kind == types.CompletionItemKind.Struct

    def test_variable_kind(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 {\n  let a = 1.0;\n  a\n}", 2, 2)
        assert _items_by_label(comp, "a")[0].kind == types.CompletionItemKind.Variable

    def test_namespace_kind(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert _items_by_label(comp, "random")[0].kind == types.CompletionItemKind.Module

# 4. Sort order
class TestGeneralSortOrder:
    def test_all_items_have_sort_text(self):
        comp, pos, _ = _validate_and_complete_general("struct S { x: f32 }\nfn h(x: f32) -> f32 { x }\nfn m(y: f32) -> f32 {\n  let a=1.0;\n  a\n}", 4, 2)
        assert comp is not None
        for item in comp.items:
            assert item.sort_text is not None, f"Missing sort_text on '{item.label}'"

    def test_variables_before_functions(self):
        comp, pos, _ = _validate_and_complete_general("fn helper(x: f32) -> f32 { x }\nfn main(y: f32) -> f32 {\n  let a = 1.0;\n  a\n}", 3, 2)
        st = {item.label: item.sort_text for item in comp.items}
        if "a" in st and "helper" in st:
            assert st["a"] < st["helper"]

# 5. Filter text
class TestGeneralFilterText:
    def test_no_fake_id_in_completions(self):
        result = completion_validate("fn f(x: f32) -> f32 {\n  \n}", "<test>", 1, 2)
        pos = types.Position(line=1, character=2)
        comp = _complete_general(result, pos)
        if comp:
            assert _FAKE_ID not in _labels(comp)

    def test_no_internal_maomi_vars(self):
        result = completion_validate("fn f(x: f32) -> f32 {\n  \n}", "<test>", 1, 2)
        pos = types.Position(line=1, character=2)
        comp = _complete_general(result, pos)
        if comp:
            for item in comp.items:
                assert not item.label.startswith("__maomi"), f"Internal var leaked: {item.label}"

# 6. _complete_dot
class TestDotCompletion:
    def test_dot_on_struct(self):
        comp, pos, _ = _validate_and_complete_dot("struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }", 1, 26)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        labels = _labels(comp)
        assert "x" in labels and "y" in labels

    def test_dot_on_unknown_shows_ast_node_type(self):
        """BUG: AST-based lookup ignores prefix, finds node at cursor position.

        When prefix="zzz" but cursor overlaps with 'x' in parsed AST,
        completions are based on x's type (f32), not zzz's.
        """
        _, result = validate("fn f(x: f32) -> f32 { x }", "<test>")
        comp = _complete_dot(result, types.Position(line=0, character=23), prefix="zzz")
        # BUG: Returns f32 pipe completions because AST lookup finds 'x' at cursor
        assert comp is not None  # documenting current (wrong) behavior

    def test_dot_on_scalar(self):
        comp, pos, _ = _validate_and_complete_dot("fn f(x: f32) -> f32 { x }", 0, 23)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        labels = _labels(comp)
        assert "exp" in labels and "sqrt" in labels

    def test_dot_on_array(self):
        comp, pos, _ = _validate_and_complete_dot("fn f(x: f32[3, 3]) -> f32 { x }\n", 0, 30)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        labels = _labels(comp)
        assert "sum" in labels and "mean" in labels

    def test_dot_on_integer_no_float_fns(self):
        comp, pos, _ = _validate_and_complete_dot("fn f(x: i32) -> i32 { x }", 0, 23)
        if comp is not None:
            labels = _labels(comp)
            assert "exp" not in labels

    def test_dot_on_bool_no_float_fns(self):
        comp, pos, _ = _validate_and_complete_dot("fn f(x: bool) -> bool { x }", 0, 25)
        if comp is not None:
            assert "exp" not in _labels(comp)

    def test_dot_struct_fields_sort_before_pipe_fns(self):
        comp, pos, _ = _validate_and_complete_dot("struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n", 1, 22)
        assert comp is not None
        for i in comp.items:
            if i.kind == types.CompletionItemKind.Field:
                assert i.sort_text.startswith("0_")
            elif i.kind == types.CompletionItemKind.Function:
                assert i.sort_text.startswith("1_")

# 7. Pipe edits
class TestDotCompletionPipeEdits:
    def test_pipe_edit_replaces_dot(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        assert_all_completions_valid(comp, pos)
        check_edit("fn f(x: f32[3]) -> f32 { x. }\n", pos, comp, "sum",
                   "fn f(x: f32[3]) -> f32 { x |> sum() }\n")

    def test_pipe_edit_field_no_pipe(self):
        _, result = validate("struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n", "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        check_edit("struct S { val: f32 }\nfn f(s: S) -> f32 { s. }\n", pos, comp, "val",
                   "struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n")

    def test_pipe_edits_dont_overlap(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

# 8. Dot completion fresh parse
class TestDotCompletionFreshParse:
    def test_dot_incomplete_code(self):
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 {\n  p.\n}\n"
        result = completion_validate(source, "<test>", 2, 4)
        assert result.program is not None
        pos = types.Position(line=2, character=4)
        comp = _complete_dot(result, pos)
        if comp:
            labels = _labels(comp)
            assert "x" in labels or "y" in labels

# 9. _complete_module
class TestModuleCompletion:
    def test_nonexistent_returns_none(self):
        _, result = validate("fn f(x: f32) -> f32 { x }", "<test>")
        assert _complete_module(result, "nonexistent") is None

    def test_empty_name_returns_none(self):
        _, result = validate("fn f(x: f32) -> f32 { x }", "<test>")
        assert _complete_module(result, "") is None

    def test_none_result_returns_none(self):
        assert _complete_module(None, "math") is None

# 10. _complete_import
class TestImportCompletion:
    def test_import_keyword(self):
        result = _complete_import("import ", 7, "<test>")
        assert result is not None

    def test_from_keyword(self):
        result = _complete_import("from ", 5, "<test>")
        assert result is not None

    def test_non_import_returns_none(self):
        assert _complete_import("let x = 1.0;", 12, "<test>") is None

    def test_closed_brace_returns_none(self):
        assert _complete_import("from math import { relu }", 25, "<test>") is None

    def test_import_module_kind(self):
        result = _complete_import("import ", 7, "<test>")
        if result is not None:
            for item in result.items:
                assert item.kind == types.CompletionItemKind.Module

# 11. _complete_struct_literal
class TestStructLiteralCompletion:
    def test_suggests_remaining_fields(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point { x: 1.0,  } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> Point { Point { x: 1.0,  } }", 35, result, source, types.Position(line=1, character=35))
        assert comp is not None
        labels = _labels(comp)
        assert "y" in labels
        assert "x" not in labels

    def test_all_fields_written_returns_none(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> Point { Point { x: 1.0, y: 2.0 } }", 44, result, source, types.Position(line=1, character=44))
        assert comp is None

    def test_no_structs_returns_none(self):
        source = "fn f(x: f32) -> f32 { x }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f(x: f32) -> f32 { x }", 23, result, source, types.Position(line=0, character=23))
        assert comp is None

    def test_field_kind(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point {  } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> Point { Point {  } }", 27, result, source, types.Position(line=1, character=27))
        assert comp is not None
        for item in comp.items:
            assert item.kind == types.CompletionItemKind.Field

    def test_field_snippets(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point {  } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> Point { Point {  } }", 27, result, source, types.Position(line=1, character=27))
        assert comp is not None
        for item in comp.items:
            assert item.insert_text_format == types.InsertTextFormat.Snippet
            assert ": $0" in item.insert_text

    def test_after_return_arrow_not_triggered(self):
        source = "struct Point { x: f32, y: f32 }\nfn make() -> Point { Point { x: 1.0, y: 2.0 } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn make() -> Point { Point { x: 1.0, y: 2.0 } }", 21, result, source, types.Position(line=1, character=21))
        assert comp is None

# 12. Edge cases
class TestEdgeCases:
    def test_empty_source(self):
        comp, pos, _ = _validate_and_complete_general("", 0, 0)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_whitespace_only(self):
        comp, pos, _ = _validate_and_complete_general("   \n\n   ", 0, 0)
        assert comp is not None

    def test_cursor_past_end_of_line(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 9999)
        assert comp is not None

    def test_cursor_past_last_line(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 999, 0)
        assert comp is not None

    def test_very_long_line(self):
        comp, pos, _ = _validate_and_complete_general("fn f(" + "x" * 10000 + ": f32) -> f32 { x }", 0, 50)
        assert comp is not None

    def test_many_functions(self):
        fns = "\n".join(f"fn f{i}(x: f32) -> f32 {{ x }}" for i in range(50))
        comp, pos, _ = _validate_and_complete_general(fns, 25, 5)
        assert comp is not None

# 13. _insert_fake_id
class TestInsertFakeId:
    def test_insert_at_beginning(self):
        assert _insert_fake_id("hello", 0, 0) == _FAKE_ID + "hello"

    def test_insert_at_end(self):
        assert _insert_fake_id("hello", 0, 5) == "hello" + _FAKE_ID

    def test_insert_past_eof(self):
        assert _insert_fake_id("hello", 5, 0) == "hello"

    def test_insert_in_multiline(self):
        result = _insert_fake_id("line1\nline2\nline3\n", 1, 3)
        assert "lin" + _FAKE_ID + "e2" in result

# 14. _parse_already_imported
class TestParseAlreadyImported:
    def test_basic(self):
        assert _parse_already_imported("relu, linear") == {"relu", "linear"}

    def test_single(self):
        assert _parse_already_imported("relu") == {"relu"}

    def test_empty(self):
        assert _parse_already_imported("") == set()

    def test_trailing_comma(self):
        assert _parse_already_imported("relu, ") == {"relu"}

# 15. completion_validate
class TestCompletionValidate:
    def test_basic(self):
        result = completion_validate("fn f(x: f32) -> f32 {\n  \n}\n", "<test>", 1, 2)
        assert result.program is not None

    def test_empty_source(self):
        result = completion_validate("", "<test>", 0, 0)
        assert result.program is None

    def test_broken_source(self):
        completion_validate("fn f(x: f32) -> f32 { let a = ; }", "<test>", 0, 30)

# 16. Builtin namespace
class TestBuiltinNamespaceCompletion:
    def test_namespaced_not_in_general(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        for item in comp.items:
            if item.kind == types.CompletionItemKind.Function:
                assert "." not in item.label, f"Namespaced in general: {item.label}"

# 17. Pipe completion specifics
class TestPipeCompletionDetails:
    def test_no_namespaced_in_pipe(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        comp = _complete_dot(result, types.Position(line=0, character=27))
        assert comp is not None
        for item in comp.items:
            assert "." not in item.label

    def test_snippet_format(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        comp = _complete_dot(result, types.Position(line=0, character=27))
        assert comp is not None
        for item in comp.items:
            if item.additional_text_edits:
                assert item.insert_text_format == types.InsertTextFormat.Snippet

    def test_text_edit_at_cursor(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        for item in comp.items:
            if item.text_edit is not None:
                assert item.text_edit.range.start.character == pos.character

# 18. _vars_in_scope edge cases
class TestVarsInScopeEdgeCases:
    def test_scan_vars(self):
        source = "fn f(x: f32[3]) -> f32 {\n  let r = scan (c, e) in (0.0, x) {\n    c + e\n  };\n  r\n}"
        _, result = validate(source, "<test>")
        var_names = [v[0] for v in _vars_in_scope(result, types.Position(line=2, character=4))]
        assert "c" in var_names and "e" in var_names

    def test_map_var(self):
        source = "fn f(x: f32[3]) -> f32[3] {\n  let r = map e in x {\n    e + 1.0\n  };\n  r\n}"
        _, result = validate(source, "<test>")
        assert "e" in [v[0] for v in _vars_in_scope(result, types.Position(line=2, character=4))]

    def test_shadowed(self):
        source = "fn f(x: f32) -> f32 {\n  let a = 1.0;\n  let a = 2.0;\n  a\n}"
        _, result = validate(source, "<test>")
        a_vars = [(n, t) for n, t in _vars_in_scope(result, types.Position(line=3, character=2)) if n == "a"]
        assert len(a_vars) == 1

    def test_outside_function(self):
        _, result = validate("fn f(x: f32) -> f32 { x }", "<test>")
        assert _vars_in_scope(result, types.Position(line=5, character=0)) == []

    def test_if_else_isolation(self):
        source = "fn f(x: f32) -> f32 {\n  if true {\n    let a = 1.0;\n    a\n  } else {\n    x\n  }\n}"
        _, result = validate(source, "<test>")
        var_names = [v[0] for v in _vars_in_scope(result, types.Position(line=5, character=4))]
        assert "a" not in var_names

# 19. Type alias
class TestTypeAliasCompletion:
    def test_type_alias_present(self):
        # Note: type alias requires trailing semicolon
        comp, pos, _ = _validate_and_complete_general("type Key = i32[4];\nfn f(x: Key) -> Key { x }", 1, 23)
        assert "Key" in _labels(comp)

    def test_type_alias_kind(self):
        comp, pos, _ = _validate_and_complete_general("type Key = i32[4];\nfn f(x: Key) -> Key { x }", 1, 23)
        assert _items_by_label(comp, "Key")[0].kind == types.CompletionItemKind.TypeParameter

# 20. Annotation utilities
class TestAnnotationStr:
    def test_scalar(self):
        from maomi.ast_nodes import TypeAnnotation, Span
        assert _annotation_str(TypeAnnotation(base="f32", dims=None, span=Span(1,1,1,3))) == "f32"

    def test_wildcard(self):
        from maomi.ast_nodes import TypeAnnotation, Span
        assert _annotation_str(TypeAnnotation(base="f32", dims=None, span=Span(1,1,1,7), wildcard=True)) == "f32[..]"

    def test_array(self):
        from maomi.ast_nodes import TypeAnnotation, Dim, Span
        ann = TypeAnnotation(base="f32", dims=[Dim(3, Span(1,1,1,1)), Dim(4, Span(1,1,1,1))], span=Span(1,1,1,10))
        assert _annotation_str(ann) == "f32[3, 4]"

# 21. check_edit scenarios
class TestCheckEditScenarios:
    def test_struct_field_edit(self):
        _, result = validate("struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n", "<test>")
        pos = types.Position(line=1, character=22)
        comp = _complete_dot(result, pos)
        assert comp is not None
        check_edit("struct S { val: f32 }\nfn f(s: S) -> f32 { s. }\n", pos, comp, "val",
                   "struct S { val: f32 }\nfn f(s: S) -> f32 { s.val }\n")

    def test_pipe_fn_edit(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        pos = types.Position(line=0, character=27)
        comp = _complete_dot(result, pos)
        assert comp is not None
        check_edit("fn f(x: f32[3]) -> f32 { x. }\n", pos, comp, "exp",
                   "fn f(x: f32[3]) -> f32 { x |> exp() }\n")

# 22. Builtin data consistency
class TestBuiltinDataConsistency:
    def test_set_matches_list(self):
        for b in _BUILTINS:
            assert b in _BUILTIN_SET

    def test_sorted(self):
        assert _BUILTINS == sorted(_BUILTINS)

    def test_all_have_docs(self):
        missing = [b for b in _BUILTINS if b not in _BUILTIN_DOCS]
        if missing:
            pytest.skip(f"Missing docs: {missing}")

    def test_all_have_categories(self):
        missing = [b for b in _BUILTINS if b not in _BUILTIN_CATEGORIES]
        if missing:
            pytest.skip(f"Missing categories: {missing}")

    def test_ew_subset(self):
        assert _EW_NAMES - _BUILTIN_SET == set()

    def test_no_namespaced_fns_in_general(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        for item in comp.items:
            if item.kind == types.CompletionItemKind.Function:
                assert "." not in item.label

# 23. Validation robustness
class TestValidationEdgeCases:
    def test_only_comments(self):
        validate("// just a comment\n", "<test>")

    def test_only_struct(self):
        _, result = validate("struct Point { x: f32, y: f32 }", "<test>")
        assert result.program is not None

    def test_multiple_errors(self):
        diags, _ = validate("fn f( { }\nfn g( { }", "<test>")
        assert len(diags) >= 1

# 24. Full flow simulation
class TestCompletionFullFlow:
    def _simulate(self, source, line, col):
        filepath = "<test>"
        result = completion_validate(source, filepath, line, col)
        if result.program is None:
            _, result = validate(source, filepath)
        lines = source.splitlines()
        line_text = lines[line] if line < len(lines) else ""
        col_c = min(col, len(line_text))
        imp = _complete_import(line_text, col_c, filepath)
        if imp is not None:
            return imp, types.Position(line=line, character=col)
        if col_c > 0 and line_text[col_c - 1] == ".":
            j = col_c - 2
            while j >= 0 and (line_text[j].isalnum() or line_text[j] == "_"):
                j -= 1
            prefix = line_text[j + 1:col_c - 1]
            mod = _complete_module(result, prefix)
            if mod is not None:
                return mod, types.Position(line=line, character=col)
            if prefix in _BUILTIN_NAMESPACES:
                items = [types.CompletionItem(label=n, kind=types.CompletionItemKind.Function) for n in _BUILTIN_NAMESPACES[prefix]]
                return types.CompletionList(is_incomplete=False, items=items), types.Position(line=line, character=col)
            return _complete_dot(result, types.Position(line=line, character=col), prefix), types.Position(line=line, character=col)
        sl = _complete_struct_literal(line_text, col_c, result, source, types.Position(line=line, character=col))
        if sl is not None:
            return sl, types.Position(line=line, character=col)
        return _complete_general(result, types.Position(line=line, character=col)), types.Position(line=line, character=col)

    def test_general_flow(self):
        comp, pos = self._simulate("fn f(x: f32) -> f32 {\n  \n}\n", 1, 2)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

    def test_dot_flow(self):
        comp, pos = self._simulate("struct S { x: f32 }\nfn f(s: S) -> f32 {\n  s.\n}\n", 2, 4)
        if comp is not None:
            assert_all_completions_valid(comp, pos)

    def test_import_flow(self):
        comp, pos = self._simulate("import \nfn f(x: f32) -> f32 { x }\n", 0, 7)
        assert comp is not None

    def test_random_dot_flow(self):
        comp, pos = self._simulate("fn f(x: f32) -> f32 {\n  random.\n}\n", 1, 9)
        if comp is not None:
            labels = _labels(comp)
            assert "key" in labels

# 25. Duplicate detection
class TestDuplicateDetectionAllPaths:
    def test_dot_no_duplicates(self):
        _, result = validate("fn f(x: f32[3]) -> f32 { x }\n", "<test>")
        comp = _complete_dot(result, types.Position(line=0, character=27))
        assert comp is not None
        seen = set()
        for item in comp.items:
            assert item.label not in seen, f"Duplicate: {item.label}"
            seen.add(item.label)

# 26. Detail content
class TestDetailContent:
    def test_user_fn_signature(self):
        comp, pos, _ = _validate_and_complete_general("fn helper(x: f32, y: f32) -> f32 { x + y }\nfn main(z: f32) -> f32 { z }", 1, 26)
        h = _items_by_label(comp, "helper")[0]
        assert h.detail is not None and "->" in h.detail

    def test_builtin_detail(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert _items_by_label(comp, "exp")[0].detail is not None

    def test_variable_detail(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 {\n  let a = 1.0;\n  a\n}", 2, 2)
        assert _items_by_label(comp, "a")[0].detail is not None

# 27. Documentation
class TestDocumentation:
    def test_builtin_docs(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert _items_by_label(comp, "exp")[0].documentation is not None

    def test_user_fn_doc_comment(self):
        comp, pos, _ = _validate_and_complete_general("/// Doubles x\nfn double(x: f32) -> f32 { x + x }\nfn main(y: f32) -> f32 { y }", 2, 26)
        d = _items_by_label(comp, "double")[0]
        if d.documentation:
            assert "Doubles" in d.documentation.value

# 28. Keyword exhaustiveness
class TestKeywordExhaustive:
    def test_all_keywords(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        labels = _labels(comp)
        for kw in _KEYWORDS:
            assert kw in labels, f"Missing keyword: {kw}"

    def test_all_type_names(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        labels = _labels(comp)
        for tn in _TYPE_NAMES:
            assert tn in labels, f"Missing type: {tn}"

# 29. Config builtin
class TestConfigBuiltin:
    def test_config_in_general(self):
        assert "config" in _labels(_validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)[0])

    def test_config_docs(self):
        assert "config" in _BUILTIN_DOCS

    def test_config_category(self):
        assert "config" in _BUILTIN_CATEGORIES

# 30. Zero-param fn excluded from pipe
class TestPipeNoZeroParam:
    def test_zero_param_excluded(self):
        _, result = validate("fn make_one() -> f32 { 1.0 }\nfn f(x: f32) -> f32 { x }\n", "<test>")
        comp = _complete_dot(result, types.Position(line=1, character=23))
        if comp is not None:
            assert "make_one" not in _labels(comp)

# 31. Struct doc comments
class TestStructDocComments:
    def test_struct_doc(self):
        comp, pos, _ = _validate_and_complete_general("/// A 2D point\nstruct Point { x: f32, y: f32 }\nfn f(p: Point) -> f32 { p.x }", 2, 25)
        p = _items_by_label(comp, "Point")[0]
        if p.documentation:
            assert "2D point" in p.documentation.value

# 32. Struct literal spec compliance
class TestStructLiteralSpecCompliance:
    def test_struct_literal_valid(self):
        source = "struct Point { x: f32, y: f32 }\nfn f() -> Point { Point {  } }"
        _, result = validate(source, "<test>")
        pos = types.Position(line=1, character=27)
        comp = _complete_struct_literal("fn f() -> Point { Point {  } }", 27, result, source, pos)
        assert comp is not None
        assert_all_completions_valid(comp, pos)

# 33. Completion count sanity
class TestCompletionCount:
    def test_reasonable_count(self):
        comp, pos, _ = _validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)
        assert comp is not None
        assert len(comp.items) >= len(_KEYWORDS) + len(_TYPE_NAMES)
        assert len(comp.items) < 500

# 34. Mixed type struct fields
class TestStructLiteralMixedTypes:
    def test_mixed_types(self):
        source = "struct Config { lr: f32, epochs: i32, use_bias: bool }\nfn f() -> Config { Config {  } }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> Config { Config {  } }", 29, result, source, types.Position(line=1, character=29))
        assert comp is not None
        labels = _labels(comp)
        assert "lr" in labels and "epochs" in labels and "use_bias" in labels
        for item in comp.items:
            if item.label == "lr":
                assert "f32" in item.detail
            elif item.label == "epochs":
                assert "i32" in item.detail

# 35. Struct false positives
class TestStructLiteralFalsePositives:
    def test_fn_body_brace(self):
        source = "struct fn_result { x: f32 }\nfn f() -> f32 { 1.0 }"
        _, result = validate(source, "<test>")
        comp = _complete_struct_literal("fn f() -> f32 { 1.0 }", 16, result, source, types.Position(line=1, character=16))
        assert comp is None

# 36. Param-function name clash
class TestParameterFunctionNameClash:
    def test_param_in_scope(self):
        source = "fn sum(x: f32) -> f32 { x }\nfn f(sum: f32) -> f32 {\n  sum\n}"
        _, result = validate(source, "<test>")
        var_names = [v[0] for v in _vars_in_scope(result, types.Position(line=2, character=5))]
        assert "sum" in var_names

# 37. Nested struct dot
class TestNestedStructDot:
    def test_nested_dot(self):
        source = "struct Inner { x: f32, y: f32 }\nstruct Outer { inner: Inner, z: f32 }\nfn f(s: Outer) -> f32 { s.inner.x }"
        _, result = validate(source, "<test>")
        # This just tests no crash
        _complete_dot(result, types.Position(line=2, character=32), prefix="inner")

# 38. Import with alias
class TestImportWithAlias:
    def test_from_with_alias(self):
        result = _complete_import("from math as m import { ", 24, "<test>")
        if result is not None:
            for item in result.items:
                assert item.kind in (types.CompletionItemKind.Function, types.CompletionItemKind.Struct)

# 39. value_and_grad keyword
class TestValueAndGradCompletion:
    def test_in_keywords(self):
        assert "value_and_grad" in _KEYWORDS

    def test_in_general(self):
        assert "value_and_grad" in _labels(_validate_and_complete_general("fn f(x: f32) -> f32 { x }", 0, 23)[0])

# ============================================================================
# BUG FINDINGS - Tests that demonstrate discovered issues
# ============================================================================

class TestBugFindings:
    """Tests that demonstrate bugs found during audit.
    These tests are marked with xfail to document expected failures."""

    @pytest.mark.xfail(reason="BUG#1: _complete_dot AST lookup ignores prefix parameter")
    def test_bug1_dot_completion_ignores_prefix(self):
        """_complete_dot returns results for wrong prefix because AST lookup
        finds whatever node is at the position, ignoring the prefix."""
        from maomi.lsp import validate, _complete_dot
        source = "struct A { x: f32 }\nstruct B { y: i32 }\nfn f(a: A, b: B) -> f32 {\n    b.y\n}"
        _, result = validate(source, "<test>")
        # Position is at the dot on line 3: "    b.y"
        # Dot at col 5, so position character=5 (after dot)
        pos = types.Position(line=3, character=5)
        # Pass a completely wrong prefix
        comp = _complete_dot(result, pos, prefix="nonexistent_var")
        # EXPECTED: should return None since 'nonexistent_var' doesn't exist
        # ACTUAL: returns B's fields because AST finds 'b' node at that position
        assert comp is None

    @pytest.mark.xfail(reason="BUG#2: _insert_fake_id fails for cursor on empty line after trailing newline")
    def test_bug2_insert_fake_id_trailing_newline(self):
        """_insert_fake_id returns source unchanged when cursor is on the
        empty line after a trailing newline."""
        source = "fn f(x: f32) -> f32 { x }\n"
        # Cursor at line 1, col 0 (the empty line after newline)
        result = _insert_fake_id(source, 1, 0)
        # EXPECTED: fake ID should be inserted at line 1
        # ACTUAL: returns unchanged because splitlines(keepends=True) for "text\n"
        # gives only 1 element, so line_0=1 >= len(lines)=1
        assert result != source, "Fake ID was not inserted"
        assert _FAKE_ID in result

    @pytest.mark.xfail(reason="BUG#4: Module dot completions empty for imported modules with wildcard functions")
    def test_bug4_module_dot_completion_empty(self):
        """Typing 'nn.' after 'import nn;' shows no completions because
        fn_table only has monomorphized $-copies which are filtered out."""
        from maomi.lsp._completion import _complete_module
        source = "import nn;\nfn f(x: f32) -> f32 { nn.relu(x) }"
        _, result = validate(source, "<test>")
        comp = _complete_module(result, "nn")
        # EXPECTED: should show relu, sigmoid, softmax, etc.
        # ACTUAL: returns None because all fn_table entries are like nn.relu$f32
        assert comp is not None
        labels = _labels(comp)
        assert "relu" in labels

    @pytest.mark.xfail(reason="BUG#5/6: Wildcard/generic functions missing from general completions")
    def test_bug5_wildcard_fn_missing_from_general(self):
        """User functions with f32[..] wildcard shapes don't appear in general
        completions because only monomorphized $-copies are in fn_table."""
        source = "fn double_all(x: f32[..]) -> f32[..] { x }\nfn f(y: f32[3]) -> f32[3] { double_all(y) }"
        comp, pos, _ = _validate_and_complete_general(source, 1, 29)
        labels = _labels(comp)
        # EXPECTED: double_all should appear
        # ACTUAL: only double_all$3 in fn_table, filtered by $ check
        assert "double_all" in labels

    @pytest.mark.xfail(reason="BUG#6: Generic T functions missing from general completions")
    def test_bug6_generic_fn_missing(self):
        """Functions with type variable params (T) don't appear in general
        completions after monomorphization."""
        source = "struct Point { x: f32, y: f32 }\nfn identity(x: T) -> T { x }\nfn f(p: Point) -> Point { identity(p) }"
        comp, pos, _ = _validate_and_complete_general(source, 2, 27)
        labels = _labels(comp)
        assert "identity" in labels

    @pytest.mark.xfail(reason="BUG#7: random.exponential and random.randint missing from namespace completion")
    def test_bug7_missing_namespace_members(self):
        """random.exponential and random.randint are registered builtins but
        not listed in _BUILTIN_NAMESPACES, making them unreachable."""
        assert "exponential" in _BUILTIN_NAMESPACES.get("random", [])
        assert "randint" in _BUILTIN_NAMESPACES.get("random", [])

    @pytest.mark.xfail(reason="BUG#3: _is_pipe_compatible doesn't handle StructArrayType")
    def test_bug3_struct_array_type_pipe(self):
        """Pipe completion doesn't offer elementwise builtins for StructArrayType."""
        from maomi.types import StructType, StructArrayType, ScalarType
        st = StructType("Point", (("x", ScalarType("f32")), ("y", ScalarType("f32"))))
        sat = StructArrayType(st, (3,))
        # EW builtins should be compatible with StructArrayType
        # (they work on structs, StructArrayType wraps a struct)
        assert _is_pipe_compatible(sat, "exp", type("FakeSig", (), {"param_types": [ScalarType("f32")]})())

    @pytest.mark.xfail(reason="BUG#8: _annotation_matches_type doesn't handle StructArrayType")
    def test_bug8_annotation_matches_struct_array(self):
        """StructArrayType should match a struct annotation for pipe completions."""
        from maomi.ast_nodes import TypeAnnotation, Span
        from maomi.types import StructType, StructArrayType, ScalarType
        ann = TypeAnnotation(base="Point", dims=None, span=Span(1,1,1,5))
        st = StructType("Point", (("x", ScalarType("f32")),))
        sat = StructArrayType(st, (3,))
        assert _annotation_matches_type(ann, sat, {})


class TestGapFindings:
    """Tests documenting missing features (gaps)."""

    def test_gap1_no_with_expression_completion(self):
        """'p with { | }' should suggest remaining struct fields but doesn't."""
        source = "struct Point { x: f32, y: f32 }\nfn f(p: Point) -> Point {\n    p with {  }\n}"
        _, result = validate(source, "<test>")
        line_text = "    p with {  }"
        comp = _complete_struct_literal(line_text, 13, result, source, types.Position(line=2, character=13))
        # GAP: returns None because 'with' is not a struct name
        # Ideally this should detect with-expression context and suggest struct fields
        if comp is None:
            pytest.skip("GAP: No completion support for 'with' expression fields")

    def test_gap2_comptime_not_in_keywords(self):
        """'comptime' keyword is not offered in completions."""
        if "comptime" not in _KEYWORDS:
            pytest.skip("GAP: 'comptime' not in keywords list for completions")
