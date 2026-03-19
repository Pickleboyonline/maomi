from __future__ import annotations

import re

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import (
    Program, Block, LetStmt, ExprStmt, ScanExpr, MapExpr, IfExpr,
)
from ._core import server, _cache
from ._builtin_data import _BUILTINS


def _ca_edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _ca_edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def _ca_find_similar(name: str, candidates, max_distance: int | None = None) -> list[str]:
    """Find candidate names within edit distance of the given name.

    Uses proportional edit distance by default: max(len(name) // 3, 2).
    Short names get max_distance=2 (catches transpositions like 'epx' -> 'exp'),
    while long names scale up (e.g. 15-char name allows 5 edits).
    Falls back to case-insensitive exact matching if no edit-distance matches found.
    """
    if max_distance is None:
        max_distance = max(len(name) // 3, 2)
    results = []
    for c in candidates:
        if c == name:
            continue
        dist = _ca_edit_distance(name, c)
        if dist <= max_distance:
            results.append((dist, c))
    # Case-insensitive fallback when no edit-distance matches found
    if not results:
        name_lower = name.lower()
        for candidate in candidates:
            if candidate.lower() == name_lower and candidate != name:
                results.append((0, candidate))
    results.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    return [c for _, c in results[:5]]


_CA_UNKNOWN_PATTERN = re.compile(r"['\u2018](\w+)['\u2019]")


def _ca_collect_var_names(program: Program | None) -> set[str]:
    """Collect all variable names from the program."""
    names: set[str] = set()
    if not program:
        return names
    for fn in program.functions:
        for p in fn.params:
            names.add(p.name)
        _ca_collect_block_vars(fn.body, names)
    return names


def _ca_collect_block_vars(block: Block, names: set[str]):
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            names.add(stmt.name)
        elif isinstance(stmt, ExprStmt):
            _ca_collect_expr_vars(stmt.expr, names)
    if block.expr is not None:
        _ca_collect_expr_vars(block.expr, names)


def _ca_collect_expr_vars(expr, names: set[str]):
    if isinstance(expr, ScanExpr):
        names.add(expr.carry_var)
        for ev in expr.elem_vars:
            names.add(ev)
        _ca_collect_block_vars(expr.body, names)
    elif isinstance(expr, MapExpr):
        names.add(expr.elem_var)
        _ca_collect_block_vars(expr.body, names)
    elif isinstance(expr, IfExpr):
        _ca_collect_block_vars(expr.then_block, names)
        _ca_collect_block_vars(expr.else_block, names)


@server.feature(types.TEXT_DOCUMENT_CODE_ACTION)
def code_actions(ls: LanguageServer, params: types.CodeActionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result:
        return None

    actions: list[types.CodeAction] = []

    def _all_candidates() -> set[str]:
        c: set[str] = set()
        c.update(_ca_collect_var_names(result.program))
        if result.fn_table:
            c.update(k for k in result.fn_table if "." not in k)
        c.update(_BUILTINS)
        if result.struct_defs:
            c.update(result.struct_defs.keys())
        return c

    for diag in params.context.diagnostics:
        msg = diag.message.lower()

        candidates: set[str] = set()

        if "function" in msg:
            if result.fn_table:
                candidates.update(k for k in result.fn_table if "." not in k)
            candidates.update(_BUILTINS)
        elif "struct" in msg:
            if result.struct_defs:
                candidates.update(result.struct_defs.keys())
        elif "variable" in msg or "undefined" in msg or "unknown" in msg:
            candidates = _all_candidates()

        if not candidates:
            candidates = _all_candidates()

        match = _CA_UNKNOWN_PATTERN.search(diag.message)
        if not match:
            continue
        unknown_name = match.group(1)

        suggestions = _ca_find_similar(unknown_name, candidates)

        for suggestion in suggestions:
            actions.append(types.CodeAction(
                title=f"Did you mean '{suggestion}'?",
                kind=types.CodeActionKind.QuickFix,
                diagnostics=[diag],
                edit=types.WorkspaceEdit(
                    changes={uri: [
                        types.TextEdit(
                            range=diag.range,
                            new_text=suggestion,
                        )
                    ]}
                ),
            ))

    return actions if actions else None
