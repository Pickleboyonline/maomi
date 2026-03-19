from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import FnDef, CallExpr, Identifier
from ._core import server, _cache, AnalysisResult, _local_functions
from ._ast_utils import _find_node_at, _span_to_range, _children_of


def _collect_calls_to(node, target_name: str, spans: list):
    """Recursively walk AST collecting spans of CallExpr nodes that call target_name."""
    if isinstance(node, CallExpr) and node.callee == target_name:
        spans.append(node.span)
    for child in _children_of(node):
        _collect_calls_to(child, target_name, spans)


def _collect_all_calls(node, calls_dict: dict):
    """Recursively walk AST collecting all CallExpr nodes grouped by callee name."""
    if isinstance(node, CallExpr):
        calls_dict.setdefault(node.callee, []).append(node.span)
    for child in _children_of(node):
        _collect_all_calls(child, calls_dict)


def _make_hierarchy_item(fn: FnDef, uri: str) -> types.CallHierarchyItem:
    """Build a CallHierarchyItem for a FnDef."""
    return types.CallHierarchyItem(
        name=fn.name,
        kind=types.SymbolKind.Function,
        uri=uri,
        range=_span_to_range(fn.span),
        selection_range=_span_to_range(fn.span),
    )


def _call_hierarchy_prepare(
    result: AnalysisResult, uri: str, line_0: int, col_0: int,
) -> list[types.CallHierarchyItem] | None:
    """Core logic for textDocument/prepareCallHierarchy."""
    if not result or not result.program:
        return None

    line = line_0 + 1
    col = col_0 + 1
    local_fns = _local_functions(result.program)
    fn_names = {fn.name for fn in local_fns}

    # Find the node at cursor
    target_name = None
    for fn in local_fns:
        node = _find_node_at(fn, line, col)
        if node is not None:
            if isinstance(node, FnDef):
                target_name = node.name
            elif isinstance(node, CallExpr):
                target_name = node.callee
            elif isinstance(node, Identifier) and node.name in fn_names:
                target_name = node.name
            break

    if target_name is None:
        return None

    # Find the FnDef for that name
    for fn in local_fns:
        if fn.name == target_name:
            return [_make_hierarchy_item(fn, uri)]

    return None


def _call_hierarchy_incoming(
    result: AnalysisResult, uri: str, fn_name: str,
) -> list[types.CallHierarchyIncomingCall]:
    """Find all functions that call fn_name."""
    if not result or not result.program:
        return []

    incoming: list[types.CallHierarchyIncomingCall] = []
    for fn in _local_functions(result.program):
        spans: list = []
        _collect_calls_to(fn.body, fn_name, spans)
        if spans:
            incoming.append(types.CallHierarchyIncomingCall(
                from_=_make_hierarchy_item(fn, uri),
                from_ranges=[_span_to_range(s) for s in spans],
            ))

    return incoming


def _call_hierarchy_outgoing(
    result: AnalysisResult, uri: str, fn_name: str,
) -> list[types.CallHierarchyOutgoingCall]:
    """Find all user-defined functions called by fn_name."""
    if not result or not result.program:
        return []

    local_fns = _local_functions(result.program)
    fn_by_name = {fn.name: fn for fn in local_fns}

    target_fn = fn_by_name.get(fn_name)
    if target_fn is None:
        return []

    # Collect all calls from this function's body
    calls_dict: dict[str, list] = {}
    _collect_all_calls(target_fn.body, calls_dict)

    outgoing: list[types.CallHierarchyOutgoingCall] = []
    for callee_name, spans in calls_dict.items():
        callee_fn = fn_by_name.get(callee_name)
        if callee_fn is None:
            continue
        outgoing.append(types.CallHierarchyOutgoingCall(
            to=_make_hierarchy_item(callee_fn, uri),
            from_ranges=[_span_to_range(s) for s in spans],
        ))

    return outgoing


@server.feature(types.TEXT_DOCUMENT_PREPARE_CALL_HIERARCHY)
def prepare_call_hierarchy(ls: LanguageServer, params: types.CallHierarchyPrepareParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    return _call_hierarchy_prepare(result, uri, params.position.line, params.position.character)


@server.feature(types.CALL_HIERARCHY_INCOMING_CALLS)
def incoming_calls(ls: LanguageServer, params: types.CallHierarchyIncomingCallsParams):
    uri = params.item.uri
    fn_name = params.item.name
    result = _cache.get(uri)
    return _call_hierarchy_incoming(result, uri, fn_name)


@server.feature(types.CALL_HIERARCHY_OUTGOING_CALLS)
def outgoing_calls(ls: LanguageServer, params: types.CallHierarchyOutgoingCallsParams):
    uri = params.item.uri
    fn_name = params.item.name
    result = _cache.get(uri)
    return _call_hierarchy_outgoing(result, uri, fn_name)
