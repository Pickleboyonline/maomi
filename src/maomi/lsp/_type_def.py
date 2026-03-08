from __future__ import annotations

from lsprotocol import types

from ..ast_nodes import FnDef, LetStmt, Param, Identifier
from ..types import StructType
from ._core import server, _cache, _local_functions
from ._ast_utils import _find_node_at, _span_to_range


def _goto_type_definition(node, fn, result):
    """Find the struct definition span for the type of the given node."""
    program = result.program
    if program is None:
        return None

    # For Param nodes, check the type annotation directly.
    if isinstance(node, Param):
        base = node.type_annotation.base
        for sdef in program.struct_defs:
            if sdef.name == base:
                return sdef.span
        return None

    # For LetStmt, look up the type of the value expression.
    if isinstance(node, LetStmt):
        typ = result.type_map.get(id(node.value))
        if isinstance(typ, StructType):
            for sdef in program.struct_defs:
                if sdef.name == typ.name:
                    return sdef.span
        return None

    # For Identifier and other expressions, look up type_map.
    node_id = id(node)
    typ = result.type_map.get(node_id)
    if isinstance(typ, StructType):
        for sdef in program.struct_defs:
            if sdef.name == typ.name:
                return sdef.span


@server.feature(types.TEXT_DOCUMENT_TYPE_DEFINITION)
def goto_type_definition(ls, params: types.TypeDefinitionParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return None
    line = params.position.line + 1
    col = params.position.character + 1
    for fn in _local_functions(result.program):
        node = _find_node_at(fn, line, col)
        if node is not None:
            span = _goto_type_definition(node, fn, result)
            if span is not None:
                return types.Location(uri=uri, range=_span_to_range(span))
    return None
