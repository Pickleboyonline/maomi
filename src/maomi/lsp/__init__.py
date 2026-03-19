"""Maomi LSP server package.

Re-exports all public symbols for backward compatibility with existing
imports from ``maomi.lsp``.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Core: server, cache, validation, entry point
# ---------------------------------------------------------------------------
from ._core import (
    server,
    _cache,
    AnalysisResult,
    validate,
    _error_to_diagnostic,
    _local_functions,
    start_server,
)

# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------
from ._ast_utils import (
    _span_contains,
    _children_of,
    _find_node_at,
    _span_to_range,
    classify_symbol,
)

# ---------------------------------------------------------------------------
# Builtin data
# ---------------------------------------------------------------------------
from ._builtin_data import (
    _KEYWORDS,
    _TYPE_NAMES,
    _BUILTINS,
    _BUILTIN_SET,
    _BUILTIN_NAMESPACES,
    _BUILTIN_SIGNATURES,
    _BUILTIN_DOCS,
)

# ---------------------------------------------------------------------------
# Feature modules — re-export test-imported symbols
# ---------------------------------------------------------------------------
from ._hover import _get_hover_text, _fmt_annotation
from ._completion import _complete_general, _complete_dot, _complete_module, _vars_in_scope
from ._goto_def import _goto_find_definition
from ._references import _refs_classify_node, _refs_collect_all
from ._symbols import _build_document_symbols, _workspace_symbols
from ._rename import prepare_rename_at, rename_at, _EditCollector
from ._signature import _sig_parse_call_context, _build_signature_help
from ._inlay_hints import _build_inlay_hints
from ._semantic import (
    _sem_collect_tokens, _sem_delta_encode,
    _ST_FUNCTION, _ST_PARAMETER, _ST_VARIABLE, _ST_STRUCT,
    _ST_PROPERTY, _ST_TYPE, _ST_NUMBER, _ST_KEYWORD,
    _ST_BUILTIN_TYPE, _ST_BUILTIN_FUNCTION, _ST_BOOLEAN, _ST_OPERATOR,
    _MOD_DECLARATION, _MOD_DEFINITION,
)
from ._code_actions import _ca_edit_distance, _ca_find_similar, code_actions
from ._folding import _build_folding_ranges
from ._selection import _sel_collect_ancestors, _sel_build_chain
from ._highlight import _build_document_highlights
from ._type_def import _goto_type_definition
from ._call_hierarchy import (
    _call_hierarchy_prepare,
    _call_hierarchy_incoming,
    _call_hierarchy_outgoing,
)
from ._code_lens import _build_code_lenses
from ._formatting import (
    _on_type_format,
    _compute_brace_depth,
    _find_matching_brace,
    _format_document,
    _format_line_content,
)

# ---------------------------------------------------------------------------
# Import all feature modules to trigger @server.feature registration.
# (Some are already imported above; list remaining ones explicitly.)
# ---------------------------------------------------------------------------
from . import (  # noqa: F401
    _core,
    _hover,
    _completion,
    _goto_def,
    _references,
    _symbols,
    _rename,
    _signature,
    _inlay_hints,
    _semantic,
    _code_actions,
    _folding,
    _selection,
    _highlight,
    _type_def,
    _call_hierarchy,
    _code_lens,
    _formatting,
)
