from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ..ast_nodes import (
    FnDef, Block, LetStmt, ExprStmt, Param,
    BinOp, UnaryOp,
    CallExpr, ScanExpr, MapExpr, FoldExpr, WhileExpr, IfExpr,
    GradExpr, ValueAndGradExpr, CastExpr,
    Identifier, IntLiteral, FloatLiteral, BoolLiteral, StringLiteral,
    StructLiteral, FieldAccess, StructDef, TypeAlias, TypeAnnotation,
    ImportDecl, WithExpr,
)
from ._core import server, _cache, _local_functions
from ._ast_utils import _children_of
from ._builtin_data import _BUILTIN_SET, _TYPE_NAMES


_SEM_TOKEN_TYPES = [
    "function",        # 0
    "parameter",       # 1
    "variable",        # 2
    "struct",          # 3
    "property",        # 4
    "type",            # 5
    "number",          # 6
    "keyword",         # 7
    "string",          # 8
    "namespace",       # 9
    "operator",        # 10
    "builtinFunction", # 11
    "builtinType",     # 12
    "boolean",         # 13
    "typeAlias",       # 14
    "comment",         # 15
]

_SEM_TOKEN_MODIFIERS = ["declaration", "definition", "controlFlow", "builtin", "readonly"]

_SEM_LEGEND = types.SemanticTokensLegend(
    token_types=_SEM_TOKEN_TYPES,
    token_modifiers=_SEM_TOKEN_MODIFIERS,
)

_ST_FUNCTION = 0
_ST_PARAMETER = 1
_ST_VARIABLE = 2
_ST_STRUCT = 3
_ST_PROPERTY = 4
_ST_TYPE = 5
_ST_NUMBER = 6
_ST_KEYWORD = 7
_ST_STRING = 8
_ST_NAMESPACE = 9
_ST_OPERATOR = 10
_ST_BUILTIN_FUNCTION = 11
_ST_BUILTIN_TYPE = 12
_ST_BOOLEAN = 13
_ST_TYPE_ALIAS = 14
_ST_COMMENT = 15

_MOD_DECLARATION = 1
_MOD_DEFINITION = 2
_MOD_CONTROL_FLOW = 4
_MOD_BUILTIN = 8
_MOD_READONLY = 16

_PRIMITIVE_TYPES = set(_TYPE_NAMES)


def _sem_collect_import(imp: ImportDecl, tokens: list):
    """Emit semantic tokens for an import declaration."""
    line = imp.span.line_start - 1
    col = imp.span.col_start - 1

    if imp.names is not None:
        # from ... import { ... };
        # "from" keyword
        tokens.append((line, col, 4, _ST_KEYWORD, 0))
    else:
        # import ...;
        # "import" keyword
        tokens.append((line, col, 6, _ST_KEYWORD, 0))

    # Module name/path
    if imp.module_span:
        ms = imp.module_span
        # Path-based imports (contain /) are string literals — highlight as string
        token_type = _ST_STRING if "/" in imp.module_path or imp.module_path.startswith(".") else _ST_NAMESPACE
        tokens.append((ms.line_start - 1, ms.col_start - 1, ms.col_end - ms.col_start, token_type, 0))

    # Alias name (after "as")
    if imp.alias_span:
        tokens.append((imp.alias_span.line_start - 1, imp.alias_span.col_start - 1,
                        imp.alias_span.col_end - imp.alias_span.col_start, _ST_NAMESPACE, _MOD_DECLARATION))

    # Imported names in { ... }
    for ns in imp.name_spans:
        tokens.append((ns.line_start - 1, ns.col_start - 1, ns.col_end - ns.col_start, _ST_FUNCTION, 0))


def _find_keyword_in_source(
    source_lines: list[str] | None, keyword: str,
    line_start: int, col_start: int, line_end: int, col_end: int,
) -> tuple[int, int] | None:
    """Search source text for a keyword within a span region (1-indexed).

    Returns (line_1indexed, col_1indexed) or None if not found.
    """
    if source_lines is None:
        return None
    for line_num in range(line_start, line_end + 1):
        line_idx = line_num - 1
        if line_idx < 0 or line_idx >= len(source_lines):
            continue
        line_text = source_lines[line_idx]
        search_start = (col_start - 1) if line_num == line_start else 0
        search_end = (col_end - 1) if line_num == line_end else len(line_text)
        idx = line_text.find(keyword, search_start, search_end)
        if idx >= 0:
            # Make sure it's a whole word (not part of a longer identifier)
            # Check character before
            if idx > 0 and (line_text[idx - 1].isalnum() or line_text[idx - 1] == '_'):
                continue
            # Check character after
            end_idx = idx + len(keyword)
            if end_idx < len(line_text) and (line_text[end_idx].isalnum() or line_text[end_idx] == '_'):
                continue
            return (line_num, idx + 1)
    return None


def _sem_collect_tokens(node, tokens: list, param_names: set[str],
                        source_lines: list[str] | None = None):
    """Recursively collect semantic tokens from AST nodes."""
    _recurse = lambda n: _sem_collect_tokens(n, tokens, param_names, source_lines)

    if isinstance(node, FnDef):
        # "fn" keyword
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            2,  # len("fn")
            _ST_KEYWORD,
            0,
        ))
        # function name
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 3,
            len(node.name),
            _ST_FUNCTION,
            _MOD_DECLARATION | _MOD_DEFINITION,
        ))
        # return type annotation
        _sem_add_type_annotation(node.return_type, tokens)
        fn_param_names = {p.name for p in node.params}
        for p in node.params:
            _sem_collect_tokens(p, tokens, fn_param_names, source_lines)
        _sem_collect_tokens(node.body, tokens, fn_param_names, source_lines)
        return

    if isinstance(node, Param):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            _ST_PARAMETER,
            _MOD_DECLARATION | _MOD_READONLY,
        ))
        _sem_add_type_annotation(node.type_annotation, tokens)
        return

    if isinstance(node, StructDef):
        # "struct" keyword
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            6,  # len("struct")
            _ST_KEYWORD,
            0,
        ))
        # struct name
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 7,
            len(node.name),
            _ST_STRUCT,
            _MOD_DECLARATION | _MOD_DEFINITION,
        ))
        for i, (field_name, type_ann) in enumerate(node.fields):
            # field name as property — use stored span if available
            if i < len(node.field_name_spans):
                fs = node.field_name_spans[i]
                tokens.append((
                    fs.line_start - 1,
                    fs.col_start - 1,
                    fs.col_end - fs.col_start,
                    _ST_PROPERTY,
                    _MOD_DECLARATION,
                ))
            else:
                # fallback for imported structs without field_name_spans
                tokens.append((
                    type_ann.span.line_start - 1,
                    type_ann.span.col_start - 1 - len(field_name) - 2,
                    len(field_name),
                    _ST_PROPERTY,
                    _MOD_DECLARATION,
                ))
            _sem_add_type_annotation(type_ann, tokens)
        return

    if isinstance(node, TypeAlias):
        # "type" keyword + alias name
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            4,  # len("type")
            _ST_KEYWORD,
            0,
        ))
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 5,
            len(node.name),
            _ST_TYPE_ALIAS,
            _MOD_DECLARATION | _MOD_DEFINITION,
        ))
        _sem_add_type_annotation(node.type_annotation, tokens)
        return

    if isinstance(node, Identifier):
        token_type = _ST_PARAMETER if node.name in param_names else _ST_VARIABLE
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            token_type,
            0,
        ))
        return

    if isinstance(node, BinOp):
        _recurse(node.left)
        _recurse(node.right)
        # Emit operator token — search source text for exact position
        if source_lines is not None and node.left.span.line_end == node.right.span.line_start:
            line_idx = node.left.span.line_end - 1  # 0-indexed
            if 0 <= line_idx < len(source_lines):
                line_text = source_lines[line_idx]
                search_start = node.left.span.col_end - 1  # 0-indexed
                search_end = node.right.span.col_start - 1  # 0-indexed
                op_str = node.op
                op_idx = line_text.find(op_str, search_start, search_end)
                if op_idx >= 0:
                    tokens.append((line_idx, op_idx, len(op_str), _ST_OPERATOR, 0))
        elif source_lines is None and node.left.span.line_end == node.right.span.line_start:
            # Fallback heuristic when no source lines available
            op_len = len(node.op)
            op_col = node.right.span.col_start - op_len - 1
            if op_col >= node.left.span.col_end:
                tokens.append((
                    node.right.span.line_start - 1,
                    op_col - 1,
                    op_len,
                    _ST_OPERATOR,
                    0,
                ))
        return

    if isinstance(node, UnaryOp):
        tt = _ST_KEYWORD if node.op == "not" else _ST_OPERATOR
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.op),
            tt,
            0,
        ))
        _recurse(node.operand)
        return

    if isinstance(node, CallExpr):
        call_type = _ST_BUILTIN_FUNCTION if node.callee in _BUILTIN_SET else _ST_FUNCTION
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.callee),
            call_type,
            0,
        ))
        for arg in node.args:
            _recurse(arg)
        return

    if isinstance(node, StructLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            _ST_STRUCT,
            0,
        ))
        # G14: Emit property tokens for field names in struct literal
        if source_lines is not None:
            # Search source text for each field name position
            search_line_start = node.span.line_start
            search_col_start = node.span.col_start + len(node.name)
            for field_name, expr in node.fields:
                pos = _find_keyword_in_source(
                    source_lines, field_name,
                    search_line_start, search_col_start,
                    expr.span.line_start, expr.span.col_start,
                )
                if pos is not None:
                    tokens.append((pos[0] - 1, pos[1] - 1, len(field_name), _ST_PROPERTY, 0))
                    # Advance search start past this field's value for the next field
                    search_line_start = expr.span.line_end
                    search_col_start = expr.span.col_end
                _recurse(expr)
        else:
            for _, expr in node.fields:
                _recurse(expr)
        return

    if isinstance(node, FieldAccess):
        _recurse(node.object)
        tokens.append((
            node.span.line_end - 1,
            node.span.col_end - len(node.field) - 1,
            len(node.field),
            _ST_PROPERTY,
            0,
        ))
        return

    if isinstance(node, WithExpr):
        _recurse(node.base)
        # Search for "with" keyword between base expr and first update
        pos = _find_keyword_in_source(
            source_lines, "with",
            node.base.span.line_end, node.base.span.col_end,
            node.span.line_end, node.span.col_end,
        )
        if pos is not None:
            tokens.append((pos[0] - 1, pos[1] - 1, 4, _ST_KEYWORD, 0))
        for _, expr in node.updates:
            _recurse(expr)
        return

    if isinstance(node, (IntLiteral, FloatLiteral)):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            node.span.col_end - node.span.col_start,
            _ST_NUMBER,
            0,
        ))
        return

    if isinstance(node, StringLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            node.span.col_end - node.span.col_start,
            _ST_STRING,
            0,
        ))
        return

    if isinstance(node, BoolLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            node.span.col_end - node.span.col_start,
            _ST_BOOLEAN,
            0,
        ))
        return

    if isinstance(node, LetStmt):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            3,  # "let"
            _ST_KEYWORD,
            0,
        ))
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1 + 4,
            len(node.name),
            _ST_VARIABLE,
            _MOD_DECLARATION | _MOD_READONLY,
        ))
        _recurse(node.value)
        return

    if isinstance(node, IfExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            2,  # "if"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        _recurse(node.condition)
        _recurse(node.then_block)
        # G13: "else" keyword between then_block and else_block
        pos = _find_keyword_in_source(
            source_lines, "else",
            node.then_block.span.line_end, node.then_block.span.col_end,
            node.else_block.span.line_start, node.else_block.span.col_start,
        )
        if pos is not None:
            tokens.append((pos[0] - 1, pos[1] - 1, 4, _ST_KEYWORD, _MOD_CONTROL_FLOW))
        _recurse(node.else_block)
        return

    if isinstance(node, (ScanExpr, FoldExpr)):
        kw = "scan" if isinstance(node, ScanExpr) else "fold"
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(kw),
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        # G13: "in" keyword between variables and init expression
        # The "in" is between the closing ) of vars and the ( of init/sequences
        pos = _find_keyword_in_source(
            source_lines, "in",
            node.span.line_start, node.span.col_start + len(kw),
            node.init.span.line_start, node.init.span.col_start,
        )
        if pos is not None:
            tokens.append((pos[0] - 1, pos[1] - 1, 2, _ST_KEYWORD, _MOD_CONTROL_FLOW))
        _recurse(node.init)
        for seq in node.sequences:
            _recurse(seq)
        _recurse(node.body)
        return

    if isinstance(node, WhileExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            5,  # "while"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        _recurse(node.init)
        # G13: "limit" keyword if max_iters is present
        if node.max_iters is not None:
            pos = _find_keyword_in_source(
                source_lines, "limit",
                node.span.line_start, node.span.col_start,
                node.cond.span.line_start, node.cond.span.col_start,
            )
            if pos is not None:
                tokens.append((pos[0] - 1, pos[1] - 1, 5, _ST_KEYWORD, _MOD_CONTROL_FLOW))
        # G13: "do" keyword between cond and body
        pos = _find_keyword_in_source(
            source_lines, "do",
            node.cond.span.line_end, node.cond.span.col_end,
            node.body.span.line_start, node.body.span.col_start,
        )
        if pos is not None:
            tokens.append((pos[0] - 1, pos[1] - 1, 2, _ST_KEYWORD, _MOD_CONTROL_FLOW))
        _recurse(node.cond)
        _recurse(node.body)
        return

    if isinstance(node, MapExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            3,  # "map"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        # G13: "in" keyword between elem_var and sequence
        pos = _find_keyword_in_source(
            source_lines, "in",
            node.span.line_start, node.span.col_start + 3,
            node.sequence.span.line_start, node.sequence.span.col_start,
        )
        if pos is not None:
            tokens.append((pos[0] - 1, pos[1] - 1, 2, _ST_KEYWORD, _MOD_CONTROL_FLOW))
        _recurse(node.sequence)
        _recurse(node.body)
        return

    if isinstance(node, (GradExpr, ValueAndGradExpr)):
        kw = "grad" if isinstance(node, GradExpr) else "value_and_grad"
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(kw),
            _ST_KEYWORD,
            0,
        ))
        _recurse(node.expr)
        return

    if isinstance(node, CastExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            4,  # "cast"
            _ST_KEYWORD,
            0,
        ))
        _recurse(node.expr)
        return

    for child in _children_of(node):
        _sem_collect_tokens(child, tokens, param_names, source_lines)


def _sem_add_type_annotation(ta: TypeAnnotation, tokens: list):
    """Add semantic token for a type annotation."""
    token_type = _ST_BUILTIN_TYPE if ta.base in _PRIMITIVE_TYPES else _ST_TYPE
    tokens.append((
        ta.span.line_start - 1,
        ta.span.col_start - 1,
        len(ta.base),
        token_type,
        0,
    ))


def _sem_delta_encode(tokens: list[tuple]) -> list[int]:
    """Delta-encode sorted tokens into LSP integer array."""
    tokens.sort(key=lambda t: (t[0], t[1]))
    data = []
    prev_line = 0
    prev_start = 0
    for line, col, length, token_type, modifiers in tokens:
        delta_line = line - prev_line
        delta_start = col - prev_start if delta_line == 0 else col
        data.extend([delta_line, delta_start, length, token_type, modifiers])
        prev_line = line
        prev_start = col
    return data


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    types.SemanticTokensRegistrationOptions(legend=_SEM_LEGEND, full=True),
)
def semantic_tokens_full(ls: LanguageServer, params: types.SemanticTokensParams):
    uri = params.text_document.uri
    result = _cache.get(uri)
    if not result or not result.program:
        return types.SemanticTokens(data=[])

    tokens: list[tuple] = []

    # Comments (doc and regular) are consumed by the parser and don't appear
    # in the AST. Scan the source directly to emit comment tokens.
    doc = ls.workspace.get_text_document(uri)
    source_lines = doc.source.splitlines()
    for i, line in enumerate(source_lines):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            col = len(line) - len(stripped)
            tokens.append((i, col, len(stripped), _ST_COMMENT, 0))

    for imp in result.program.imports:
        _sem_collect_import(imp, tokens)

    for ta in result.program.type_aliases:
        _sem_collect_tokens(ta, tokens, set(), source_lines)

    for sd in result.program.struct_defs:
        _sem_collect_tokens(sd, tokens, set(), source_lines)

    for fn in _local_functions(result.program):
        _sem_collect_tokens(fn, tokens, set(), source_lines)

    data = _sem_delta_encode(tokens)
    return types.SemanticTokens(data=data)
