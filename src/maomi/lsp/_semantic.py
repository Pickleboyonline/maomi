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
    ImportDecl,
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


def _sem_collect_tokens(node, tokens: list, param_names: set[str]):
    """Recursively collect semantic tokens from AST nodes."""
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
            _sem_collect_tokens(p, tokens, fn_param_names)
        _sem_collect_tokens(node.body, tokens, fn_param_names)
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
        _sem_collect_tokens(node.left, tokens, param_names)
        _sem_collect_tokens(node.right, tokens, param_names)
        # Emit operator token if both operands are on the same line
        if node.left.span.line_end == node.right.span.line_start:
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
        _sem_collect_tokens(node.operand, tokens, param_names)
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
            _sem_collect_tokens(arg, tokens, param_names)
        return

    if isinstance(node, StructLiteral):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            len(node.name),
            _ST_STRUCT,
            0,
        ))
        for _, expr in node.fields:
            _sem_collect_tokens(expr, tokens, param_names)
        return

    if isinstance(node, FieldAccess):
        _sem_collect_tokens(node.object, tokens, param_names)
        tokens.append((
            node.span.line_end - 1,
            node.span.col_end - len(node.field) - 1,
            len(node.field),
            _ST_PROPERTY,
            0,
        ))
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
        _sem_collect_tokens(node.value, tokens, param_names)
        return

    if isinstance(node, IfExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            2,  # "if"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        _sem_collect_tokens(node.condition, tokens, param_names)
        _sem_collect_tokens(node.then_block, tokens, param_names)
        _sem_collect_tokens(node.else_block, tokens, param_names)
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
        _sem_collect_tokens(node.init, tokens, param_names)
        for seq in node.sequences:
            _sem_collect_tokens(seq, tokens, param_names)
        _sem_collect_tokens(node.body, tokens, param_names)
        return

    if isinstance(node, WhileExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            5,  # "while"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        _sem_collect_tokens(node.init, tokens, param_names)
        _sem_collect_tokens(node.cond, tokens, param_names)
        _sem_collect_tokens(node.body, tokens, param_names)
        return

    if isinstance(node, MapExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            3,  # "map"
            _ST_KEYWORD,
            _MOD_CONTROL_FLOW,
        ))
        _sem_collect_tokens(node.sequence, tokens, param_names)
        _sem_collect_tokens(node.body, tokens, param_names)
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
        _sem_collect_tokens(node.expr, tokens, param_names)
        return

    if isinstance(node, CastExpr):
        tokens.append((
            node.span.line_start - 1,
            node.span.col_start - 1,
            4,  # "cast"
            _ST_KEYWORD,
            0,
        ))
        _sem_collect_tokens(node.expr, tokens, param_names)
        return

    for child in _children_of(node):
        _sem_collect_tokens(child, tokens, param_names)


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
    for i, line in enumerate(doc.source.splitlines()):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            col = len(line) - len(stripped)
            tokens.append((i, col, len(stripped), _ST_COMMENT, 0))

    for imp in result.program.imports:
        _sem_collect_import(imp, tokens)

    for ta in result.program.type_aliases:
        _sem_collect_tokens(ta, tokens, set())

    for sd in result.program.struct_defs:
        _sem_collect_tokens(sd, tokens, set())

    for fn in _local_functions(result.program):
        _sem_collect_tokens(fn, tokens, set())

    data = _sem_delta_encode(tokens)
    return types.SemanticTokens(data=data)
