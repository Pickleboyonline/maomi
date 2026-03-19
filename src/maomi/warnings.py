"""Warning analysis pass for Maomi programs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from maomi import ast_nodes as ast


@dataclass
class Warning:
    message: str
    filename: str
    line: int
    col: int
    col_end: int
    hint: str | None = None
    kind: str = "unused_variable"


def analyze(
    program: ast.Program, filename: str, fn_table: dict | None = None
) -> list[Warning]:
    """Run all warning analyses on a program."""
    warnings: list[Warning] = []
    warnings.extend(_check_unused_variables(program, filename))
    warnings.extend(_check_unused_imports(program, filename))
    warnings.extend(_check_unused_functions(program, filename, fn_table))
    return warnings


# ---------------------------------------------------------------------------
# Generic AST walker
# ---------------------------------------------------------------------------

def _walk_expr(expr: object, visitor: Callable[[object], None]) -> None:
    """Walk an AST expression tree, calling visitor on each node.

    The visitor is called before recursing into children. This is the
    single traversal function that all collection helpers build upon.
    """
    if expr is None:
        return

    visitor(expr)

    if isinstance(expr, ast.BinOp):
        _walk_expr(expr.left, visitor)
        _walk_expr(expr.right, visitor)
    elif isinstance(expr, ast.UnaryOp):
        _walk_expr(expr.operand, visitor)
    elif isinstance(expr, ast.CallExpr):
        for arg in expr.args:
            _walk_expr(arg, visitor)
        for _name, arg in expr.named_args:
            _walk_expr(arg, visitor)
    elif isinstance(expr, ast.IfExpr):
        _walk_expr(expr.condition, visitor)
        _walk_block(expr.then_block, visitor)
        _walk_block(expr.else_block, visitor)
    elif isinstance(expr, ast.Block):
        _walk_block(expr, visitor)
    elif isinstance(expr, ast.ScanExpr):
        _walk_expr(expr.init, visitor)
        for seq in expr.sequences:
            _walk_expr(seq, visitor)
        _walk_block(expr.body, visitor)
    elif isinstance(expr, ast.MapExpr):
        _walk_expr(expr.sequence, visitor)
        _walk_block(expr.body, visitor)
    elif isinstance(expr, ast.FoldExpr):
        _walk_expr(expr.init, visitor)
        for seq in expr.sequences:
            _walk_expr(seq, visitor)
        _walk_block(expr.body, visitor)
    elif isinstance(expr, ast.WhileExpr):
        _walk_expr(expr.init, visitor)
        _walk_block(expr.cond, visitor)
        _walk_block(expr.body, visitor)
    elif isinstance(expr, (ast.GradExpr, ast.ValueAndGradExpr)):
        _walk_expr(expr.expr, visitor)
    elif isinstance(expr, ast.CastExpr):
        _walk_expr(expr.expr, visitor)
    elif isinstance(expr, ast.ArrayLiteral):
        for elem in expr.elements:
            _walk_expr(elem, visitor)
    elif isinstance(expr, ast.StructLiteral):
        for _field_name, val in expr.fields:
            _walk_expr(val, visitor)
    elif isinstance(expr, ast.FieldAccess):
        _walk_expr(expr.object, visitor)
    elif isinstance(expr, ast.WithExpr):
        _walk_expr(expr.base, visitor)
        for _path, val in expr.updates:
            _walk_expr(val, visitor)
    elif isinstance(expr, ast.IndexExpr):
        _walk_expr(expr.base, visitor)
        for comp in expr.indices:
            if comp.kind == "single":
                _walk_expr(comp.value, visitor)
            elif comp.kind == "slice":
                _walk_expr(comp.start, visitor)
                _walk_expr(comp.end, visitor)
    elif isinstance(expr, ast.LetStmt):
        _walk_expr(expr.value, visitor)
    elif isinstance(expr, ast.ExprStmt):
        _walk_expr(expr.expr, visitor)
    # Literals, Identifier, ErrorExpr: leaf nodes, no children to recurse into.


def _walk_block(block: ast.Block, visitor: Callable[[object], None]) -> None:
    """Walk all statements and trailing expression in a block."""
    for stmt in block.stmts:
        _walk_expr(stmt, visitor)
    _walk_expr(block.expr, visitor)


# ---------------------------------------------------------------------------
# Reference collection (built on _walk_expr)
# ---------------------------------------------------------------------------

def _make_ref_visitor(refs: set[str]) -> Callable[[object], None]:
    """Create a visitor that collects Identifier and CallExpr name references."""
    def _visitor(node: object) -> None:
        if isinstance(node, ast.Identifier):
            refs.add(node.name)
        elif isinstance(node, ast.CallExpr):
            # Qualified names like "math.relu" register full, prefix, and bare name.
            refs.add(node.callee)
            if "." in node.callee:
                parts = node.callee.split(".")
                refs.add(parts[0])   # module prefix
                refs.add(parts[-1])  # function name
        elif isinstance(node, (ast.GradExpr, ast.ValueAndGradExpr)):
            # wrt can be a string or tuple -- the variable name(s) are references
            if isinstance(node.wrt, str):
                refs.add(node.wrt)
            elif isinstance(node.wrt, tuple):
                refs.add(node.wrt[0])
    return _visitor


def _collect_references_block(block: ast.Block, refs: set[str]) -> None:
    """Collect references from a Block's statements and trailing expression."""
    _walk_block(block, _make_ref_visitor(refs))


def _collect_references_program(program: ast.Program, refs: set[str]) -> None:
    """Collect all references across all function bodies in the program."""
    visitor = _make_ref_visitor(refs)
    for fn in program.functions:
        _walk_block(fn.body, visitor)


# ---------------------------------------------------------------------------
# Binding collection
# ---------------------------------------------------------------------------

@dataclass
class _Binding:
    name: str
    line: int
    col: int
    col_end: int


def _collect_bindings_block(block: ast.Block, bindings: list[_Binding]) -> None:
    """Collect variable bindings from a block's statements and nested expressions."""
    for stmt in block.stmts:
        if isinstance(stmt, ast.LetStmt):
            span = stmt.span
            bindings.append(_Binding(stmt.name, span.line_start, span.col_start, span.col_end))
        elif isinstance(stmt, ast.ExprStmt):
            _collect_bindings_expr(stmt.expr, bindings)
    if block.expr is not None:
        _collect_bindings_expr(block.expr, bindings)


def _collect_bindings_expr(expr: object, bindings: list[_Binding]) -> None:
    """Collect bindings introduced inside expressions (scan, map, fold, while, if)."""
    if expr is None:
        return

    if isinstance(expr, ast.ScanExpr):
        span = expr.span
        bindings.append(_Binding(expr.carry_var, span.line_start, span.col_start, span.col_end))
        for ev in expr.elem_vars:
            bindings.append(_Binding(ev, span.line_start, span.col_start, span.col_end))
        _collect_bindings_block(expr.body, bindings)

    elif isinstance(expr, ast.MapExpr):
        span = expr.span
        bindings.append(_Binding(expr.elem_var, span.line_start, span.col_start, span.col_end))
        _collect_bindings_block(expr.body, bindings)

    elif isinstance(expr, ast.FoldExpr):
        span = expr.span
        bindings.append(_Binding(expr.carry_var, span.line_start, span.col_start, span.col_end))
        for ev in expr.elem_vars:
            bindings.append(_Binding(ev, span.line_start, span.col_start, span.col_end))
        _collect_bindings_block(expr.body, bindings)

    elif isinstance(expr, ast.WhileExpr):
        span = expr.span
        bindings.append(_Binding(expr.state_var, span.line_start, span.col_start, span.col_end))
        _collect_bindings_block(expr.cond, bindings)
        _collect_bindings_block(expr.body, bindings)

    elif isinstance(expr, ast.IfExpr):
        _collect_bindings_block(expr.then_block, bindings)
        _collect_bindings_block(expr.else_block, bindings)

    elif isinstance(expr, ast.Block):
        _collect_bindings_block(expr, bindings)


# ---------------------------------------------------------------------------
# Unused variable detection
# ---------------------------------------------------------------------------

def _check_unused_variables(program: ast.Program, filename: str) -> list[Warning]:
    """Detect unused let bindings and loop variables.

    Note: function parameters are intentionally excluded. Unused params are
    common in interface-conforming functions and test code, and are too noisy.
    """
    warnings: list[Warning] = []

    for fn in program.functions:
        # Skip imported/prefixed functions
        if "." in fn.name or fn.source_file is not None:
            continue

        # Collect all references in the function body
        refs: set[str] = set()
        _collect_references_block(fn.body, refs)

        # Collect bindings from let stmts and loop vars (NOT parameters)
        bindings: list[_Binding] = []
        _collect_bindings_block(fn.body, bindings)

        # Check each binding
        seen: set[str] = set()
        for binding in bindings:
            name = binding.name
            # Skip underscore-prefixed, duplicates, and compiler-generated names
            if name.startswith("_") or name in seen:
                continue
            seen.add(name)
            if name not in refs:
                warnings.append(Warning(
                    message=f"Unused variable '{name}'",
                    filename=filename,
                    line=binding.line,
                    col=binding.col,
                    col_end=binding.col_end,
                    hint=f"If this is intentional, prefix with underscore: _{name}",
                    kind="unused_variable",
                ))

    return warnings


# ---------------------------------------------------------------------------
# Unused import detection
# ---------------------------------------------------------------------------

def _check_unused_imports(program: ast.Program, filename: str) -> list[Warning]:
    """Detect unused selective imports (from X import { name })."""
    warnings: list[Warning] = []

    # Collect all references across the program
    refs: set[str] = set()
    _collect_references_program(program, refs)

    for imp in program.imports:
        # Only check selective imports (ones with .names)
        if imp.names is None:
            continue
        for i, name in enumerate(imp.names):
            if name not in refs:
                # Use per-name span if available, else fall back to import span
                if i < len(imp.name_spans):
                    span = imp.name_spans[i]
                else:
                    span = imp.span
                warnings.append(Warning(
                    message=f"Unused import '{name}'",
                    filename=filename,
                    line=span.line_start,
                    col=span.col_start,
                    col_end=span.col_end,
                    hint="Remove this import or use the imported name",
                    kind="unused_import",
                ))

    return warnings


# ---------------------------------------------------------------------------
# Unused function detection
# ---------------------------------------------------------------------------

def _check_unused_functions(
    program: ast.Program, filename: str, fn_table: dict | None = None
) -> list[Warning]:
    """Detect unused local functions that are never called."""
    warnings: list[Warning] = []

    # Only check files with multiple local functions -- a single function
    # is typically the entry point or the sole purpose of the file.
    local_fns = [fn for fn in program.functions
                 if "." not in fn.name and fn.source_file is None]
    if len(local_fns) < 2:
        return warnings

    # Collect all call targets across the program
    call_targets: set[str] = set()
    for fn in program.functions:
        _collect_call_targets_block(fn.body, call_targets)

    for fn in local_fns:
        name = fn.name
        # Skip underscore-prefixed
        if name.startswith("_"):
            continue
        # Skip if called anywhere
        if name in call_targets:
            continue
        # Only warn if the function has at least one symbolic dimension
        # (concrete-shape functions may be entry points)
        if not _has_symbolic_dim(fn):
            continue

        warnings.append(Warning(
            message=f"Unused function '{name}'",
            filename=filename,
            line=fn.span.line_start,
            col=fn.span.col_start,
            col_end=fn.span.col_end,
            hint="This function is never called. Prefix with _ if intentional.",
            kind="unused_function",
        ))

    return warnings


def _has_symbolic_dim(fn: ast.FnDef) -> bool:
    """Check if a function has at least one symbolic (non-concrete) dimension."""
    for param in fn.params:
        ann = param.type_annotation
        if ann.dims is not None:
            for dim in ann.dims:
                if isinstance(dim.value, str):
                    return True
        if ann.wildcard:
            return True
    return False


def _collect_call_targets_block(block: ast.Block, targets: set[str]) -> None:
    """Collect all call targets from a block using the generic walker."""
    def _visitor(node: object) -> None:
        if isinstance(node, ast.CallExpr):
            targets.add(node.callee)
            if "." in node.callee:
                targets.add(node.callee.split(".")[-1])
    _walk_block(block, _visitor)
