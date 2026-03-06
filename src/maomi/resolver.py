from __future__ import annotations

import os
from pathlib import Path

from .ast_nodes import (
    Program,
    ImportDecl,
    FnDef,
    Block,
    LetStmt,
    ExprStmt,
    CallExpr,
    ScanExpr,
    MapExpr,
    IfExpr,
    GradExpr,
    UnaryOp,
    BinOp,
    IndexExpr,
    IndexComponent,
    Expr,
)
from .lexer import Lexer
from .parser import Parser
from .errors import MaomiError


def resolve(program: Program, file_path: str) -> Program:
    """Resolve all imports and produce a merged Program with a flat function list."""
    ctx = _ResolveContext()
    return ctx.resolve(program, file_path)


class _ResolveContext:
    def __init__(self):
        self._cache: dict[str, list[FnDef]] = {}  # abs path -> prefixed FnDefs
        self._resolving: set[str] = set()          # cycle detection

    def resolve(self, program: Program, file_path: str) -> Program:
        abs_path = os.path.abspath(file_path)
        merged_fns: list[FnDef] = []

        for imp in program.imports:
            mod_path = self._resolve_file(imp, abs_path)
            module_name = imp.alias if imp.alias else _stem(imp.module_path)

            # Get prefixed functions for this module (cached)
            prefixed_fns = self._load_module(mod_path, module_name)

            if imp.names is not None:
                # from math import { relu, linear };
                # Include ALL module fns (needed for internal calls), but also
                # add alias FnDefs for the selected names
                merged_fns.extend(prefixed_fns)
                available = {fn.name.split(".", 1)[1]: fn for fn in prefixed_fns
                             if "." in fn.name}
                for name in imp.names:
                    if name not in available:
                        avail_list = sorted(available.keys())
                        raise MaomiError(
                            f"module '{module_name}' has no function '{name}'. "
                            f"Available: {avail_list}",
                            file_path,
                            imp.span.line_start,
                            imp.span.col_start,
                        )
                    # Check no collision with main file functions
                    for fn in program.functions:
                        if fn.name == name:
                            raise MaomiError(
                                f"imported name '{name}' conflicts with local function '{name}'",
                                file_path,
                                imp.span.line_start,
                                imp.span.col_start,
                            )
                    # Create an alias: a copy of the FnDef with the unqualified name
                    orig = available[name]
                    alias_fn = FnDef(name, orig.params, orig.return_type,
                                     _rewrite_calls_in_block(orig.body, {name: orig.name}),
                                     orig.span)
                    merged_fns.append(alias_fn)
            else:
                # import math; — qualified access only
                merged_fns.extend(prefixed_fns)

        # Main file's own functions come last (can call imported ones)
        merged_fns.extend(program.functions)
        return Program([], program.struct_defs, merged_fns, program.span)

    def _load_module(self, mod_path: str, module_name: str) -> list[FnDef]:
        cache_key = f"{mod_path}::{module_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if mod_path in self._resolving:
            raise MaomiError(
                f"circular import detected: {mod_path}",
                mod_path, 0, 0,
            )

        self._resolving.add(mod_path)
        try:
            with open(mod_path) as f:
                source = f.read()
        except FileNotFoundError:
            raise MaomiError(
                f"imported module not found: {mod_path}",
                mod_path, 0, 0,
            )

        tokens = Lexer(source, filename=mod_path).tokenize()
        mod_program = Parser(tokens, filename=mod_path).parse()

        # Recursively resolve the module's own imports
        mod_program = self.resolve(mod_program, mod_path)

        # Prefix all functions with module_name
        local_names = {fn.name for fn in mod_program.functions}
        rename_map = {name: f"{module_name}.{name}" for name in local_names}
        prefixed_fns = [_prefix_fn(fn, module_name, rename_map) for fn in mod_program.functions]

        self._resolving.discard(mod_path)
        self._cache[cache_key] = prefixed_fns
        return prefixed_fns

    def _resolve_file(self, imp: ImportDecl, importing_file: str) -> str:
        base_dir = os.path.dirname(importing_file)
        path = imp.module_path

        # If it looks like a relative/absolute path
        if "/" in path or path.startswith("."):
            candidate = os.path.join(base_dir, path)
            if not candidate.endswith(".mao"):
                candidate += ".mao"
            return os.path.abspath(candidate)

        # Simple module name: look for <name>.mao relative to importing file
        candidate = os.path.join(base_dir, f"{path}.mao")
        return os.path.abspath(candidate)


def _stem(module_path: str) -> str:
    """Derive module name from path: '../lib/nn.mao' -> 'nn', 'math' -> 'math'."""
    base = os.path.basename(module_path)
    if base.endswith(".mao"):
        base = base[:-4]
    return base


def _prefix_fn(fn: FnDef, module_name: str, rename_map: dict[str, str]) -> FnDef:
    """Rename a function and rewrite its internal calls."""
    new_name = rename_map.get(fn.name, fn.name)
    new_body = _rewrite_calls_in_block(fn.body, rename_map)
    return FnDef(new_name, fn.params, fn.return_type, new_body, fn.span)


def _rewrite_calls_in_block(block: Block, rename_map: dict[str, str]) -> Block:
    new_stmts = []
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            new_stmts.append(LetStmt(stmt.name, stmt.type_annotation,
                                     _rewrite_expr(stmt.value, rename_map), stmt.span))
        elif isinstance(stmt, ExprStmt):
            new_stmts.append(ExprStmt(_rewrite_expr(stmt.expr, rename_map), stmt.span))
    new_expr = _rewrite_expr(block.expr, rename_map) if block.expr is not None else None
    return Block(new_stmts, new_expr, block.span)


def _rewrite_expr(expr: Expr, rename_map: dict[str, str]) -> Expr:
    match expr:
        case CallExpr(callee=callee, args=args, span=span):
            new_callee = rename_map.get(callee, callee)
            new_args = [_rewrite_expr(a, rename_map) for a in args]
            return CallExpr(new_callee, new_args, span)
        case BinOp(op=op, left=left, right=right, span=span):
            return BinOp(op, _rewrite_expr(left, rename_map),
                         _rewrite_expr(right, rename_map), span)
        case UnaryOp(op=op, operand=operand, span=span):
            return UnaryOp(op, _rewrite_expr(operand, rename_map), span)
        case IfExpr(condition=cond, then_block=then_b, else_block=else_b, span=span):
            return IfExpr(_rewrite_expr(cond, rename_map),
                          _rewrite_calls_in_block(then_b, rename_map),
                          _rewrite_calls_in_block(else_b, rename_map), span)
        case ScanExpr():
            new_init = _rewrite_expr(expr.init, rename_map)
            new_seqs = [_rewrite_expr(s, rename_map) for s in expr.sequences]
            new_body = _rewrite_calls_in_block(expr.body, rename_map)
            return ScanExpr(expr.carry_var, expr.elem_vars, new_init, new_seqs,
                            new_body, expr.span, expr.reverse)
        case MapExpr():
            new_seq = _rewrite_expr(expr.sequence, rename_map)
            new_body = _rewrite_calls_in_block(expr.body, rename_map)
            return MapExpr(expr.elem_var, new_seq, new_body, expr.span)
        case GradExpr():
            return GradExpr(_rewrite_expr(expr.expr, rename_map), expr.wrt, expr.span)
        case IndexExpr(base=base, indices=indices, span=span):
            new_base = _rewrite_expr(base, rename_map)
            new_indices = [
                IndexComponent(
                    ic.kind,
                    _rewrite_expr(ic.value, rename_map) if ic.value is not None else None,
                    _rewrite_expr(ic.start, rename_map) if ic.start is not None else None,
                    _rewrite_expr(ic.end, rename_map) if ic.end is not None else None,
                    ic.span,
                )
                for ic in indices
            ]
            return IndexExpr(new_base, new_indices, span)
        case _:
            return expr
