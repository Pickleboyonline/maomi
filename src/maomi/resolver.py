from __future__ import annotations

import os
from pathlib import Path

_STDLIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib")

from .ast_nodes import (
    Program,
    ImportDecl,
    FnDef,
    Param,
    StructDef,
    StructLiteral,
    FoldExpr,
    WithExpr,
    Block,
    LetStmt,
    ExprStmt,
    CallExpr,
    ScanExpr,
    WhileExpr,
    MapExpr,
    IfExpr,
    GradExpr,
    ValueAndGradExpr,
    UnaryOp,
    BinOp,
    IndexExpr,
    IndexComponent,
    TypeAnnotation,
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
        self._cache: dict[str, tuple[list[FnDef], list[StructDef]]] = {}
        self._resolving: set[str] = set()  # cycle detection

    def resolve(self, program: Program, file_path: str) -> Program:
        abs_path = os.path.abspath(file_path)
        merged_fns: list[FnDef] = []
        merged_structs: list[StructDef] = []
        seen_struct_names: dict[str, str] = {}  # name -> source module (for collision detection)

        for imp in program.imports:
            mod_path = self._resolve_file(imp, abs_path)
            module_name = imp.alias if imp.alias else _stem(imp.module_path)

            # Get prefixed functions and struct_defs for this module (cached)
            prefixed_fns, prefixed_structs = self._load_module(mod_path, module_name)

            if imp.names is not None:
                # from math import { relu, Point };
                # Include ALL module fns and structs (needed for internal calls/types),
                # plus add aliases for the selected names
                merged_fns.extend(prefixed_fns)

                # Deduplicate structs (diamond dependency)
                for sd in prefixed_structs:
                    if sd.name not in seen_struct_names:
                        merged_structs.append(sd)
                        seen_struct_names[sd.name] = module_name

                available_fns = {fn.name.split(".", 1)[1]: fn for fn in prefixed_fns
                                 if "." in fn.name}
                available_structs = {sd.name.split(".", 1)[1]: sd for sd in prefixed_structs
                                     if "." in sd.name}

                for name in imp.names:
                    if name in available_structs:
                        # Check no collision with main file structs
                        for local_sd in program.struct_defs:
                            if local_sd.name == name:
                                raise MaomiError(
                                    f"imported struct '{name}' conflicts with local struct '{name}'",
                                    file_path,
                                    imp.span.line_start,
                                    imp.span.col_start,
                                )
                        # Check no collision with previously imported struct aliases
                        if name in seen_struct_names:
                            raise MaomiError(
                                f"struct '{name}' imported from multiple modules",
                                file_path,
                                imp.span.line_start,
                                imp.span.col_start,
                            )
                        # Create an alias StructDef
                        orig = available_structs[name]
                        alias_sd = StructDef(name, orig.fields, orig.span,
                                             doc=orig.doc, canonical_name=orig.name)
                        merged_structs.append(alias_sd)
                        seen_struct_names[name] = module_name
                    elif name in available_fns:
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
                        orig = available_fns[name]
                        alias_fn = FnDef(name, orig.params, orig.return_type,
                                         _rewrite_calls_in_block(orig.body, {name: orig.name}),
                                         orig.span, doc=orig.doc,
                                         source_file=orig.source_file)
                        merged_fns.append(alias_fn)
                    else:
                        avail_list = sorted(set(available_fns.keys()) | set(available_structs.keys()))
                        raise MaomiError(
                            f"module '{module_name}' has no function or struct '{name}'. "
                            f"Available: {avail_list}",
                            file_path,
                            imp.span.line_start,
                            imp.span.col_start,
                        )
            else:
                # import math; — qualified access only
                merged_fns.extend(prefixed_fns)
                # Deduplicate structs (diamond dependency)
                for sd in prefixed_structs:
                    if sd.name not in seen_struct_names:
                        merged_structs.append(sd)
                        seen_struct_names[sd.name] = module_name

        # Imported structs first (so they're registered before main file structs that may reference them)
        all_structs = merged_structs + list(program.struct_defs)
        # Main file's own functions come last (can call imported ones)
        merged_fns.extend(program.functions)
        return Program(program.imports, all_structs, merged_fns, program.span,
                       type_aliases=program.type_aliases)

    def _load_module(self, mod_path: str, module_name: str) -> tuple[list[FnDef], list[StructDef]]:
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

        # Build rename maps for functions and structs
        local_fn_names = {fn.name for fn in mod_program.functions}
        fn_rename_map = {name: f"{module_name}.{name}" for name in local_fn_names}

        local_struct_names = {sd.name for sd in mod_program.struct_defs}
        struct_rename_map = {name: f"{module_name}.{name}" for name in local_struct_names}

        # Prefix functions (rewriting calls and struct references)
        prefixed_fns = [_prefix_fn(fn, fn_rename_map, struct_rename_map, mod_path)
                        for fn in mod_program.functions]

        # Prefix struct defs
        prefixed_structs = [_prefix_struct(sd, struct_rename_map)
                            for sd in mod_program.struct_defs]

        self._resolving.discard(mod_path)
        self._cache[cache_key] = (prefixed_fns, prefixed_structs)
        return (prefixed_fns, prefixed_structs)

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
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

        # Fallback: look in stdlib
        stdlib_candidate = os.path.join(_STDLIB_DIR, f"{path}.mao")
        if os.path.exists(stdlib_candidate):
            return os.path.abspath(stdlib_candidate)

        # Return original candidate (will error in _load_module with good message)
        return os.path.abspath(candidate)


def _stem(module_path: str) -> str:
    """Derive module name from path: '../lib/nn.mao' -> 'nn', 'math' -> 'math'."""
    base = os.path.basename(module_path)
    if base.endswith(".mao"):
        base = base[:-4]
    return base


def _rewrite_type_annotation(ta: TypeAnnotation, struct_rename_map: dict[str, str]) -> TypeAnnotation:
    """Rewrite struct references in a type annotation."""
    new_base = struct_rename_map.get(ta.base, ta.base)
    if new_base == ta.base:
        return ta
    return TypeAnnotation(new_base, ta.dims, ta.span, wildcard=ta.wildcard)


def _prefix_struct(sd: StructDef, struct_rename_map: dict[str, str]) -> StructDef:
    """Rename a struct and rewrite struct references in its field types."""
    new_name = struct_rename_map.get(sd.name, sd.name)
    new_fields = [(fname, _rewrite_type_annotation(fta, struct_rename_map))
                  for fname, fta in sd.fields]
    return StructDef(new_name, new_fields, sd.span, doc=sd.doc)


def _prefix_fn(fn: FnDef, fn_rename_map: dict[str, str],
               struct_rename_map: dict[str, str],
               source_file: str | None = None) -> FnDef:
    """Rename a function and rewrite its internal calls and struct references."""
    new_name = fn_rename_map.get(fn.name, fn.name)
    new_params = [Param(p.name, _rewrite_type_annotation(p.type_annotation, struct_rename_map),
                        p.span, p.comptime)
                  for p in fn.params]
    new_return_type = _rewrite_type_annotation(fn.return_type, struct_rename_map)
    new_body = _rewrite_calls_in_block(fn.body, fn_rename_map, struct_rename_map)
    return FnDef(new_name, new_params, new_return_type, new_body, fn.span,
                 doc=fn.doc, source_file=fn.source_file or source_file)


def _rewrite_calls_in_block(block: Block, fn_rename_map: dict[str, str],
                             struct_rename_map: dict[str, str] | None = None) -> Block:
    if struct_rename_map is None:
        struct_rename_map = {}
    new_stmts = []
    for stmt in block.stmts:
        if isinstance(stmt, LetStmt):
            new_ta = (_rewrite_type_annotation(stmt.type_annotation, struct_rename_map)
                      if stmt.type_annotation else None)
            new_stmts.append(LetStmt(stmt.name, new_ta,
                                     _rewrite_expr(stmt.value, fn_rename_map, struct_rename_map),
                                     stmt.span))
        elif isinstance(stmt, ExprStmt):
            new_stmts.append(ExprStmt(_rewrite_expr(stmt.expr, fn_rename_map, struct_rename_map),
                                      stmt.span))
    new_expr = (_rewrite_expr(block.expr, fn_rename_map, struct_rename_map)
                if block.expr is not None else None)
    return Block(new_stmts, new_expr, block.span)


def _rewrite_expr(expr: Expr, fn_rename_map: dict[str, str],
                  struct_rename_map: dict[str, str] | None = None) -> Expr:
    if struct_rename_map is None:
        struct_rename_map = {}
    match expr:
        case CallExpr(callee=callee, args=args, span=span):
            new_callee = fn_rename_map.get(callee, callee)
            new_args = [_rewrite_expr(a, fn_rename_map, struct_rename_map) for a in args]
            new_named = [(n, _rewrite_expr(v, fn_rename_map, struct_rename_map))
                         for n, v in expr.named_args]
            return CallExpr(new_callee, new_args, span, named_args=new_named)
        case StructLiteral(name=name, fields=fields, span=span):
            new_name = struct_rename_map.get(name, name)
            new_fields = [(fname, _rewrite_expr(fval, fn_rename_map, struct_rename_map))
                          for fname, fval in fields]
            return StructLiteral(new_name, new_fields, span)
        case BinOp(op=op, left=left, right=right, span=span):
            return BinOp(op, _rewrite_expr(left, fn_rename_map, struct_rename_map),
                         _rewrite_expr(right, fn_rename_map, struct_rename_map), span)
        case UnaryOp(op=op, operand=operand, span=span):
            return UnaryOp(op, _rewrite_expr(operand, fn_rename_map, struct_rename_map), span)
        case IfExpr(condition=cond, then_block=then_b, else_block=else_b, span=span):
            return IfExpr(_rewrite_expr(cond, fn_rename_map, struct_rename_map),
                          _rewrite_calls_in_block(then_b, fn_rename_map, struct_rename_map),
                          _rewrite_calls_in_block(else_b, fn_rename_map, struct_rename_map), span)
        case ScanExpr():
            new_init = _rewrite_expr(expr.init, fn_rename_map, struct_rename_map)
            new_seqs = [_rewrite_expr(s, fn_rename_map, struct_rename_map) for s in expr.sequences]
            new_body = _rewrite_calls_in_block(expr.body, fn_rename_map, struct_rename_map)
            return ScanExpr(expr.carry_var, expr.elem_vars, new_init, new_seqs,
                            new_body, expr.span, expr.reverse)
        case FoldExpr():
            new_init = _rewrite_expr(expr.init, fn_rename_map, struct_rename_map)
            new_seqs = [_rewrite_expr(s, fn_rename_map, struct_rename_map) for s in expr.sequences]
            new_body = _rewrite_calls_in_block(expr.body, fn_rename_map, struct_rename_map)
            return FoldExpr(expr.carry_var, expr.elem_vars, new_init, new_seqs,
                            new_body, expr.span)
        case WithExpr():
            new_base = _rewrite_expr(expr.base, fn_rename_map, struct_rename_map)
            new_updates = [(path, _rewrite_expr(val, fn_rename_map, struct_rename_map))
                           for path, val in expr.updates]
            return WithExpr(new_base, new_updates, expr.span)
        case WhileExpr():
            new_init = _rewrite_expr(expr.init, fn_rename_map, struct_rename_map)
            new_cond = _rewrite_calls_in_block(expr.cond, fn_rename_map, struct_rename_map)
            new_body = _rewrite_calls_in_block(expr.body, fn_rename_map, struct_rename_map)
            return WhileExpr(expr.state_var, new_init, expr.max_iters,
                             new_cond, new_body, expr.span)
        case MapExpr():
            new_seq = _rewrite_expr(expr.sequence, fn_rename_map, struct_rename_map)
            new_body = _rewrite_calls_in_block(expr.body, fn_rename_map, struct_rename_map)
            return MapExpr(expr.elem_var, new_seq, new_body, expr.span)
        case GradExpr():
            return GradExpr(_rewrite_expr(expr.expr, fn_rename_map, struct_rename_map),
                            expr.wrt, expr.span)
        case ValueAndGradExpr():
            return ValueAndGradExpr(_rewrite_expr(expr.expr, fn_rename_map, struct_rename_map),
                                    expr.wrt, expr.span)
        case IndexExpr(base=base, indices=indices, span=span):
            new_base = _rewrite_expr(base, fn_rename_map, struct_rename_map)
            new_indices = [
                IndexComponent(
                    ic.kind,
                    _rewrite_expr(ic.value, fn_rename_map, struct_rename_map) if ic.value is not None else None,
                    _rewrite_expr(ic.start, fn_rename_map, struct_rename_map) if ic.start is not None else None,
                    _rewrite_expr(ic.end, fn_rename_map, struct_rename_map) if ic.end is not None else None,
                    ic.span,
                )
                for ic in indices
            ]
            return IndexExpr(new_base, new_indices, span)
        case _:
            return expr
