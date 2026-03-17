import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from .lexer import Lexer
from .parser import Parser
from .type_checker import TypeChecker, FnSignature
from .codegen.stablehlo import StableHLOCodegen
from .ad import transform_grad
from .resolver import resolve
from .errors import MaomiError
from .ast_nodes import Span
from .types import MaomiType, ArrayType, StructType


@dataclass
class CompileResult:
    mlir_text: str
    fn_table: dict[str, FnSignature]
    callback_count: int = 0
    callback_labels: dict[int, list[str]] = field(default_factory=dict)
    struct_defs: dict[str, StructType] = field(default_factory=dict)


@dataclass
class RelaxCompileResult:
    ir_mod: object  # tvm.IRModule (lazy import)
    fn_table: dict[str, FnSignature]


def compile_source(source: str, filename: str = "<stdin>", config: dict | None = None) -> CompileResult:
    """Full compilation pipeline: lex -> parse -> resolve imports -> typecheck -> AD -> codegen."""
    tokens = Lexer(source, filename=filename).tokenize()
    program = Parser(tokens, filename=filename).parse()
    program = resolve(program, filename)
    if config is not None:
        _substitute_config(program, config, filename)
    checker = TypeChecker(filename=filename)
    errors = checker.check(program)
    if errors:
        raise errors[0]
    program = transform_grad(program, checker.type_map)
    codegen = StableHLOCodegen(program, checker.type_map)
    mlir_text = codegen.generate()
    return CompileResult(mlir_text, dict(checker.fn_table), codegen._callback_count, codegen._callback_labels, dict(checker.struct_defs))


def _substitute_config(program, config: dict, filename: str):
    """Replace config("key") calls with literal values from config dict."""
    from .ast_nodes import (CallExpr, StringLiteral, FloatLiteral, IntLiteral,
                            LetStmt, ExprStmt, Block, FnDef)

    def _resolve_key(key: str, config: dict):
        """Resolve dotted keys like 'optimizer.lr'."""
        parts = key.split(".")
        val = config
        for part in parts:
            if not isinstance(val, dict) or part not in val:
                raise MaomiError(f"config key '{key}' not found", filename, 0, 0)
            val = val[part]
        return val

    def _to_literal(value, span):
        if isinstance(value, float):
            return FloatLiteral(value, span)
        if isinstance(value, int):
            return IntLiteral(value, span)
        if isinstance(value, str):
            return StringLiteral(value, span)
        raise MaomiError(f"config value type '{type(value).__name__}' not supported", filename, 0, 0)

    def _walk_expr(expr):
        if isinstance(expr, CallExpr) and expr.callee == "config":
            if len(expr.args) != 1 or not isinstance(expr.args[0], StringLiteral):
                raise MaomiError("config() requires exactly one string argument", filename,
                                 expr.span.line_start, expr.span.col_start)
            key = expr.args[0].value
            value = _resolve_key(key, config)
            return _to_literal(value, expr.span)
        # Recurse into sub-expressions
        if isinstance(expr, CallExpr):
            expr.args[:] = [_walk_expr(a) for a in expr.args]
        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            expr.left = _walk_expr(expr.left)
            expr.right = _walk_expr(expr.right)
        elif hasattr(expr, 'operand'):
            expr.operand = _walk_expr(expr.operand)
        elif hasattr(expr, 'expr') and not hasattr(expr, 'wrt'):
            expr.expr = _walk_expr(expr.expr)
        elif hasattr(expr, 'condition'):
            expr.condition = _walk_expr(expr.condition)
        elif hasattr(expr, 'value') and hasattr(expr, 'name') and hasattr(expr, 'type_annotation'):
            # LetStmt
            expr.value = _walk_expr(expr.value)
        return expr

    def _walk_block(block):
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                stmt.value = _walk_expr(stmt.value)
            elif isinstance(stmt, ExprStmt):
                stmt.expr = _walk_expr(stmt.expr)
        if block.expr is not None:
            block.expr = _walk_expr(block.expr)

    for fn in program.functions:
        _walk_block(fn.body)


def compile_source_relax(source: str, filename: str = "<stdin>") -> RelaxCompileResult:
    """Full compilation pipeline with Relax backend: lex -> parse -> resolve -> typecheck -> AD -> relax codegen."""
    from .codegen.relax import RelaxCodegen

    tokens = Lexer(source, filename=filename).tokenize()
    program = Parser(tokens, filename=filename).parse()
    program = resolve(program, filename)
    checker = TypeChecker(filename=filename)
    errors = checker.check(program)
    if errors:
        raise errors[0]
    program = transform_grad(program, checker.type_map)
    ir_mod = RelaxCodegen(program, checker.type_map).generate()
    return RelaxCompileResult(ir_mod, dict(checker.fn_table))


def main():
    parser = argparse.ArgumentParser(prog="maomi", description="Maomi - a pure functional ML language")
    subparsers = parser.add_subparsers(dest="command")

    compile_p = subparsers.add_parser("compile", help="Compile a .mao file")
    compile_p.add_argument("file", help="Path to .mao source file")
    compile_p.add_argument(
        "--emit",
        choices=["tokens", "ast", "types", "stablehlo"],
        default="stablehlo",
        help="Output format (default: stablehlo)",
    )
    compile_p.add_argument(
        "--backend",
        choices=["stablehlo", "relax"],
        default="stablehlo",
        help="Code generation backend (default: stablehlo)",
    )
    compile_p.add_argument("--config", help="Path to TOML config file for compile-time constants")

    run_p = subparsers.add_parser("run", help="Compile and run a .mao file")
    run_p.add_argument("file", help="Path to .mao source file")
    run_p.add_argument("--fn", required=True, help="Function to execute")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed for input generation (default: 42)")
    run_p.add_argument(
        "--backend",
        choices=["stablehlo", "relax"],
        default="stablehlo",
        help="Code generation backend (default: stablehlo)",
    )
    run_p.add_argument(
        "--target",
        choices=["llvm", "metal", "cuda"],
        default="llvm",
        help="Execution target for relax backend (default: llvm)",
    )
    run_p.add_argument("--config", help="Path to TOML config file for compile-time constants")

    lsp_p = subparsers.add_parser("lsp", help="Start Language Server Protocol server")
    lsp_p.add_argument("--stdio", action="store_true", default=True, help="Use stdio transport (default)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compile":
        _compile(args.file, args.emit, getattr(args, "backend", "stablehlo"),
                 config_path=getattr(args, "config", None))
    elif args.command == "run":
        backend = getattr(args, "backend", "stablehlo")
        config_path = getattr(args, "config", None)
        if backend == "relax":
            _run_relax(args.file, args.fn, args.seed, args.target)
        else:
            _run(args.file, args.fn, args.seed, config_path=config_path)
    elif args.command == "lsp":
        try:
            from .lsp import start_server
        except ImportError:
            print(
                "error: pygls is required for the 'lsp' command.\n"
                "Install with: uv sync --extra lsp",
                file=sys.stderr,
            )
            sys.exit(1)
        start_server()


def _load_config(config_path: str | None) -> dict | None:
    if config_path is None:
        return None
    import tomllib
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)


def _compile(path: str, emit: str, backend: str = "stablehlo", config_path: str | None = None):
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Lex
        tokens = Lexer(source, filename=path).tokenize()
        if emit == "tokens":
            for tok in tokens:
                print(f"{tok.line:4d}:{tok.col:<4d} {tok.type.value:<12s} {tok.value!r}")
            return

        # Parse
        program = Parser(tokens, filename=path).parse()

        # Resolve imports
        program = resolve(program, path)

        # Config substitution
        config = _load_config(config_path)
        if config is not None:
            _substitute_config(program, config, path)

        if emit == "ast":
            print(json.dumps(asdict(program), indent=2, default=_json_default))
            return

        # Type check
        checker = TypeChecker(filename=path)
        errors = checker.check(program)
        if errors:
            for err in errors:
                print(f"{err}", file=sys.stderr)
            sys.exit(1)
        if emit == "types":
            if program.struct_defs:
                print(f"{len(program.struct_defs)} struct(s):")
                for sd in program.struct_defs:
                    stype = checker.struct_defs.get(sd.name)
                    if stype:
                        print(f"  {stype}")
            print(f"Type check passed: {len(program.functions)} function(s)")
            for fn in program.functions:
                sig = checker.fn_table.get(fn.name)
                if sig:
                    params_str = ", ".join(
                        f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
                    )
                    print(f"  fn {fn.name}({params_str}) -> {sig.return_type}")
            return

        # AD transform (rewrite grad expressions before codegen)
        program = transform_grad(program, checker.type_map)

        # Codegen
        if backend == "relax":
            from .codegen.relax import RelaxCodegen
            ir_mod = RelaxCodegen(program, checker.type_map).generate()
            print(ir_mod)
        else:
            output = StableHLOCodegen(program, checker.type_map).generate()
            print(output)

    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


def _run(path: str, fn_name: str, seed: int, config_path: str | None = None):
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = compile_source(source, filename=path, config=_load_config(config_path))
    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

    if fn_name not in result.fn_table:
        available = [k for k in result.fn_table if k not in ("mean", "sum", "exp", "log", "tanh", "sqrt", "abs")]
        print(f"error: function '{fn_name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    fn_sig = result.fn_table[fn_name]

    # Check all dims are concrete
    from .api import flatten_type
    for pt in fn_sig.param_types:
        for leaf_t in flatten_type(pt):
            if isinstance(leaf_t, ArrayType):
                for d in leaf_t.dims:
                    if isinstance(d, str):
                        print(
                            f"error: cannot run function with symbolic dimension '{d}'. "
                            f"All dimensions must be concrete integers.",
                            file=sys.stderr,
                        )
                        sys.exit(1)

    try:
        from .jax_runner import run_stablehlo
    except ImportError:
        print(
            "error: JAX is required for the 'run' command.\n"
            "Install with: uv sync --extra run",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create default print callbacks
    import numpy as np
    host_callbacks = []
    for i in range(result.callback_count):
        labels = result.callback_labels.get(i, [])
        prefix = labels[0] if labels else "callback"
        def _print_cb(*args, _idx=i, _prefix=prefix):
            vals = [np.asarray(a) for a in args]
            print(f"[{_prefix}]", *vals)
            return ()
        host_callbacks.append(_print_cb)

    try:
        inputs, outputs = run_stablehlo(
            result.mlir_text, fn_name, fn_sig, seed=seed,
            host_callbacks=host_callbacks if host_callbacks else None,
        )
    except Exception as e:
        print(f"error: execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"--- {fn_name} ---")
    for name, val in zip(fn_sig.param_names, inputs):
        print(f"  {name} = {val}")
    print(f"  -> {outputs}")


def _run_relax(path: str, fn_name: str, seed: int, target: str):
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = compile_source_relax(source, filename=path)
    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

    if fn_name not in result.fn_table:
        available = [k for k in result.fn_table if k not in ("mean", "sum", "exp", "log", "tanh", "sqrt", "abs")]
        print(f"error: function '{fn_name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    fn_sig = result.fn_table[fn_name]

    for pt in fn_sig.param_types:
        if isinstance(pt, ArrayType):
            for d in pt.dims:
                if isinstance(d, str):
                    print(
                        f"error: cannot run function with symbolic dimension '{d}'. "
                        f"All dimensions must be concrete integers.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    try:
        from .tvm_runner import run_relax
    except ImportError:
        print(
            "error: TVM is required for the 'relax' backend.\n"
            "Install with: uv sync --extra tvm",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        inputs, outputs = run_relax(result.ir_mod, fn_name, fn_sig, target=target, seed=seed)
    except Exception as e:
        print(f"error: execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- {fn_name} ({target}) ---")
    for name, val in zip(fn_sig.param_names, inputs):
        print(f"  {name} = {val}")
    print(f"  -> {outputs}")


def _json_default(obj):
    if isinstance(obj, Span):
        return {"line_start": obj.line_start, "col_start": obj.col_start, "line_end": obj.line_end, "col_end": obj.col_end}
    return str(obj)


if __name__ == "__main__":
    main()
