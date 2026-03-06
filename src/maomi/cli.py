import argparse
import json
import sys
from dataclasses import asdict, dataclass

from .lexer import Lexer
from .parser import Parser
from .type_checker import TypeChecker, FnSignature
from .codegen_stablehlo import StableHLOCodegen
from .ad import transform_grad
from .errors import MaomiError
from .ast_nodes import Span
from .types import MaomiType, ArrayType


@dataclass
class CompileResult:
    mlir_text: str
    fn_table: dict[str, FnSignature]


def compile_source(source: str, filename: str = "<stdin>") -> CompileResult:
    """Full compilation pipeline: lex -> parse -> typecheck -> AD -> codegen."""
    tokens = Lexer(source, filename=filename).tokenize()
    program = Parser(tokens, filename=filename).parse()
    checker = TypeChecker(filename=filename)
    errors = checker.check(program)
    if errors:
        raise errors[0]
    program = transform_grad(program, checker.type_map)
    mlir_text = StableHLOCodegen(program, checker.type_map).generate()
    return CompileResult(mlir_text, dict(checker.fn_table))


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

    run_p = subparsers.add_parser("run", help="Compile and run a .mao file (requires JAX)")
    run_p.add_argument("file", help="Path to .mao source file")
    run_p.add_argument("--fn", required=True, help="Function to execute")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed for input generation (default: 42)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compile":
        _compile(args.file, args.emit)
    elif args.command == "run":
        _run(args.file, args.fn, args.seed)


def _compile(path: str, emit: str):
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
        output = StableHLOCodegen(program, checker.type_map).generate()
        print(output)

    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


def _run(path: str, fn_name: str, seed: int):
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = compile_source(source, filename=path)
    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

    if fn_name not in result.fn_table:
        available = [k for k in result.fn_table if k not in ("mean", "sum", "exp", "log", "tanh", "sqrt", "abs")]
        print(f"error: function '{fn_name}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    fn_sig = result.fn_table[fn_name]

    # Check all dims are concrete
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
        from .iree_runner import run_stablehlo
    except ImportError:
        print(
            "error: IREE is required for the 'run' command.\n"
            "Install with: uv sync --extra run",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        inputs, outputs = run_stablehlo(result.mlir_text, fn_name, fn_sig, seed=seed)
    except Exception as e:
        print(f"error: execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"--- {fn_name} ---")
    for name, val in zip(fn_sig.param_names, inputs):
        print(f"  {name} = {val}")
    print(f"  -> {outputs}")


def _json_default(obj):
    if isinstance(obj, Span):
        return {"line_start": obj.line_start, "col_start": obj.col_start, "line_end": obj.line_end, "col_end": obj.col_end}
    return str(obj)


if __name__ == "__main__":
    main()
