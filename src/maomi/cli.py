import argparse
import json
import sys
from dataclasses import asdict

from .lexer import Lexer
from .parser import Parser
from .type_checker import TypeChecker
from .codegen_stablehlo import StableHLOCodegen
from .ad import transform_grad
from .errors import MaomiError
from .ast_nodes import Span


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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compile":
        _compile(args.file, args.emit)


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
            print(f"Type check passed: {len(program.functions)} function(s)")
            for fn in program.functions:
                sig = checker.fn_table.get(fn.name)
                if sig:
                    params_str = ", ".join(
                        f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types)
                    )
                    eff = f" ! {fn.effect}" if fn.effect else ""
                    print(f"  fn {fn.name}({params_str}) -> {sig.return_type}{eff}")
            return

        # AD transform (rewrite grad expressions before codegen)
        program = transform_grad(program, checker.type_map)

        # Codegen
        output = StableHLOCodegen(program, checker.type_map).generate()
        print(output)

    except MaomiError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


def _json_default(obj):
    if isinstance(obj, Span):
        return {"line_start": obj.line_start, "col_start": obj.col_start, "line_end": obj.line_end, "col_end": obj.col_end}
    return str(obj)


if __name__ == "__main__":
    main()
