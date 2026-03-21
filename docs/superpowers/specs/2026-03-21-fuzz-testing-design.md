# Fuzz Testing Design for Maomi Compiler

**Date:** 2026-03-21
**Status:** Approved
**Approach:** Approach B — separate test files per pipeline stage, shared strategy library

## Goals

1. **Crash detection** — no input should cause an unhandled Python exception. Every failure should be a clean `MaomiError`.
2. **Correctness validation** — if it type-checks, codegen shouldn't fail; AD gradients should match finite differences.
3. **LSP robustness** — the language server should handle any document content and edit sequence without crashing, and maintain consistency across features.

## Decisions

- **Framework:** hypothesis (property-based testing for Python)
- **Generation strategy:** mutation-based for raw input stages (parser, LSP), grammar-based for deeper stages (type checker, AD/codegen), plus from-scratch AST generation (Futhark-style) for type checker
- **CI:** not integrated yet — local-only for now
- **Budget:** 500 examples per property (medium), 5s deadline per example
- **Crash regression:** hypothesis example database + explicit `@example()` decorators for important crashes

## Inspirations from Reference Compilers

- **rust-analyzer:** parser fuzzing with raw bytes, minimized crash cases committed as regression tests
- **Halide:** type-aware random expression generators with depth limits to prevent runaway recursion
- **Futhark:** QuickCheck-based property testing for types, syntax, and AD derivatives
- **Dex-lang:** finite-difference gradient verification as gold-standard AD correctness check

## File Structure

```
pyproject.toml                      # add hypothesis>=6.0 to dev deps
tests/
├── conftest.py                     # hypothesis profile registration, fuzz marker
├── fuzz_strategies.py              # shared strategy library
├── test_fuzz_parser.py             # Stage 1: parser robustness (4 properties)
├── test_fuzz_typechecker.py        # Stage 2: type checker robustness (5 properties)
├── test_fuzz_ad_codegen.py         # Stage 3: AD & codegen robustness (4 properties)
└── test_fuzz_lsp.py                # Stage 4: LSP robustness (15 properties)
```

**Total: 28 fuzz properties** across 4 test files.

## Strategy Library (`tests/fuzz_strategies.py`)

### Mutation Strategies (Stages 1 & 4)

- `mutated_source()` — takes valid `.mao` snippets, randomly corrupts them (insert/delete/swap chars, truncate, inject Unicode, duplicate lines)
- `random_bytes()` — pure random strings including null bytes, control chars, very long inputs
- `near_valid_source()` — valid code with one targeted corruption (missing semicolon, swapped brace, wrong keyword)

### Grammar-Based Strategies (Stages 2 & 3)

All take a `max_depth` parameter (a la Halide) to prevent runaway recursion, defaulting to 4.

- `valid_type()` — generates scalar types (`f32`, `i32`, `bool`) and shaped arrays (`f32[N]`, `f32[N,M]`)
- `valid_expr(type, depth)` — generates expressions of a given type, recursing into sub-expressions with `depth-1`. At depth 0, only generates leaves (literals, identifiers)
- `valid_function()` — composes params + typed body + return into a complete `fn`
- `valid_program()` — composes functions and optionally structs into a full program
- `grad_program()` — generates a scalar-returning function + `grad()` call, ensuring the function is differentiable (no `argmax`, no integer-only ops)

### From-Scratch AST Strategies (Stage 2)

- `random_ast_node()` — generates arbitrary AST nodes with random compositions (types and expressions independent, Futhark-style)
- `random_ast_program()` — composes random AST nodes into a `Program`

### LSP Strategies (Stage 4)

- `partial_source()` — valid code truncated at random points (simulates mid-typing)
- `lsp_edit_sequence()` — sequences of text edits (insert/delete/replace at random positions)

## Stage 1: Parser Robustness (`tests/test_fuzz_parser.py`)

### Properties

1. **"Lexer never crashes"** — `Lexer(arbitrary_string).tokenize()` always returns a token list or populates `lexer.errors` with clean `LexerError`s. Never an unhandled exception.

2. **"Parser never crashes"** — `Parser(tokens).parse()` always returns a `Program` AST (possibly empty/partial) or populates `parser.errors` with clean `ParseError`s. Never an unhandled exception.

3. **"Lexer roundtrip stability"** — tokenizing the same input twice produces the same token list (determinism).

4. **"Error messages are valid"** — every error in `lexer.errors` and `parser.errors` has non-None `line`, `col`, and `message` fields.

### Input Strategies

- `random_bytes()` and `mutated_source()` for properties 1-2
- `near_valid_source()` for higher-value edge case discovery
- Valid `.mao` source for property 3

## Stage 2: Type Checker Robustness (`tests/test_fuzz_typechecker.py`)

### Properties

1. **"Type checker never crashes on valid syntax"** — given a syntactically valid program (from `valid_program()`), `TypeChecker().check(program)` either returns an empty error list or a list of clean `MaomiTypeError`s. Never an unhandled exception.

2. **"Type checker never crashes on mutated ASTs"** — take a valid program, mutate the AST (swap types, rename identifiers, change operator kinds, remove parameters, add extra arguments). Type checker rejects gracefully.

3. **"Type checker never crashes on random ASTs"** — from-scratch composed ASTs (Futhark-style). Types and expressions generated independently, producing combinations nobody would write but that the type checker must handle.

4. **"Type map consistency"** — if `check()` returns no errors, every AST node that should have a type has an entry in `type_map`, and every entry is a valid `MaomiType`.

5. **"Idempotent rejection"** — checking the same program twice produces the same errors (determinism, no state leakage).

### Input Strategies

- `valid_program()` for properties 1, 4, 5
- `valid_program()` + AST mutation for property 2
- `random_ast_program()` for property 3

## Stage 3: AD & Codegen Robustness (`tests/test_fuzz_ad_codegen.py`)

### Properties

1. **"Codegen never crashes on well-typed programs"** — if `TypeChecker().check()` returns no errors, then `StableHLOCodegen(program, type_map).generate()` produces a string of valid MLIR. Never an unhandled exception.

2. **"AD transform never crashes on differentiable programs"** — given a `grad_program()`, `transform_grad(program, type_map)` either succeeds or raises a clean `MaomiError`. Never an unhandled exception.

3. **"AD output is valid for codegen"** — if `transform_grad` succeeds, the resulting program passes through codegen without crashing. Tests the contract between AD and codegen.

4. **"Finite-difference agreement"** — for programs where `grad(f)(x)` succeeds through codegen, the gradient approximately matches `(f(x+h) - f(x-h)) / 2h` for randomly generated input values. Tolerance: `rtol=1e-3`. Requires JAX (`@pytest.mark.skipif(not has_jax)`). Uses fewer examples (`max_examples=100`).

### Input Strategies

- `valid_program()` for property 1
- `grad_program()` for properties 2-4 (restricted to differentiable ops)
- `hypothesis.strategies.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)` for property 4 inputs

## Stage 4: LSP Robustness (`tests/test_fuzz_lsp.py`)

### Crash Resistance Properties (5)

1. **"LSP never crashes on malformed documents"** — open a document with `random_bytes()` or `mutated_source()` content, trigger diagnostics. Returns diagnostics without crashing.

2. **"LSP never crashes on completion requests"** — open a document with `partial_source()`, request completion at a random position. Returns a completion list or null.

3. **"LSP never crashes on hover/goto-def at arbitrary positions"** — open a document, request hover or go-to-definition at random line/column. Returns a result or null.

4. **"LSP survives rapid edit sequences"** — open a document, apply `lsp_edit_sequence()` edits, request diagnostics after each. Server stays responsive.

5. **"LSP never crashes on rename/references at arbitrary positions"** — rename and find-references with partial/broken documents.

### Determinism Properties (2)

6. **"Same document produces same diagnostics"** — publishing the same content twice produces identical diagnostic lists. Catches state leakage.

7. **"Same document produces same semantic tokens"** — semantic token output is deterministic.

### Consistency Properties (4)

8. **"Hover type matches type checker"** — if hover returns a type string, it matches what `TypeChecker.type_map` has for the AST node at that position.

9. **"Go-to-def target contains the symbol"** — if go-to-def returns a location, the text at that location contains the symbol name.

10. **"Completion-then-insert produces parseable code"** — if completion suggests `foo`, inserting it at the cursor produces code that lexes/parses without crashing.

11. **"Find references is a superset of rename locations"** — every location that rename would change also appears in find-references for that symbol.

### Structural Properties (2)

12. **"Semantic tokens don't overlap"** — token ranges are non-overlapping and sorted.

13. **"Document symbols match definitions"** — every `fn` and `struct` in the source appears in document symbols, and every returned symbol corresponds to a real definition.

### State Properties (2)

14. **"Diagnostics clear when errors are fixed"** — invalid code produces diagnostics; fixing to valid code and re-publishing clears them.

15. **"Rename preserves validity"** — if a program type-checks, renaming any symbol to a fresh name (no collisions) produces a program that still type-checks.

## Test Dependencies & Configuration

### New Dependency

Add `hypothesis>=6.0` to the `dev` dependency group in `pyproject.toml`.

### Hypothesis Settings

- Default profile: `max_examples=500`, `deadline=5000` (5 seconds per example)
- `suppress_health_check=[HealthCheck.too_slow]` for AD/codegen and LSP tests
- Configured via `hypothesis.settings.register_profile` in `conftest.py`

### Test Markers

- `@pytest.mark.fuzz` — custom marker on all fuzz tests, run with `pytest -m fuzz` or exclude with `pytest -m "not fuzz"`
- `@pytest.mark.skipif(not has_jax)` — on finite-difference property only

### Crash Regression

- Hypothesis saves failing examples to `.hypothesis/examples/` automatically (replayed every run)
- Important crashes additionally get explicit `@example(...)` decorators committed to git

## What We're NOT Building

- No custom fuzz runner or harness (just pytest)
- No corpus directory or management (hypothesis handles its own example database)
- No CI integration (local-only for now)
- No profile switching beyond the single default
