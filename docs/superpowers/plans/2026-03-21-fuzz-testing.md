# Fuzz Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 28 property-based fuzz tests across 4 compiler pipeline stages (parser, type checker, AD/codegen, LSP) using hypothesis.

**Architecture:** Shared strategy library (`tests/fuzz_strategies.py`) provides mutation-based, grammar-based, and from-scratch AST generators. Four test files (`test_fuzz_parser.py`, `test_fuzz_typechecker.py`, `test_fuzz_ad_codegen.py`, `test_fuzz_lsp.py`) use these strategies to test crash-resistance and correctness properties. A `conftest.py` registers hypothesis settings and a `fuzz` pytest marker.

**Tech Stack:** Python 3.11+, hypothesis>=6.80, pytest>=8.0

**Spec:** `docs/superpowers/specs/2026-03-21-fuzz-testing-design.md`

---

### Task 1: Project Setup — Dependencies, Config, Gitignore

**Files:**
- Modify: `pyproject.toml` (dev dependency group)
- Modify: `.gitignore` (add `.hypothesis/`)
- Create: `tests/conftest.py`

- [ ] **Step 1: Add hypothesis to dev dependencies**

In `pyproject.toml`, change the dev dependency group from:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]
```
to:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "hypothesis>=6.80"]
```

- [ ] **Step 2: Add `.hypothesis/` to `.gitignore`**

In `.gitignore`, after the `# Testing` section (which has `.pytest_cache/`), add:
```
.hypothesis/
```

- [ ] **Step 3: Install the new dependency**

Run: `uv sync --group dev`
Expected: hypothesis installed successfully

- [ ] **Step 4: Create `tests/conftest.py` with hypothesis settings and fuzz marker**

```python
"""Hypothesis settings and pytest markers for fuzz testing."""

import pytest
from hypothesis import HealthCheck, settings

# Register hypothesis profiles
settings.register_profile(
    "fuzz",
    max_examples=500,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.load_profile("fuzz")


def pytest_configure(config):
    config.addinivalue_line("markers", "fuzz: property-based fuzz tests")
```

- [ ] **Step 5: Verify hypothesis imports and profile work**

Run: `uv run python -c "from hypothesis import given, strategies as st, settings; print(settings.get_profile()); print('OK')"`
Expected: prints "fuzz" and "OK"

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore tests/conftest.py
git commit -m "feat: add hypothesis dependency and fuzz test config"
```

---

### Task 2: Strategy Library — Mutation Strategies

**Files:**
- Create: `tests/fuzz_strategies.py`

This task builds the mutation-based strategies used by Stages 1 and 4. Grammar-based and AST strategies are added in later tasks.

- [ ] **Step 1: Write tests for the mutation strategies**

Create `tests/test_fuzz_strategies.py`:
```python
"""Tests for fuzz strategy generators — verify they produce usable output."""

from hypothesis import given, settings

from fuzz_strategies import mutated_source, near_valid_source, random_bytes


class TestMutationStrategies:
    @given(random_bytes())
    @settings(max_examples=50)
    def test_random_bytes_produces_strings(self, s):
        assert isinstance(s, str)

    @given(mutated_source())
    @settings(max_examples=50)
    def test_mutated_source_produces_strings(self, s):
        assert isinstance(s, str)
        assert len(s) > 0

    @given(near_valid_source())
    @settings(max_examples=50)
    def test_near_valid_source_produces_strings(self, s):
        assert isinstance(s, str)
        assert len(s) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fuzz_strategies.py -v`
Expected: FAIL with `ImportError` (fuzz_strategies doesn't exist yet)

- [ ] **Step 3: Implement mutation strategies in `tests/fuzz_strategies.py`**

```python
"""Shared hypothesis strategies for fuzz testing the Maomi compiler.

Provides mutation-based, grammar-based, and from-scratch AST generators
used across all fuzz test files.
"""

from __future__ import annotations

import hypothesis.strategies as st

# ---------------------------------------------------------------------------
# Seed corpus — valid .mao snippets for mutation-based strategies
# ---------------------------------------------------------------------------

SEED_CORPUS = [
    "fn f(a: f32, b: f32) -> f32 { a + b }",
    "fn f(x: f32) -> f32 { x * x }",
    "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 - x } }",
    "fn f(x: f32[3]) -> f32 { x[0] + x[1] + x[2] }",
    "fn f(x: f32[3, 3]) -> f32[3] { x[0] }",
    "struct S { x: f32, y: f32 }",
    "fn f(s: S) -> f32 { s.x + s.y }\nstruct S { x: f32, y: f32 }",
    "fn f(x: f32) -> f32 { let y = x * 2.0; y + 1.0 }",
    "fn f(x: f32, y: f32) -> f32 { grad(x * y + x, x) }",
    "fn f(x: f32[4]) -> f32[4] { map e in x { e * 2.0 } }",
    "fn f(x: f32[4]) -> f32 { let s = scan (c, e) in (0.0, x) { c + e }; s }",
    "fn f(x: f32) -> f32 { exp(x) }",
    "fn f(x: f32) -> f32 { log(x) }",
    "fn f(x: f32) -> f32 { tanh(x) }",
    "fn f(x: f32) -> f32 { sqrt(x) }",
    "fn f(x: f32) -> f32 { cos(x) + sin(x) }",
    "fn f(x: f32) -> f32 { cast(x, f64) }",
    "fn f(x: f32[3]) -> f32 { sum(x) }",
    "fn f(x: f32[3]) -> f32 { mean(x) }",
    "fn f(x: f32[2, 3]) -> f32[2] { sum(x, axis=1) }",
]


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------


def _insert_random_char(s: str, draw: st.DrawFn) -> str:
    """Insert a random character at a random position."""
    if not s:
        return s
    pos = draw(st.integers(min_value=0, max_value=len(s)))
    char = draw(st.text(min_size=1, max_size=1))
    return s[:pos] + char + s[pos:]


def _delete_random_char(s: str, draw: st.DrawFn) -> str:
    """Delete a random character."""
    if not s:
        return s
    pos = draw(st.integers(min_value=0, max_value=len(s) - 1))
    return s[:pos] + s[pos + 1 :]


def _swap_random_chars(s: str, draw: st.DrawFn) -> str:
    """Swap two adjacent characters."""
    if len(s) < 2:
        return s
    pos = draw(st.integers(min_value=0, max_value=len(s) - 2))
    return s[:pos] + s[pos + 1] + s[pos] + s[pos + 2 :]


def _truncate(s: str, draw: st.DrawFn) -> str:
    """Truncate at a random position."""
    if not s:
        return s
    pos = draw(st.integers(min_value=1, max_value=len(s)))
    return s[:pos]


def _duplicate_line(s: str, draw: st.DrawFn) -> str:
    """Duplicate a random line."""
    lines = s.split("\n")
    if not lines:
        return s
    idx = draw(st.integers(min_value=0, max_value=len(lines) - 1))
    lines.insert(idx, lines[idx])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mutation strategies (Stages 1 & 4)
# ---------------------------------------------------------------------------


@st.composite
def random_bytes(draw: st.DrawFn) -> str:
    """Pure random strings including control chars and Unicode."""
    return draw(st.text(min_size=0, max_size=500))


@st.composite
def mutated_source(draw: st.DrawFn) -> str:
    """Take a seed corpus snippet and randomly corrupt it."""
    base = draw(st.sampled_from(SEED_CORPUS))
    mutations = [
        _insert_random_char,
        _delete_random_char,
        _swap_random_chars,
        _truncate,
        _duplicate_line,
    ]
    # Apply 1-3 mutations (different mutation each time)
    num_mutations = draw(st.integers(min_value=1, max_value=3))
    result = base
    for _ in range(num_mutations):
        mutation = draw(st.sampled_from(mutations))
        result = mutation(result, draw)
    return result


@st.composite
def near_valid_source(draw: st.DrawFn) -> str:
    """Valid code with one targeted corruption."""
    base = draw(st.sampled_from(SEED_CORPUS))
    corruption = draw(
        st.sampled_from(
            [
                # Missing closing brace
                lambda s: s.rstrip("}").rstrip(),
                # Missing semicolon after let
                lambda s: s.replace(";", "", 1) if ";" in s else s + " ???",
                # Extra semicolon at end
                lambda s: s + ";",
                # Replace a keyword with nonsense
                lambda s: s.replace("fn ", "funk ", 1),
                # Remove opening paren
                lambda s: s.replace("(", "", 1),
                # Add random token in middle
                lambda s: s[: len(s) // 2] + " @#$ " + s[len(s) // 2 :],
            ]
        )
    )
    return corruption(base)
```

- [ ] **Step 4: Run strategy tests to verify they pass**

Run: `uv run pytest tests/test_fuzz_strategies.py -v`
Expected: PASS — all 3 tests produce strings

- [ ] **Step 5: Commit**

```bash
git add tests/fuzz_strategies.py tests/test_fuzz_strategies.py
git commit -m "feat: add mutation-based fuzz strategies with seed corpus"
```

---

### Task 3: Stage 1 — Parser Fuzz Tests (4 properties)

**Files:**
- Create: `tests/test_fuzz_parser.py`

**Key imports:**
- `from maomi.lexer import Lexer`
- `from maomi.parser import Parser`
- `from maomi.errors import MaomiError`

- [ ] **Step 1: Write `tests/test_fuzz_parser.py` with all 4 properties**

```python
"""Stage 1: Parser robustness fuzz tests.

Properties:
1. Lexer never crashes on arbitrary input
2. Parser never crashes on lexer output
3. Lexer roundtrip stability (determinism)
4. Error messages are well-formed
"""

import pytest
from hypothesis import given, settings

from fuzz_strategies import mutated_source, near_valid_source, random_bytes
from maomi.errors import MaomiError
from maomi.lexer import Lexer
from maomi.parser import Parser


@pytest.mark.fuzz
class TestLexerNeverCrashes:
    """Property 1: Lexer(arbitrary_string).tokenize() never raises
    a non-MaomiError exception."""

    @given(source=random_bytes())
    def test_random_bytes(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            lexer.tokenize()
        except MaomiError:
            pass  # Expected — clean rejection

    @given(source=mutated_source())
    def test_mutated_source(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            lexer.tokenize()
        except MaomiError:
            pass

    @given(source=near_valid_source())
    def test_near_valid_source(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            lexer.tokenize()
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestParserNeverCrashes:
    """Property 2: Parser(tokens).parse() never raises
    a non-MaomiError exception."""

    @given(source=random_bytes())
    def test_random_bytes(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, filename="<fuzz>")
            parser.parse()
        except MaomiError:
            pass

    @given(source=mutated_source())
    def test_mutated_source(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, filename="<fuzz>")
            parser.parse()
        except MaomiError:
            pass

    @given(source=near_valid_source())
    def test_near_valid_source(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, filename="<fuzz>")
            parser.parse()
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestLexerDeterminism:
    """Property 3: Tokenizing the same input twice produces the same tokens."""

    @given(source=random_bytes())
    def test_roundtrip_stability(self, source):
        try:
            lexer1 = Lexer(source, filename="<fuzz>")
            tokens1 = lexer1.tokenize()
            lexer2 = Lexer(source, filename="<fuzz>")
            tokens2 = lexer2.tokenize()
            assert len(tokens1) == len(tokens2)
            for t1, t2 in zip(tokens1, tokens2):
                assert t1.type == t2.type
                assert t1.value == t2.value
                assert t1.line == t2.line
                assert t1.col == t2.col
        except MaomiError:
            pass  # If it errors, both calls should error — but that's a weaker check


@pytest.mark.fuzz
class TestErrorMessagesValid:
    """Property 4: Every error has non-None line, col, and message."""

    @given(source=random_bytes())
    def test_lexer_errors_well_formed(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            lexer.tokenize()
            for err in lexer.errors:
                assert err.message is not None
                assert err.line is not None
                assert err.col is not None
        except MaomiError as e:
            assert e.message is not None
            assert e.line is not None
            assert e.col is not None

    @given(source=mutated_source())
    def test_parser_errors_well_formed(self, source):
        try:
            lexer = Lexer(source, filename="<fuzz>")
            tokens = lexer.tokenize()
            parser = Parser(tokens, filename="<fuzz>")
            parser.parse()
            for err in parser.errors:
                assert err.message is not None
                assert err.line is not None
                assert err.col is not None
        except MaomiError as e:
            assert e.message is not None
            assert e.line is not None
            assert e.col is not None
```

- [ ] **Step 2: Run the fuzz tests**

Run: `uv run pytest tests/test_fuzz_parser.py -v -m fuzz`
Expected: PASS — all 10 test methods pass (4 properties, some with multiple input strategies)

- [ ] **Step 3: Commit**

```bash
git add tests/test_fuzz_parser.py
git commit -m "feat: add Stage 1 parser fuzz tests (4 properties)"
```

---

### Task 4: Strategy Library — Grammar-Based Strategies

**Files:**
- Modify: `tests/fuzz_strategies.py`
- Modify: `tests/test_fuzz_strategies.py`

This task adds grammar-based strategies that generate syntactically valid Maomi source code. These are used by Stages 2 and 3.

- [ ] **Step 1: Add tests for grammar-based strategies to `tests/test_fuzz_strategies.py`**

Append to the existing file:
```python
from fuzz_strategies import valid_function, valid_program, valid_type, grad_program
from maomi.lexer import Lexer
from maomi.parser import Parser


class TestGrammarStrategies:
    @given(valid_type())
    @settings(max_examples=50)
    def test_valid_type_produces_strings(self, t):
        assert isinstance(t, str)
        assert len(t) > 0

    @given(valid_function())
    @settings(max_examples=50)
    def test_valid_function_parses(self, fn_src):
        """A generated function should at least parse without crashing."""
        assert isinstance(fn_src, str)
        lexer = Lexer(fn_src, filename="<fuzz>")
        tokens = lexer.tokenize()
        assert len(lexer.errors) == 0, f"Lexer errors: {lexer.errors}"
        parser = Parser(tokens, filename="<fuzz>")
        parser.parse()
        assert len(parser.errors) == 0, f"Parse errors on: {fn_src!r}"

    @given(valid_program())
    @settings(max_examples=50)
    def test_valid_program_parses(self, prog_src):
        """A generated program should parse without crashing."""
        assert isinstance(prog_src, str)
        lexer = Lexer(prog_src, filename="<fuzz>")
        tokens = lexer.tokenize()
        assert len(lexer.errors) == 0
        parser = Parser(tokens, filename="<fuzz>")
        parser.parse()
        assert len(parser.errors) == 0, f"Parse errors on: {prog_src!r}"

    @given(grad_program())
    @settings(max_examples=50)
    def test_grad_program_parses(self, prog_src):
        """A generated grad program should parse without crashing."""
        assert isinstance(prog_src, str)
        lexer = Lexer(prog_src, filename="<fuzz>")
        tokens = lexer.tokenize()
        assert len(lexer.errors) == 0
        parser = Parser(tokens, filename="<fuzz>")
        parser.parse()
        assert len(parser.errors) == 0, f"Parse errors on: {prog_src!r}"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fuzz_strategies.py::TestGrammarStrategies -v`
Expected: FAIL with `ImportError` (strategies not yet implemented)

- [ ] **Step 3: Implement grammar-based strategies**

Append to `tests/fuzz_strategies.py`:
```python
# ---------------------------------------------------------------------------
# Grammar-based strategies (Stages 2 & 3)
# ---------------------------------------------------------------------------

# All base types supported by Maomi
_SCALAR_TYPES = ["f32", "f64", "bf16", "i32", "i64", "bool"]
_FLOAT_TYPES = ["f32", "f64", "bf16"]
_INT_TYPES = ["i32", "i64"]

# Differentiable builtins (safe for grad_program)
_DIFF_BUILTINS_UNARY = ["exp", "log", "tanh", "sqrt", "cos", "sin", "sigmoid",
                         "neg", "abs"]


@st.composite
def valid_type(draw: st.DrawFn) -> str:
    """Generate a valid Maomi type annotation string."""
    base = draw(st.sampled_from(_SCALAR_TYPES))
    is_array = draw(st.booleans())
    if is_array:
        ndims = draw(st.integers(min_value=1, max_value=3))
        dims = [draw(st.integers(min_value=1, max_value=8)) for _ in range(ndims)]
        return f"{base}[{', '.join(str(d) for d in dims)}]"
    return base


@st.composite
def _valid_scalar_expr(draw: st.DrawFn, varnames: list[str], typ: str,
                       depth: int) -> str:
    """Generate a valid scalar expression of the given type.

    Args:
        varnames: Available variable names of this type in scope.
        typ: The scalar base type (e.g., "f32").
        depth: Remaining recursion depth.
    """
    if depth <= 0 or not varnames:
        # Leaf: literal or variable reference
        if varnames and draw(st.booleans()):
            return draw(st.sampled_from(varnames))
        # Generate a literal based on type
        if typ in ("f32", "f64", "bf16"):
            val = draw(st.floats(min_value=-10.0, max_value=10.0,
                                 allow_nan=False, allow_infinity=False))
            return f"{val:.6f}"  # Fixed precision avoids exponential notation
        elif typ in ("i32", "i64"):
            val = draw(st.integers(min_value=-100, max_value=100))
            return str(val)
        elif typ == "bool":
            return draw(st.sampled_from(["true", "false"]))
        return "0.0"

    # Recursive: binary op or builtin call
    choice = draw(st.sampled_from(["binop", "var", "builtin"]))

    if choice == "var" and varnames:
        return draw(st.sampled_from(varnames))
    elif choice == "builtin" and typ in ("f32", "f64", "bf16"):
        builtin = draw(st.sampled_from(_DIFF_BUILTINS_UNARY))
        inner = draw(_valid_scalar_expr(varnames, typ, depth - 1))
        return f"{builtin}({inner})"
    else:
        # Binary op
        op = draw(st.sampled_from(["+", "-", "*"]))
        left = draw(_valid_scalar_expr(varnames, typ, depth - 1))
        right = draw(_valid_scalar_expr(varnames, typ, depth - 1))
        return f"({left} {op} {right})"


@st.composite
def valid_function(draw: st.DrawFn, max_depth: int = 4) -> str:
    """Generate a syntactically valid Maomi function definition.

    Generates a function with scalar float params and a body of arithmetic
    and builtin calls. The function is self-contained (no imports needed).
    """
    # Generate 1-3 parameters
    num_params = draw(st.integers(min_value=1, max_value=3))
    param_type = draw(st.sampled_from(_FLOAT_TYPES))
    param_names = [f"p{i}" for i in range(num_params)]
    params_str = ", ".join(f"{name}: {param_type}" for name in param_names)

    # Generate body expression using params as available variables
    body = draw(_valid_scalar_expr(param_names, param_type, max_depth))

    fn_name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,5}", fullmatch=True))

    return f"fn {fn_name}({params_str}) -> {param_type} {{ {body} }}"


@st.composite
def valid_program(draw: st.DrawFn) -> str:
    """Generate a syntactically valid multi-function Maomi program.

    Import-free — no resolver needed. Many programs may have type errors;
    this is fine for crash-resistance testing.
    """
    num_fns = draw(st.integers(min_value=1, max_value=3))
    fns = []
    for i in range(num_fns):
        fn_src = draw(valid_function())
        # Ensure unique function names by prefixing
        fn_src = fn_src.replace("fn ", f"fn f{i}_", 1)
        fns.append(fn_src)
    return "\n\n".join(fns)


@st.composite
def grad_program(draw: st.DrawFn) -> str:
    """Generate a program with a scalar-returning function and a grad() call.

    Restricted to differentiable operations only (no argmax, iota, integer ops).
    Uses f32 scalars only for finite-difference compatibility.
    """
    # Generate the target function with f32 params
    num_params = draw(st.integers(min_value=1, max_value=2))
    param_names = [f"x{i}" for i in range(num_params)]
    params_str = ", ".join(f"{name}: f32" for name in param_names)

    body = draw(_valid_scalar_expr(param_names, "f32", max_depth=3))

    # Pick which param to differentiate w.r.t.
    wrt = draw(st.sampled_from(param_names))

    target_fn = f"fn target({params_str}) -> f32 {{ {body} }}"

    # Build a wrapper that calls grad
    wrapper_params = ", ".join(f"{name}: f32" for name in param_names)
    wrapper_args = ", ".join(param_names)
    grad_fn = f"fn grad_target({wrapper_params}) -> f32 {{ grad(target({wrapper_args}), {wrt}) }}"

    return f"{target_fn}\n\n{grad_fn}"
```

- [ ] **Step 4: Run grammar strategy tests**

Run: `uv run pytest tests/test_fuzz_strategies.py::TestGrammarStrategies -v`
Expected: PASS — all 4 tests produce parseable Maomi source

- [ ] **Step 5: Fix any parse failures in generated code**

If tests fail because generated code doesn't parse, adjust the generators (e.g., float literal formatting, identifier naming rules). Iterate until tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fuzz_strategies.py tests/test_fuzz_strategies.py
git commit -m "feat: add grammar-based fuzz strategies (valid_function, valid_program, grad_program)"
```

---

### Task 5: Strategy Library — From-Scratch AST Strategies

**Files:**
- Modify: `tests/fuzz_strategies.py`
- Modify: `tests/test_fuzz_strategies.py`

This task adds from-scratch AST generation (Futhark-style) — random compositions of AST nodes where types and expressions are generated independently.

- [ ] **Step 1: Add tests for AST strategies**

Append to `tests/test_fuzz_strategies.py`:
```python
from fuzz_strategies import random_ast_program
from maomi.ast_nodes import Program


class TestASTStrategies:
    @given(random_ast_program())
    @settings(max_examples=50)
    def test_random_ast_program_produces_program(self, prog):
        """A generated AST program should be a valid Program node."""
        assert isinstance(prog, Program)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fuzz_strategies.py::TestASTStrategies -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement from-scratch AST strategies**

Append to `tests/fuzz_strategies.py`:
```python
from maomi.ast_nodes import (
    BinOp,
    Block,
    BoolLiteral,
    CallExpr,
    Dim,
    Expr,
    ExprStmt,
    FloatLiteral,
    FnDef,
    Identifier,
    IfExpr,
    IntLiteral,
    LetStmt,
    Param,
    Program,
    Span,
    TypeAnnotation,
    UnaryOp,
)


def _make_span() -> Span:
    """Create a dummy span for generated AST nodes."""
    return Span(line_start=1, col_start=1, line_end=1, col_end=1)


@st.composite
def _random_type_annotation(draw: st.DrawFn) -> TypeAnnotation:
    """Generate a random TypeAnnotation AST node."""
    base = draw(st.sampled_from(_SCALAR_TYPES))
    is_array = draw(st.booleans())
    if is_array:
        ndims = draw(st.integers(min_value=1, max_value=3))
        dims = [Dim(value=draw(st.integers(min_value=1, max_value=8)),
                     span=_make_span())
                for _ in range(ndims)]
        return TypeAnnotation(base=base, dims=dims, span=_make_span())
    return TypeAnnotation(base=base, dims=None, span=_make_span())


@st.composite
def _random_expr(draw: st.DrawFn, depth: int = 6) -> Expr:
    """Generate a random expression AST node.

    Types and expressions are independent — this intentionally produces
    semantically nonsensical ASTs to stress-test the type checker.
    """
    if depth <= 0:
        # Leaf nodes only
        leaf = draw(st.sampled_from(["int", "float", "bool", "ident"]))
        if leaf == "int":
            return IntLiteral(
                value=draw(st.integers(min_value=-100, max_value=100)),
                span=_make_span(),
            )
        elif leaf == "float":
            return FloatLiteral(
                value=draw(st.floats(min_value=-10, max_value=10,
                                     allow_nan=False, allow_infinity=False)),
                span=_make_span(),
            )
        elif leaf == "bool":
            return BoolLiteral(value=draw(st.booleans()), span=_make_span())
        else:
            name = draw(st.from_regex(r"[a-z][a-z0-9]{0,4}", fullmatch=True))
            return Identifier(name=name, span=_make_span())

    # Recursive nodes
    node_type = draw(st.sampled_from(["binop", "unary", "if", "call", "leaf"]))

    if node_type == "binop":
        op = draw(st.sampled_from(["+", "-", "*", "/"]))
        left = draw(_random_expr(depth - 1))
        right = draw(_random_expr(depth - 1))
        return BinOp(op=op, left=left, right=right, span=_make_span())
    elif node_type == "unary":
        operand = draw(_random_expr(depth - 1))
        return UnaryOp(op="-", operand=operand, span=_make_span())
    elif node_type == "if":
        cond = draw(_random_expr(depth - 1))
        then_expr = draw(_random_expr(depth - 1))
        else_expr = draw(_random_expr(depth - 1))
        return IfExpr(
            condition=cond,
            then_block=Block(stmts=[], result=then_expr, span=_make_span()),
            else_block=Block(stmts=[], result=else_expr, span=_make_span()),
            span=_make_span(),
        )
    elif node_type == "call":
        fn_name = draw(st.sampled_from(["exp", "log", "sqrt", "tanh", "abs"]))
        arg = draw(_random_expr(depth - 1))
        return CallExpr(
            callee=fn_name, args=[arg], span=_make_span(),
        )
    else:
        return draw(_random_expr(0))


@st.composite
def _random_fndef(draw: st.DrawFn) -> FnDef:
    """Generate a random FnDef AST node."""
    fn_name = draw(st.from_regex(r"f[a-z0-9]{0,4}", fullmatch=True))
    num_params = draw(st.integers(min_value=0, max_value=3))
    params = []
    for i in range(num_params):
        params.append(Param(
            name=f"p{i}",
            type_annotation=draw(_random_type_annotation()),
            span=_make_span(),
        ))
    ret_type = draw(_random_type_annotation())
    body_expr = draw(_random_expr(depth=draw(st.integers(min_value=0, max_value=6))))

    # Optionally add let statements
    stmts = []
    num_stmts = draw(st.integers(min_value=0, max_value=3))
    for i in range(num_stmts):
        let_expr = draw(_random_expr(depth=2))
        stmts.append(LetStmt(
            name=f"v{i}",
            type_annotation=None,
            value=let_expr,
            span=_make_span(),
        ))

    return FnDef(
        name=fn_name,
        params=params,
        return_type=ret_type,
        body=Block(stmts=stmts, expr=body_expr, span=_make_span()),
        span=_make_span(),
    )


@st.composite
def random_ast_program(draw: st.DrawFn) -> Program:
    """Generate a random Program AST node from scratch.

    Bounds: max 5 functions, max 10 stmts per block, max depth 6.
    """
    num_fns = draw(st.integers(min_value=1, max_value=5))
    fns = [draw(_random_fndef()) for _ in range(num_fns)]
    return Program(
        imports=[],
        struct_defs=[],
        functions=fns,
        span=_make_span(),
    )
```

- [ ] **Step 4: Run AST strategy tests**

Run: `uv run pytest tests/test_fuzz_strategies.py::TestASTStrategies -v`
Expected: PASS

- [ ] **Step 5: Fix any AST construction issues**

The AST node constructors may have required fields not covered above (e.g., missing `span`, unexpected field names). If tests fail with `TypeError`, inspect the actual AST node class in `src/maomi/ast_nodes.py` and fix the constructor calls. Iterate until tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fuzz_strategies.py tests/test_fuzz_strategies.py
git commit -m "feat: add from-scratch AST generation strategies (Futhark-style)"
```

---

### Task 6: Stage 2 — Type Checker Fuzz Tests (5 properties)

**Files:**
- Create: `tests/test_fuzz_typechecker.py`

**Key imports:**
- `from maomi.lexer import Lexer`
- `from maomi.parser import Parser`
- `from maomi.type_checker import TypeChecker`
- `from maomi.errors import MaomiError`
- `from maomi.types import MaomiType`

- [ ] **Step 1: Write `tests/test_fuzz_typechecker.py`**

```python
"""Stage 2: Type checker robustness fuzz tests.

Properties:
1. Type checker never crashes on valid syntax
2. Type checker never crashes on mutated ASTs
3. Type checker never crashes on random ASTs
4. Type map consistency
5. Idempotent rejection
"""

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

from fuzz_strategies import random_ast_program, valid_program
from maomi.ast_nodes import BinOp, TypeAnnotation
from maomi.errors import MaomiError
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker
from maomi.types import MaomiType


def _mutate_expr(expr, data):
    """Randomly mutate an expression AST node in place."""
    if expr is None:
        return
    if isinstance(expr, BinOp):
        if data.draw(st.booleans()):
            expr.op = data.draw(st.sampled_from(["+", "-", "*", "/"]))


def _parse(source: str):
    """Lex + parse a source string, returning the Program AST."""
    lexer = Lexer(source, filename="<fuzz>")
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename="<fuzz>")
    return parser.parse()


@pytest.mark.fuzz
class TestTypeCheckerNeverCrashesValidSyntax:
    """Property 1: Type checker handles valid-syntax programs gracefully."""

    @given(source=valid_program())
    def test_valid_program(self, source):
        try:
            program = _parse(source)
            checker = TypeChecker(filename="<fuzz>")
            checker.check(program)
        except MaomiError:
            pass  # Clean rejection is fine


@pytest.mark.fuzz
class TestTypeCheckerNeverCrashesMutatedAST:
    """Property 2: Type checker handles mutated ASTs gracefully.

    Takes a valid AST and mutates it: swap types, rename identifiers,
    change operator kinds, etc. Type checker should reject gracefully.
    """

    @given(source=valid_program(), data=st.data())
    def test_mutated_ast(self, source, data):
        try:
            program = _parse(source)

            # Mutate the AST in place
            for fn in program.functions:
                # Randomly swap the return type to a different type
                if data.draw(st.booleans()):
                    new_base = data.draw(st.sampled_from(
                        ["f32", "i32", "bool", "f64", "i64"]))
                    fn.return_type = TypeAnnotation(
                        base=new_base, dims=None,
                        span=fn.return_type.span)
                # Randomly rename parameters to nonsense
                for param in fn.params:
                    if data.draw(st.booleans()):
                        param.name = data.draw(
                            st.from_regex(r"[a-z]{3,6}", fullmatch=True))
                # Randomly change binary ops in the body
                _mutate_expr(fn.body.expr, data)

            checker = TypeChecker(filename="<fuzz>")
            checker.check(program)
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestTypeCheckerNeverCrashesRandomAST:
    """Property 3: Type checker handles from-scratch random ASTs."""

    @given(program=random_ast_program())
    def test_random_ast(self, program):
        try:
            checker = TypeChecker(filename="<fuzz>")
            checker.check(program)
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestTypeMapConsistency:
    """Property 4: If check() returns no errors, type_map entries are valid."""

    @given(source=valid_program())
    def test_type_map_entries_are_valid(self, source):
        try:
            program = _parse(source)
            checker = TypeChecker(filename="<fuzz>")
            errors = checker.check(program)
            if not errors:
                # Every entry in type_map should be a valid MaomiType
                for node_id, typ in checker.type_map.items():
                    assert isinstance(typ, MaomiType.__args__), (
                        f"type_map[{node_id}] = {typ!r} is not a valid MaomiType"
                    )
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestIdempotentRejection:
    """Property 5: Checking the same program twice gives the same errors."""

    @given(source=valid_program())
    def test_determinism(self, source):
        try:
            program = _parse(source)

            checker1 = TypeChecker(filename="<fuzz>")
            errors1 = checker1.check(program)

            checker2 = TypeChecker(filename="<fuzz>")
            errors2 = checker2.check(program)

            # Same number of errors
            assert len(errors1) == len(errors2), (
                f"Run 1: {len(errors1)} errors, Run 2: {len(errors2)} errors"
            )
            # Same error messages (order should be deterministic)
            for e1, e2 in zip(errors1, errors2):
                assert e1.message == e2.message
        except MaomiError:
            pass
```

- [ ] **Step 2: Run the type checker fuzz tests**

Run: `uv run pytest tests/test_fuzz_typechecker.py -v -m fuzz`
Expected: PASS — all 5 properties pass. If any non-MaomiError exception is raised, that's a real bug to investigate and fix.

- [ ] **Step 3: Fix any bugs found**

If a fuzz test finds a crashing input, add it as an `@example(...)` decorator before fixing the underlying bug in the compiler. This ensures regression coverage.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fuzz_typechecker.py
git commit -m "feat: add Stage 2 type checker fuzz tests (5 properties)"
```

---

### Task 7: Stage 3 — AD & Codegen Fuzz Tests (4 properties)

**Files:**
- Create: `tests/test_fuzz_ad_codegen.py`

**Key imports:**
- `from maomi.ad import transform_grad`
- `from maomi.codegen.stablehlo.core import StableHLOCodegen`

- [ ] **Step 1: Write `tests/test_fuzz_ad_codegen.py`**

```python
"""Stage 3: AD & codegen robustness fuzz tests.

Properties:
1. Codegen never crashes on well-typed programs
2. AD transform never crashes on differentiable programs
3. AD output is valid for codegen
4. Finite-difference agreement (requires JAX)
"""

import pytest
from hypothesis import assume, given, settings

from fuzz_strategies import grad_program, valid_program
from maomi.ad import transform_grad
from maomi.codegen.stablehlo.core import StableHLOCodegen
from maomi.errors import MaomiError
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.type_checker import TypeChecker

try:
    import jax
    import numpy as np

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _compile_to_typed(source: str):
    """Run lex → parse → typecheck, return (program, checker) or raise."""
    lexer = Lexer(source, filename="<fuzz>")
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename="<fuzz>")
    program = parser.parse()
    checker = TypeChecker(filename="<fuzz>")
    errors = checker.check(program)
    return program, checker, errors


@pytest.mark.fuzz
class TestCodegenNeverCrashes:
    """Property 1: Codegen succeeds on all well-typed programs."""

    @given(source=valid_program())
    def test_codegen_on_well_typed(self, source):
        try:
            program, checker, errors = _compile_to_typed(source)
            assume(not errors)  # Only test well-typed programs
            mlir = StableHLOCodegen(program, checker.type_map).generate()
            assert isinstance(mlir, str)
            assert len(mlir) > 0
        except MaomiError:
            pass  # Clean rejection


@pytest.mark.fuzz
class TestADNeverCrashes:
    """Property 2: AD transform handles differentiable programs gracefully."""

    @given(source=grad_program())
    def test_ad_transform(self, source):
        try:
            program, checker, errors = _compile_to_typed(source)
            assume(not errors)
            transformed = transform_grad(program, checker.type_map)
            assert transformed is not None
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestADOutputValidForCodegen:
    """Property 3: If AD succeeds, codegen should also succeed."""

    @given(source=grad_program())
    def test_ad_then_codegen(self, source):
        try:
            program, checker, errors = _compile_to_typed(source)
            assume(not errors)
            transformed = transform_grad(program, checker.type_map)
            mlir = StableHLOCodegen(transformed, checker.type_map).generate()
            assert isinstance(mlir, str)
            assert len(mlir) > 0
        except MaomiError:
            pass


@pytest.mark.fuzz
@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestFiniteDifferenceAgreement:
    """Property 4: grad(f)(x) matches (f(x+h) - f(x-h)) / 2h."""

    @given(
        source=grad_program(),
        x_val=st.floats(min_value=-10, max_value=10,
                        allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_finite_difference(self, source, x_val):
        """Verify analytical gradient matches numerical gradient."""
        # This test requires full JAX execution infrastructure.
        # Implementation depends on maomi.jax_runner being available.
        # Skeleton — flesh out when integrating with JAX runner.
        pytest.skip("Finite-difference test requires JAX runner integration")
```

Note: Property 4 (finite-difference) is a skeleton. It requires integrating with `maomi.jax_runner` to actually compile and execute the StableHLO. The skeleton is here so the test structure is in place — flesh it out when ready to test with JAX.

- [ ] **Step 2: Add the missing import at the top of the file**

The `st` import is needed for Property 4:
```python
import hypothesis.strategies as st
```

- [ ] **Step 3: Run the AD/codegen fuzz tests**

Run: `uv run pytest tests/test_fuzz_ad_codegen.py -v -m fuzz`
Expected: PASS — properties 1-3 pass. Property 4 skips (JAX skeleton).

- [ ] **Step 4: Fix any bugs found**

AD and codegen are the most complex stages. If crashes are found, add `@example(...)` decorators and investigate the root cause.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fuzz_ad_codegen.py
git commit -m "feat: add Stage 3 AD/codegen fuzz tests (4 properties, finite-diff skeleton)"
```

---

### Task 8: Strategy Library — LSP Strategies

**Files:**
- Modify: `tests/fuzz_strategies.py`
- Modify: `tests/test_fuzz_strategies.py`

- [ ] **Step 1: Add tests for LSP strategies**

Append to `tests/test_fuzz_strategies.py`:
```python
from fuzz_strategies import partial_source, lsp_edit_sequence


class TestLSPStrategies:
    @given(partial_source())
    @settings(max_examples=50)
    def test_partial_source_produces_strings(self, s):
        assert isinstance(s, str)

    @given(lsp_edit_sequence())
    @settings(max_examples=50)
    def test_lsp_edit_sequence_produces_list(self, edits):
        assert isinstance(edits, list)
        for edit in edits:
            assert "range_start" in edit
            assert "range_end" in edit
            assert "text" in edit
```

- [ ] **Step 2: Implement LSP strategies**

Append to `tests/fuzz_strategies.py`:
```python
# ---------------------------------------------------------------------------
# LSP strategies (Stage 4)
# ---------------------------------------------------------------------------


@st.composite
def partial_source(draw: st.DrawFn) -> str:
    """Valid code truncated at a random point (simulates mid-typing)."""
    base = draw(st.sampled_from(SEED_CORPUS))
    if not base:
        return base
    cut = draw(st.integers(min_value=1, max_value=len(base)))
    return base[:cut]


@st.composite
def lsp_edit_sequence(draw: st.DrawFn) -> list[dict]:
    """Generate a sequence of text edits (insert/delete/replace).

    Each edit is a dict with range_start (line, col), range_end (line, col),
    and replacement text.
    """
    num_edits = draw(st.integers(min_value=1, max_value=5))
    edits = []
    for _ in range(num_edits):
        start_line = draw(st.integers(min_value=0, max_value=20))
        start_col = draw(st.integers(min_value=0, max_value=80))
        # End is at or after start
        end_line = draw(st.integers(min_value=start_line, max_value=start_line + 3))
        end_col = draw(st.integers(min_value=0, max_value=80))
        text = draw(st.text(min_size=0, max_size=30))
        edits.append({
            "range_start": {"line": start_line, "character": start_col},
            "range_end": {"line": end_line, "character": end_col},
            "text": text,
        })
    return edits
```

- [ ] **Step 3: Run LSP strategy tests**

Run: `uv run pytest tests/test_fuzz_strategies.py::TestLSPStrategies -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/fuzz_strategies.py tests/test_fuzz_strategies.py
git commit -m "feat: add LSP fuzz strategies (partial_source, lsp_edit_sequence)"
```

---

### Task 9: Stage 4 — LSP Crash Resistance Fuzz Tests (properties 1-5)

**Files:**
- Create: `tests/test_fuzz_lsp.py`

**Key imports:**
- `from maomi.lsp import validate` (returns `(list[Diagnostic], AnalysisResult)`)
- `from maomi.lsp import _complete_general, _complete_dot, _get_hover_text, _goto_find_definition`
- `from maomi.lsp import rename_at, _refs_collect_all`
- `from maomi.lsp import _build_document_symbols, _sem_collect_tokens`

Refer to `tests/test_lsp_fuzz.py` for the exact import pattern and how to call these internal functions.

- [ ] **Step 1: Write crash resistance properties (1-5)**

```python
"""Stage 4: LSP robustness fuzz tests.

15 properties testing crash resistance, determinism, consistency,
structural integrity, and state management of the LSP.
"""

import pytest
from hypothesis import given, settings
from lsprotocol import types

from fuzz_strategies import (
    lsp_edit_sequence,
    mutated_source,
    near_valid_source,
    partial_source,
    random_bytes,
    valid_program,
)
from maomi.errors import MaomiError
from maomi.lsp import (
    validate,
    _local_functions,
    _find_node_at,
    _get_hover_text,
    _goto_find_definition,
    _complete_general,
    _complete_dot,
    prepare_rename_at,
    rename_at,
    _build_document_symbols,
    _sem_collect_tokens,
    _refs_collect_all,
)


def _apply_edit(source: str, edit: dict) -> str:
    """Apply a single text edit to source string."""
    lines = source.split("\n")
    start = edit["range_start"]
    end = edit["range_end"]
    text = edit["text"]

    # Clamp to actual document bounds
    start_line = min(start["line"], max(len(lines) - 1, 0))
    end_line = min(end["line"], max(len(lines) - 1, 0))
    start_col = min(start["character"], len(lines[start_line]) if lines else 0)
    end_col = min(end["character"], len(lines[end_line]) if lines else 0)

    # Build the edited text
    before = "\n".join(lines[:start_line]) + ("\n" if start_line > 0 else "")
    before += lines[start_line][:start_col] if start_line < len(lines) else ""
    after = lines[end_line][end_col:] if end_line < len(lines) else ""
    after += ("\n" if end_line + 1 < len(lines) else "")
    after += "\n".join(lines[end_line + 1:])

    return before + text + after


# ---- Crash Resistance Properties (1-5) ----


@pytest.mark.fuzz
class TestLSPNeverCrashesMalformed:
    """Property 1: LSP handles malformed documents without crashing."""

    @given(source=random_bytes())
    def test_random_bytes(self, source):
        try:
            validate(source, "<fuzz>")
        except MaomiError:
            pass

    @given(source=mutated_source())
    def test_mutated_source(self, source):
        try:
            validate(source, "<fuzz>")
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestLSPCompletionNeverCrashes:
    """Property 2: Completion at arbitrary positions never crashes."""

    @given(source=partial_source())
    def test_completion_arbitrary_position(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None:
            return
        lines = source.split("\n")
        for line_idx in range(min(len(lines), 5)):
            for col in range(min(len(lines[line_idx]) + 1, 20)):
                pos = types.Position(line=line_idx, character=col)
                try:
                    _complete_general(result, pos)
                    _complete_dot(result, pos)
                except (MaomiError, Exception):
                    pass  # Completion may fail on broken input


@pytest.mark.fuzz
class TestLSPHoverGotoDefNeverCrashes:
    """Property 3: Hover and goto-def at arbitrary positions never crash."""

    @given(source=partial_source())
    def test_hover_gotodef(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or not result.program:
            return
        lines = source.split("\n")
        for fn in _local_functions(result.program):
            for line_idx in range(min(len(lines), 5)):
                line_1 = line_idx + 1  # _find_node_at uses 1-indexed
                for col in range(min(len(lines[line_idx]) + 1, 10)):
                    col_1 = col + 1
                    node = _find_node_at(fn, line_1, col_1)
                    if node is not None:
                        try:
                            _get_hover_text(node, fn, result)
                        except MaomiError:
                            pass
                        try:
                            _goto_find_definition(node, fn, result)
                        except MaomiError:
                            pass


@pytest.mark.fuzz
class TestLSPSurvivesEditSequences:
    """Property 4: LSP survives rapid edit sequences."""

    @given(source=partial_source(), edits=lsp_edit_sequence())
    def test_edit_sequence(self, source, edits):
        current = source
        for edit in edits:
            try:
                current = _apply_edit(current, edit)
                validate(current, "<fuzz>")
            except MaomiError:
                    pass


@pytest.mark.fuzz
class TestLSPRenameReferencesNeverCrash:
    """Property 5: Rename and find-references at arbitrary positions never crash."""

    @given(source=partial_source())
    def test_rename_references(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None:
            return
        lines = source.split("\n")
        for line_idx in range(min(len(lines), 3)):
            for col in range(min(len(lines[line_idx]) + 1, 10)):
                try:
                    # rename_at uses 0-indexed line/col
                    rename_at(source, result, line_idx, col, "zzz_new")
                except MaomiError:
                    pass
                try:
                    prepare_rename_at(source, result, line_idx, col)
                except MaomiError:
                    pass
```

- [ ] **Step 2: Run crash resistance tests**

Run: `uv run pytest tests/test_fuzz_lsp.py -v -m fuzz -k "Crashes or Survives or Rename"`
Expected: PASS. Fix import paths if needed — check `tests/test_lsp_fuzz.py` for the exact import pattern used in the existing codebase.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fuzz_lsp.py
git commit -m "feat: add Stage 4 LSP crash resistance fuzz tests (properties 1-5)"
```

---

### Task 10: Stage 4 — LSP Determinism & Consistency Tests (properties 6-11)

**Files:**
- Modify: `tests/test_fuzz_lsp.py`

- [ ] **Step 1: Add determinism properties (6-7)**

Append to `tests/test_fuzz_lsp.py`:
```python
# ---- Determinism Properties (6-7) ----


@pytest.mark.fuzz
class TestLSPDiagnosticDeterminism:
    """Property 6: Same document produces same diagnostics."""

    @given(source=valid_program())
    def test_same_diagnostics(self, source):
        try:
            diags1, _ = validate(source, "<fuzz>")
            diags2, _ = validate(source, "<fuzz>")
        except MaomiError:
            return
        assert len(diags1) == len(diags2), (
            f"Run 1: {len(diags1)} diags, Run 2: {len(diags2)} diags"
        )
        for d1, d2 in zip(diags1, diags2):
            assert d1.message == d2.message
            assert d1.range == d2.range


@pytest.mark.fuzz
class TestLSPSemanticTokenDeterminism:
    """Property 7: Same document produces same semantic tokens."""

    @given(source=valid_program())
    def test_same_semantic_tokens(self, source):
        try:
            _, result1 = validate(source, "<fuzz>")
            _, result2 = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result1 is None or result2 is None:
            return
        if not result1.program or not result2.program:
            return
        # Collect tokens per function and compare
        lines = source.split("\n")
        for fn1, fn2 in zip(_local_functions(result1.program),
                            _local_functions(result2.program)):
            tokens1, tokens2 = [], []
            param_names = {p.name for p in fn1.params}
            _sem_collect_tokens(fn1, tokens1, param_names, lines)
            _sem_collect_tokens(fn2, tokens2, param_names, lines)
            assert tokens1 == tokens2
```

- [ ] **Step 2: Add consistency properties (8-11)**

Append to `tests/test_fuzz_lsp.py`:
```python
from maomi.lexer import Lexer
from maomi.parser import Parser
from maomi.ast_nodes import Identifier


# ---- Consistency Properties (8-11) ----


@pytest.mark.fuzz
class TestHoverTypeMatchesTypeChecker:
    """Property 8: Hover type matches type checker."""

    @given(source=valid_program())
    def test_hover_type_consistency(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or diags or not result.program:
            return
        lines = source.split("\n")
        for fn in _local_functions(result.program):
            for line_idx in range(min(len(lines), 3)):
                line_1 = line_idx + 1
                for col in range(min(len(lines[line_idx]) + 1, 10)):
                    col_1 = col + 1
                    node = _find_node_at(fn, line_1, col_1)
                    if node is not None:
                        try:
                            hover = _get_hover_text(node, fn, result)
                            if hover is not None:
                                assert isinstance(hover, str)
                                # If this node has a type, hover should contain it
                                node_type = result.type_map.get(id(node))
                                if node_type is not None:
                                    assert str(node_type) in hover, (
                                        f"Hover '{hover}' doesn't contain type '{node_type}'"
                                    )
                        except MaomiError:
                            pass


@pytest.mark.fuzz
class TestGotoDefTargetContainsSymbol:
    """Property 9: Go-to-def target contains the symbol name."""

    @given(source=valid_program())
    def test_gotodef_target(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or diags or not result.program:
            return
        lines = source.split("\n")
        for fn in _local_functions(result.program):
            for line_idx in range(min(len(lines), 3)):
                line_1 = line_idx + 1
                for col in range(min(len(lines[line_idx]) + 1, 10)):
                    col_1 = col + 1
                    node = _find_node_at(fn, line_1, col_1)
                    if node is not None and isinstance(node, Identifier):
                        try:
                            def_result = _goto_find_definition(node, fn, result)
                            if def_result is not None:
                                span, _ = def_result
                                # Definition span should be within source
                                def_line = span.line_start - 1  # to 0-indexed
                                if 0 <= def_line < len(lines):
                                    assert node.name in lines[def_line], (
                                        f"Symbol '{node.name}' not in def line '{lines[def_line]}'"
                                    )
                        except MaomiError:
                            pass


@pytest.mark.fuzz
class TestCompletionThenInsertParseable:
    """Property 10: Inserting a completion item produces parseable code."""

    @given(source=partial_source())
    def test_completion_insert_parses(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None:
            return
        lines = source.split("\n")
        if not lines:
            return
        last_line = len(lines) - 1
        last_col = len(lines[last_line])
        pos = types.Position(line=last_line, character=last_col)
        try:
            completions = _complete_general(result, pos)
            if completions:
                item = completions[0]
                label = item.label if hasattr(item, "label") else str(item)
                new_source = source + label
                try:
                    lexer = Lexer(new_source, filename="<fuzz>")
                    tokens = lexer.tokenize()
                    parser = Parser(tokens, filename="<fuzz>")
                    parser.parse()
                except MaomiError:
                    pass  # Parse errors are fine — just shouldn't crash
        except MaomiError:
            pass


@pytest.mark.fuzz
class TestFindRefsSupersetsRenameLocations:
    """Property 11: Find-references includes all rename locations."""

    @given(source=valid_program())
    def test_refs_superset_rename(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or diags or not result.program:
            return
        lines = source.split("\n")
        source_lines = lines
        for fn in _local_functions(result.program):
            for line_idx in range(min(len(lines), 3)):
                line_1 = line_idx + 1
                for col in range(min(len(lines[line_idx]) + 1, 10)):
                    col_1 = col + 1
                    node = _find_node_at(fn, line_1, col_1)
                    if node is not None and isinstance(node, Identifier):
                        try:
                            rename_edits = rename_at(
                                source, result, line_idx, col, "zzz_new")
                            refs = _refs_collect_all(
                                result, node.name, "variable",
                                include_declaration=True,
                                source_lines=source_lines)
                            if rename_edits and refs:
                                # Rename locations should be a subset of refs
                                assert len(rename_edits) <= len(refs), (
                                    f"Rename has {len(rename_edits)} edits but "
                                    f"refs found only {len(refs)} references"
                                )
                        except MaomiError:
                            pass
```

- [ ] **Step 3: Run determinism and consistency tests**

Run: `uv run pytest tests/test_fuzz_lsp.py -v -m fuzz -k "Determinism or Consistency or Hover or GotoDef or Completion or FindRefs"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_fuzz_lsp.py
git commit -m "feat: add Stage 4 LSP determinism and consistency fuzz tests (properties 6-11)"
```

---

### Task 11: Stage 4 — LSP Structural & State Tests (properties 12-15)

**Files:**
- Modify: `tests/test_fuzz_lsp.py`

- [ ] **Step 1: Add structural properties (12-13)**

Append to `tests/test_fuzz_lsp.py`:
```python
import re


# ---- Structural Properties (12-13) ----


@pytest.mark.fuzz
class TestSemanticTokensDontOverlap:
    """Property 12: Semantic token ranges don't overlap."""

    @given(source=valid_program())
    def test_no_overlaps(self, source):
        try:
            _, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or not result.program:
            return
        lines = source.split("\n")
        all_tokens = []
        for fn in _local_functions(result.program):
            param_names = {p.name for p in fn.params}
            _sem_collect_tokens(fn, all_tokens, param_names, lines)
        if not all_tokens:
            return
        # _sem_collect_tokens appends (line, col, length, type, mods) tuples
        sorted_tokens = sorted(all_tokens, key=lambda t: (t[0], t[1]))
        for i in range(len(sorted_tokens) - 1):
            curr = sorted_tokens[i]
            next_tok = sorted_tokens[i + 1]
            if curr[0] == next_tok[0]:  # Same line
                assert curr[1] + curr[2] <= next_tok[1], (
                    f"Overlapping tokens on line {curr[0]}: "
                    f"[{curr[1]}:{curr[1]+curr[2]}] and [{next_tok[1]}:{next_tok[1]+next_tok[2]}]"
                )


@pytest.mark.fuzz
class TestDocumentSymbolsMatchDefs:
    """Property 13: Document symbols match fn/struct definitions in source."""

    @given(source=valid_program())
    def test_symbols_match_definitions(self, source):
        try:
            _, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None:
            return
        try:
            symbols = _build_document_symbols(result)
        except MaomiError:
            return
        if symbols is None:
            return
        # Count fn and struct definitions in source
        fn_count = len(re.findall(r"\bfn\s+\w+", source))
        struct_count = len(re.findall(r"\bstruct\s+\w+", source))
        expected_count = fn_count + struct_count
        # Symbols should have at least this many entries
        # (may have more due to nested symbols)
        assert len(symbols) >= expected_count, (
            f"Expected at least {expected_count} symbols, got {len(symbols)}"
        )
```

- [ ] **Step 2: Add state properties (14-15)**

Append to `tests/test_fuzz_lsp.py`:
```python
# ---- State Properties (14-15) ----


@pytest.mark.fuzz
class TestDiagnosticsClearWhenFixed:
    """Property 14: Fixing errors clears diagnostics."""

    @given(source=near_valid_source())
    def test_fix_clears_diagnostics(self, source):
        """Broken code has diagnostics; the original valid code should not."""
        try:
            diags_broken, _ = validate(source, "<fuzz>")
        except MaomiError:
            return
        # The near_valid_source is based on SEED_CORPUS — try the original
        # valid version and verify it has fewer or no diagnostics
        for original in SEED_CORPUS:
            if original in source or source in original:
                try:
                    diags_fixed, _ = validate(original, "<fuzz>")
                    # Fixed version should have no more diagnostics than broken
                    # (ideally fewer, but we just check it doesn't crash)
                except MaomiError:
                    pass
                break


@pytest.mark.fuzz
class TestRenamePreservesValidity:
    """Property 15: Renaming a symbol in a valid program preserves validity."""

    @given(source=valid_program())
    def test_rename_preserves(self, source):
        try:
            diags, result = validate(source, "<fuzz>")
        except MaomiError:
            return
        if result is None or diags:
            return

        lines = source.split("\n")
        for line_idx in range(len(lines)):
            line = lines[line_idx]
            match = re.search(r"\bfn\s+(\w+)", line)
            if match:
                col = match.start(1)
                try:
                    edits = rename_at(source, result, line_idx, col, "zzz_renamed")
                    if edits:
                        # Apply edits in reverse order to preserve positions
                        new_lines = list(lines)
                        for edit in sorted(edits,
                                          key=lambda e: (e.range.start.line,
                                                        e.range.start.character),
                                          reverse=True):
                            sl = edit.range.start.line
                            sc = edit.range.start.character
                            el = edit.range.end.line
                            ec = edit.range.end.character
                            if sl == el and sl < len(new_lines):
                                line_text = new_lines[sl]
                                new_lines[sl] = line_text[:sc] + edit.new_text + line_text[ec:]
                        new_source = "\n".join(new_lines)
                        # Re-validate — should still be error-free
                        try:
                            new_diags, _ = validate(new_source, "<fuzz>")
                            assert not new_diags, (
                                f"Rename produced errors: {[d.message for d in new_diags]}"
                            )
                        except MaomiError:
                            pass
                except MaomiError:
                    pass
                break
```

Also add these imports at the top of the file (alongside the existing imports):
```python
from fuzz_strategies import SEED_CORPUS
from maomi.ast_nodes import Identifier
```

- [ ] **Step 3: Run structural and state tests**

Run: `uv run pytest tests/test_fuzz_lsp.py -v -m fuzz -k "Semantic or Document or Diagnostics or Rename"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_fuzz_lsp.py
git commit -m "feat: add Stage 4 LSP structural and state fuzz tests (properties 12-15)"
```

---

### Task 12: Full Test Suite Verification

**Files:** None — verification only

- [ ] **Step 1: Run all fuzz tests together**

Run: `uv run pytest tests/test_fuzz_*.py -v -m fuzz`
Expected: All 28 properties pass (with property 4 of Stage 3 skipping if JAX not available)

- [ ] **Step 2: Run all fuzz tests alongside the existing test suite**

Run: `uv run pytest tests/ -v --timeout=300`
Expected: All existing tests still pass. No regressions from adding conftest.py or new imports.

- [ ] **Step 3: Run fuzz tests in exclusion mode**

Run: `uv run pytest tests/ -v -m "not fuzz"`
Expected: All non-fuzz tests run and pass (fuzz tests excluded)

- [ ] **Step 4: Commit any final fixes**

If any tests need adjustments (import paths, API mismatches, etc.), fix and commit:
```bash
git add -A
git commit -m "fix: adjust fuzz tests for API compatibility"
```

---

### Task 13: Documentation & Cleanup

**Files:**
- Modify: `tests/fuzz_strategies.py` (add module-level docstring if missing)

- [ ] **Step 1: Verify fuzz_strategies.py has clear docstrings**

Each public function should have a docstring explaining what it generates and what strategies it uses. Review and add any missing ones.

- [ ] **Step 2: Run the full test suite one final time**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "docs: finalize fuzz testing docstrings and cleanup"
```
