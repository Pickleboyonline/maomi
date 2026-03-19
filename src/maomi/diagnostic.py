from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class Level(Enum):
    ERROR = "error"
    WARNING = "warning"
    HINT = "hint"


@dataclass
class Label:
    """A labeled source span (primary or secondary)."""

    text: str
    filename: str
    line: int
    col: int
    col_end: int


@dataclass
class Diagnostic:
    """A rich compiler diagnostic following Gleam/codespan-reporting conventions."""

    title: str
    text: str
    level: Level
    filename: str
    line: int
    col: int
    col_end: int
    secondary_labels: list[Label] = field(default_factory=list)
    hint: str | None = None


def from_error(error, source: str | None = None) -> Diagnostic:
    """Convert a MaomiError to a Diagnostic."""
    title = _extract_title(error.message)
    level = Level.WARNING if getattr(error, "severity", "error") == "warning" else Level.ERROR

    secondary = []
    for lab in getattr(error, "secondary_labels", []):
        if isinstance(lab, Label):
            secondary.append(lab)
        elif isinstance(lab, (tuple, list)) and len(lab) >= 5:
            secondary.append(
                Label(text=lab[0], filename=lab[1], line=lab[2], col=lab[3], col_end=lab[4])
            )

    diag = Diagnostic(
        title=title,
        text=error.message,
        level=level,
        filename=error.filename,
        line=error.line,
        col=error.col,
        col_end=getattr(error, "col_end", error.col + 1),
        secondary_labels=secondary,
        hint=getattr(error, "hint", None),
    )

    if source is not None:
        diag = enrich(diag, source)

    return diag


def _extract_title(message: str) -> str:
    """Extract a short title from an error message."""
    if message.startswith("undefined variable"):
        return "Undefined variable"
    if message.startswith("undefined function") or "unknown function" in message:
        return "Unknown function"
    if message.startswith("return type"):
        return "Return type mismatch"
    if "type mismatch" in message.lower() or "mismatched types" in message.lower():
        return "Type mismatch"
    if message.startswith("unknown struct") or "no struct" in message:
        return "Unknown struct"
    if "expects" in message and "arg" in message:
        return "Incorrect argument count"
    if message.startswith("duplicate"):
        return "Duplicate definition"
    if "shape" in message.lower():
        return "Shape error"
    if message.startswith("expected"):
        return "Syntax error"
    if "import" in message.lower():
        return "Import error"
    if "field" in message.lower() and ("no " in message.lower() or "unknown" in message.lower()):
        return "Unknown field"
    # Fallback: capitalize first phrase before colon
    colon_idx = message.find(":")
    if 0 < colon_idx < 40:
        return message[:colon_idx].strip().capitalize()
    return "Error" if len(message) > 60 else message[:60]


_SHAPE_PATTERN = re.compile(r"f32\[([^\]]+)\].*f32\[([^\]]+)\]")
_SCALAR_ARRAY_PATTERN = re.compile(r"(f32|f64|i32|i64)\b.*\b(f32|f64|i32|i64)\[")
_TYPE_MISMATCH_PATTERN = re.compile(r"expected (f32|f64|i32|i64|bool),?\s+got (f32|f64|i32|i64|bool)")


def enrich(diag: Diagnostic, source: str) -> Diagnostic:
    """Apply contextual enrichment based on error message content."""
    msg = diag.text

    # Shape mismatch with broadcast potential
    m = _SHAPE_PATTERN.search(msg)
    if m and diag.hint is None:
        s1, s2 = m.group(1), m.group(2)
        if s1 != s2:
            dims1 = [d.strip() for d in s1.split(",")]
            dims2 = [d.strip() for d in s2.split(",")]
            if len(dims1) != len(dims2):
                diag.hint = (
                    f"Shapes [{s1}] and [{s2}] have different ranks. "
                    f"Broadcasting requires matching ranks (with size-1 dims for expansion)."
                )

    # Scalar vs array mismatch
    if _SCALAR_ARRAY_PATTERN.search(msg) and diag.hint is None:
        diag.hint = "A scalar and an array can be combined using broadcast(scalar, dims...)."

    # Type conversion hint
    m2 = _TYPE_MISMATCH_PATTERN.search(msg)
    if m2 and diag.hint is None:
        expected, got = m2.group(1), m2.group(2)
        if expected != got:
            diag.hint = f"Use cast(expr, {expected}) to convert from {got} to {expected}."

    return diag
