"""ANSI-colored source-snippet renderer for Maomi compiler errors.

Produces codespan-reporting style output matching Gleam/Rust conventions:

    error: Type mismatch
      --> file.mao:3:5
       |
     3 |     a + b
       |     ^^^^^ expected f32, got i32
       |
       = hint: Use cast(b, f32) to convert
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .errors import MaomiError

# ---------------------------------------------------------------------------
# ANSI escape codes
# ---------------------------------------------------------------------------
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _c(code: str, text: str, use_color: bool) -> str:
    """Wrap *text* in an ANSI escape sequence when color is enabled."""
    if use_color:
        return f"{code}{text}{RESET}"
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_error(
    error: "MaomiError",
    source: str | None = None,
    use_color: bool | None = None,
) -> str:
    """Render a ``MaomiError`` as a rich source-snippet string.

    Parameters
    ----------
    error:
        The compiler error to render.
    source:
        Full source text of the file that produced the error.  When ``None``
        the renderer falls back to a compact one-line format.
    use_color:
        Force color on/off.  ``None`` means auto-detect via
        ``sys.stderr.isatty()``.
    """
    if use_color is None:
        use_color = sys.stderr.isatty()

    severity: str = getattr(error, "severity", "error")
    hint: str | None = getattr(error, "hint", None)
    secondary_labels: list = getattr(error, "secondary_labels", [])

    filename: str = error.filename
    line: int = error.line
    col: int = error.col
    col_end: int = error.col_end
    message: str = error.message

    # ----- Fallback: no source available or line out of range -----
    if source is None:
        return _render_fallback(severity, message, filename, line, col, use_color)

    source_lines = source.splitlines()

    if line < 1 or line > len(source_lines):
        return _render_fallback(severity, message, filename, line, col, use_color)

    # ----- Collect all line numbers we need to display -----
    display_lines: list[int] = [line]
    for label in secondary_labels:
        sec_line = label.get("line", 0)
        if 1 <= sec_line <= len(source_lines):
            display_lines.append(sec_line)
    display_lines = sorted(set(display_lines))

    # Gutter width based on the largest line number shown
    max_line_no = max(display_lines)
    gutter_w = len(str(max_line_no))

    parts: list[str] = []

    # ----- Title line -----
    parts.append(_title(severity, message, use_color))

    # ----- Location line -----
    pad = " " * gutter_w
    parts.append(f"  {_c(BLUE, pad + '-->', use_color)} {filename}:{line}:{col}")

    # ----- Source sections -----
    # We emit each display line in order, with separators between
    # non-contiguous lines.
    prev_line_no: int | None = None
    for disp_line in display_lines:
        # Separator between non-contiguous lines
        if prev_line_no is not None and disp_line != prev_line_no + 1:
            parts.append(f"   {_c(BLUE, pad + '|', use_color)}")

        # Blank separator before section
        if prev_line_no is None:
            parts.append(f"   {_c(BLUE, pad + '|', use_color)}")

        src = source_lines[disp_line - 1]
        line_no_str = str(disp_line).rjust(gutter_w)
        parts.append(
            f" {_c(BLUE, line_no_str + ' |', use_color)} {src}"
        )

        # Underline(s) for this display line
        underlines = _underlines_for_line(
            disp_line, line, col, col_end, message,
            secondary_labels, src, gutter_w, severity, use_color,
        )
        parts.extend(underlines)

        prev_line_no = disp_line

    # Closing gutter
    parts.append(f"   {_c(BLUE, pad + '|', use_color)}")

    # ----- Hint -----
    if hint is not None:
        parts.append(f"   {_c(BLUE, pad + '=', use_color)} hint: {hint}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _title(severity: str, message: str, use_color: bool) -> str:
    if severity == "warning":
        return _c(BOLD + YELLOW, f"warning: {message}", use_color)
    return _c(BOLD + RED, f"error: {message}", use_color)


def _render_fallback(
    severity: str,
    message: str,
    filename: str,
    line: int,
    col: int,
    use_color: bool,
) -> str:
    label = "warning" if severity == "warning" else "error"
    return f"{label}: {message} (at {filename}:{line}:{col})"


def _make_underline(col: int, col_end: int, line_len: int, char: str) -> str:
    """Build a single underline string (e.g. ``    ^^^`` or ``  ---``)."""
    col_start = max(col - 1, 0)
    col_stop = max(col_end - 1, col_start + 1)
    col_stop = min(col_stop, line_len)
    col_start = min(col_start, line_len)
    span_len = max(col_stop - col_start, 1)
    return " " * col_start + char * span_len


def _underlines_for_line(
    disp_line: int,
    primary_line: int,
    primary_col: int,
    primary_col_end: int,
    primary_msg: str,
    secondary_labels: list,
    src: str,
    gutter_w: int,
    severity: str,
    use_color: bool,
) -> list[str]:
    """Return underline annotation lines for a single source line."""
    pad = " " * gutter_w
    results: list[str] = []
    line_len = len(src)

    # Primary underline
    if disp_line == primary_line:
        underline = _make_underline(primary_col, primary_col_end, line_len, "^")
        label_text = f"{underline} {primary_msg}"
        color = YELLOW if severity == "warning" else RED
        results.append(
            f"   {_c(BLUE, pad + '|', use_color)} {_c(color, label_text, use_color)}"
        )

    # Secondary underlines
    for sec in secondary_labels:
        if sec.get("line", 0) != disp_line:
            continue
        sec_col = sec.get("col", 1)
        sec_col_end = sec.get("col_end", sec_col + 1)
        sec_msg = sec.get("message", "")

        underline = _make_underline(sec_col, sec_col_end, line_len, "-")
        label_text = f"{underline} {sec_msg}"
        results.append(
            f"   {_c(BLUE, pad + '|', use_color)} {_c(BLUE, label_text, use_color)}"
        )

    return results
