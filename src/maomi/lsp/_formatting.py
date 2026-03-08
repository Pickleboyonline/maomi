from __future__ import annotations

from pygls.lsp.server import LanguageServer
from lsprotocol import types

from ._core import server, _cache


def _compute_brace_depth(lines: list[str], target_line: int) -> int:
    """Count net { minus } through all lines before *target_line*."""
    depth = 0
    for i in range(target_line):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return max(0, depth)


def _indent_edit(
    line_0: int, current_line: str, expected_indent: str
) -> types.TextEdit | None:
    """Return a TextEdit replacing leading whitespace, or *None* if already correct."""
    stripped = current_line.lstrip()
    actual_len = len(current_line) - len(stripped)
    if current_line[:actual_len] == expected_indent:
        return None
    return types.TextEdit(
        range=types.Range(
            start=types.Position(line=line_0, character=0),
            end=types.Position(line=line_0, character=actual_len),
        ),
        new_text=expected_indent,
    )


def _on_type_format(
    source: str, line_0: int, col_0: int, ch: str
) -> list[types.TextEdit]:
    lines = source.splitlines()
    if line_0 >= len(lines):
        return []
    current_line = lines[line_0]
    edits: list[types.TextEdit] = []

    if ch == "}":
        # Fix indentation of closing brace.
        stripped = current_line.lstrip()
        if stripped.startswith("}"):
            depth = max(0, _compute_brace_depth(lines, line_0) - 1)
            edit = _indent_edit(line_0, current_line, "    " * depth)
            if edit is not None:
                edits.append(edit)

    elif ch == ";":
        # Remove trailing whitespace.
        rstripped = current_line.rstrip()
        if len(rstripped) < len(current_line):
            edits.append(
                types.TextEdit(
                    range=types.Range(
                        start=types.Position(
                            line=line_0, character=len(rstripped)
                        ),
                        end=types.Position(
                            line=line_0, character=len(current_line)
                        ),
                    ),
                    new_text="",
                )
            )

    elif ch == "\n":
        # Auto-indent new line based on brace depth.
        if line_0 > 0:
            depth = _compute_brace_depth(lines, line_0)
            edit = _indent_edit(line_0, current_line, "    " * depth)
            if edit is not None:
                edits.append(edit)

    return edits


def _find_matching_brace(source: str, line_0: int, col_0: int) -> types.Position | None:
    """Return position of matching brace for the brace at/near (line_0, col_0)."""
    lines = source.splitlines()
    if line_0 >= len(lines):
        return None
    line_text = lines[line_0]

    # Find brace at or near cursor
    ch = line_text[col_0] if col_0 < len(line_text) else None
    if ch not in ('{', '}'):
        # Check adjacent position (cursor right after brace)
        if col_0 > 0 and col_0 - 1 < len(line_text) and line_text[col_0 - 1] in ('{', '}'):
            col_0 -= 1
            ch = line_text[col_0]
        else:
            return None

    if ch == '{':
        # Search forward for matching '}'
        depth = 0
        for i in range(line_0, len(lines)):
            start_col = col_0 if i == line_0 else 0
            for j in range(start_col, len(lines[i])):
                c = lines[i][j]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return types.Position(line=i, character=j)
    elif ch == '}':
        # Search backward for matching '{'
        depth = 0
        for i in range(line_0, -1, -1):
            end_col = col_0 if i == line_0 else len(lines[i]) - 1
            for j in range(end_col, -1, -1):
                if j >= len(lines[i]):
                    continue
                c = lines[i][j]
                if c == '}':
                    depth += 1
                elif c == '{':
                    depth -= 1
                    if depth == 0:
                        return types.Position(line=i, character=j)
    return None


def _format_line_content(stripped: str) -> str:
    """Format a single line's content (already stripped)."""
    if stripped.startswith("///"):
        if len(stripped) > 3 and stripped[3] != " ":
            return "/// " + stripped[3:]
    elif stripped.startswith("//"):
        if len(stripped) > 2 and stripped[2] != " ":
            return "// " + stripped[2:]
    return stripped


def _format_document(source: str) -> list[types.TextEdit]:
    """Format a .mao source string, returning a list of TextEdits."""
    lines = source.splitlines()
    formatted_lines: list[str] = []
    depth = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        # Count leading } characters for this line's indent
        temp = stripped
        leading_closes = 0
        while temp.startswith("}"):
            leading_closes += 1
            temp = temp[1:].lstrip()

        # This line's depth = depth - leading_closes
        line_depth = max(0, depth - leading_closes)
        content = _format_line_content(stripped)
        formatted_lines.append("    " * line_depth + content)

        # Update depth for next line
        depth += stripped.count("{") - stripped.count("}")
        depth = max(0, depth)

    result = "\n".join(formatted_lines)
    # Ensure single trailing newline
    result = result.rstrip("\n") + "\n"

    if result == source:
        return []

    # Single edit replacing entire document
    line_count = len(lines)
    return [types.TextEdit(
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=line_count, character=0),
        ),
        new_text=result,
    )]


@server.feature(
    types.TEXT_DOCUMENT_ON_TYPE_FORMATTING,
    types.DocumentOnTypeFormattingOptions(
        first_trigger_character="}",
        more_trigger_character=[";", "\n"],
    ),
)
def on_type_formatting(
    ls: LanguageServer, params: types.DocumentOnTypeFormattingParams
):
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    line = params.position.line  # already 0-indexed
    col = params.position.character
    return _on_type_format(doc.source, line, col, params.ch)


@server.feature("maomi/matchingBrace")
def matching_brace(ls: LanguageServer, params):
    uri = params.get("textDocument", {}).get("uri")
    position = params.get("position", {})
    line = position.get("line", 0)
    col = position.get("character", 0)
    doc = ls.workspace.get_text_document(uri)
    result = _find_matching_brace(doc.source, line, col)
    if result is None:
        return None
    return {"line": result.line, "character": result.character}


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
def document_formatting(
    ls: LanguageServer, params: types.DocumentFormattingParams
):
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    return _format_document(doc.source)
