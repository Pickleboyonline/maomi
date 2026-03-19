"""LSP response validation utilities.

Inspired by rust-analyzer's test infrastructure:
- `assert_completion_valid`: checks LSP spec invariants on every completion item
- `check_edit`: applies a completion's edits to source and asserts the result

These catch protocol-level bugs that unit tests miss but real editors enforce
(e.g., VS Code filtering out items whose text_edit range doesn't contain the cursor).
"""

from __future__ import annotations

from lsprotocol import types


# ---------------------------------------------------------------------------
# Completion item invariant checks (rust-analyzer's get_all_items pattern)
# ---------------------------------------------------------------------------

def assert_completion_valid(item: types.CompletionItem, cursor: types.Position) -> None:
    """Validate a single CompletionItem against LSP spec rules.

    Checks enforced by VS Code that will silently hide items if violated:
    1. text_edit range must be single-line
    2. text_edit range must be on the same line as the cursor
    3. text_edit range must contain the cursor position
    4. additional_text_edits ranges must not overlap with text_edit range
    """
    if item.text_edit is not None:
        r = _get_edit_range(item.text_edit)

        # Single-line range (VS Code marks multi-line as isInvalid)
        assert r.start.line == r.end.line, (
            f"completion '{item.label}': text_edit range spans multiple lines "
            f"({r.start.line}:{r.start.character} - {r.end.line}:{r.end.character})"
        )

        # Same line as cursor (VS Code marks different-line as isInvalid)
        assert r.start.line == cursor.line, (
            f"completion '{item.label}': text_edit range line {r.start.line} "
            f"!= cursor line {cursor.line}"
        )

        # Range must contain cursor position (VS Code uses range for filtering)
        assert r.start.character <= cursor.character <= r.end.character, (
            f"completion '{item.label}': text_edit range "
            f"[{r.start.character}, {r.end.character}] does not contain "
            f"cursor at character {cursor.character}"
        )

    # Validate additional_text_edits
    if item.additional_text_edits:
        for i, edit in enumerate(item.additional_text_edits):
            # Each must be single-line
            assert edit.range.start.line == edit.range.end.line, (
                f"completion '{item.label}': additional_text_edits[{i}] "
                f"spans multiple lines"
            )

            # Must not overlap with text_edit range
            if item.text_edit is not None:
                te_range = _get_edit_range(item.text_edit)
                if edit.range.start.line == te_range.start.line:
                    edits_disjoint = (
                        edit.range.end.character <= te_range.start.character
                        or edit.range.start.character >= te_range.end.character
                    )
                    assert edits_disjoint, (
                        f"completion '{item.label}': additional_text_edits[{i}] "
                        f"range [{edit.range.start.character}, {edit.range.end.character}) "
                        f"overlaps with text_edit range "
                        f"[{te_range.start.character}, {te_range.end.character})"
                    )


def assert_all_completions_valid(
    comp: types.CompletionList | None, cursor: types.Position,
) -> None:
    """Validate all items in a CompletionList. Call this on every completion test."""
    if comp is None:
        return
    for item in comp.items:
        assert_completion_valid(item, cursor)


# ---------------------------------------------------------------------------
# Edit application (rust-analyzer's check_edit pattern)
# ---------------------------------------------------------------------------

def check_edit(
    source: str,
    cursor: types.Position,
    comp: types.CompletionList,
    label: str,
    expected: str,
) -> None:
    """Select a completion by label, apply its edits to source, assert the result.

    This simulates exactly what the editor does when the user accepts a completion:
    1. Apply text_edit (or insert insert_text at cursor)
    2. Apply additional_text_edits
    3. Compare resulting source to expected

    Snippet placeholders ($0, $1, etc.) are stripped before comparison.
    """
    import re

    # Find the item
    matches = [i for i in comp.items if i.label == label]
    assert matches, (
        f"completion '{label}' not found. "
        f"Available: {[i.label for i in comp.items[:20]]}..."
    )
    item = matches[0]

    # Validate invariants first
    assert_completion_valid(item, cursor)

    lines = source.splitlines(keepends=True)
    # Pad lines if source doesn't end with newline
    while len(lines) <= cursor.line:
        lines.append("")

    # Collect all edits: (line, start_col, end_col, new_text)
    edits: list[tuple[int, int, int, str]] = []

    if item.text_edit is not None:
        r = _get_edit_range(item.text_edit)
        text = _get_edit_new_text(item.text_edit)
        edits.append((r.start.line, r.start.character, r.end.character, text))
    elif item.insert_text is not None:
        edits.append((cursor.line, cursor.character, cursor.character, item.insert_text))
    else:
        edits.append((cursor.line, cursor.character, cursor.character, item.label))

    if item.additional_text_edits:
        for edit in item.additional_text_edits:
            r = edit.range
            edits.append((r.start.line, r.start.character, r.end.character, edit.new_text))

    # Apply edits in reverse order (right to left) so positions stay valid
    # All edits should be on the same line for completions
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)

    for line_idx, start_col, end_col, new_text in edits:
        line = lines[line_idx] if line_idx < len(lines) else ""
        has_newline = line.endswith("\n")
        line_content = line.rstrip("\n")
        new_line = line_content[:start_col] + new_text + line_content[end_col:]
        if has_newline:
            new_line += "\n"
        lines[line_idx] = new_line

    actual = "".join(lines)

    # Strip snippet placeholders for comparison
    actual = re.sub(r'\$\d+|\$\{[^}]*\}', '', actual)
    expected = re.sub(r'\$\d+|\$\{[^}]*\}', '', expected)

    assert actual == expected, (
        f"check_edit failed for completion '{label}':\n"
        f"Expected:\n{expected}\n"
        f"Actual:\n{actual}"
    )


def _get_edit_range(edit) -> types.Range:
    """Extract range from TextEdit or InsertReplaceEdit."""
    if isinstance(edit, types.InsertReplaceEdit):
        return edit.insert
    return edit.range


def _get_edit_new_text(edit) -> str:
    """Extract new_text from TextEdit or InsertReplaceEdit."""
    return edit.new_text
