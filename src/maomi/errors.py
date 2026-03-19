class MaomiError(Exception):
    def __init__(self, message: str, filename: str, line: int, col: int,
                 col_end: int | None = None,
                 hint: str | None = None,
                 secondary_labels: list | None = None,
                 severity: str = "error"):
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.col_end = col_end if col_end is not None else col + 1
        self.hint = hint
        self.secondary_labels = secondary_labels or []
        self.severity = severity
        super().__init__(f"{filename}:{line}:{col}: {message}")


class LexerError(MaomiError):
    pass


class ParseError(MaomiError):
    pass


class MaomiTypeError(MaomiError):
    pass
