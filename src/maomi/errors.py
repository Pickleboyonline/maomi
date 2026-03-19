class MaomiError(Exception):
    def __init__(self, message: str, filename: str, line: int, col: int, col_end: int | None = None):
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.col_end = col_end if col_end is not None else col + 1
        super().__init__(f"{filename}:{line}:{col}: {message}")


class LexerError(MaomiError):
    pass


class ParseError(MaomiError):
    pass


class MaomiTypeError(MaomiError):
    pass
