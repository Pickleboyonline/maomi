class MaomiError(Exception):
    def __init__(self, message: str, filename: str, line: int, col: int):
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        super().__init__(f"{filename}:{line}:{col}: {message}")


class LexerError(MaomiError):
    pass


class ParseError(MaomiError):
    pass


class MaomiTypeError(MaomiError):
    pass
