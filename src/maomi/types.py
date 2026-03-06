from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    base: str  # "f32", "f64", "i32", "i64", "bool"

    def __str__(self) -> str:
        return self.base


@dataclass(frozen=True)
class ArrayType:
    base: str
    dims: tuple[int | str, ...]  # int = concrete, str = symbolic

    def __str__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
        return f"{self.base}[{dims_str}]"


MaomiType = ScalarType | ArrayType

# Convenience constants
F32 = ScalarType("f32")
F64 = ScalarType("f64")
I32 = ScalarType("i32")
I64 = ScalarType("i64")
BOOL = ScalarType("bool")
