from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    base: str  # "f32", "f64", "bf16", "i32", "i64", "bool"

    def __str__(self) -> str:
        return self.base


@dataclass(frozen=True)
class ArrayType:
    base: str
    dims: tuple[int | str, ...]  # int = concrete, str = symbolic

    def __str__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
        return f"{self.base}[{dims_str}]"


@dataclass(frozen=True)
class StructType:
    name: str
    fields: tuple[tuple[str, MaomiType], ...]  # ordered (name, type) pairs

    def __str__(self) -> str:
        fields_str = ", ".join(f"{n}: {t}" for n, t in self.fields)
        return f"{self.name} {{ {fields_str} }}"


@dataclass(frozen=True)
class WildcardArrayType:
    base: str  # "f32", "i32", etc.

    def __str__(self) -> str:
        return f"{self.base}[..]"


@dataclass(frozen=True)
class StringType:
    def __str__(self) -> str:
        return "str"


@dataclass(frozen=True)
class TypeVar:
    name: str  # single uppercase letter: "T", "U", etc.

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class StructArrayType:
    """Array of structs, stored as struct-of-arrays (SoA).
    Batch[N] where Batch{x: f32[64,784], y: f32[64,10]}
    → XLA tuple<tensor<N×64×784×f32>, tensor<N×64×10×f32>>"""
    struct_type: StructType
    dims: tuple[int | str, ...]  # leading batch dimensions

    @property
    def name(self) -> str:
        return self.struct_type.name

    def __str__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
        return f"{self.struct_type.name}[{dims_str}]"


MaomiType = ScalarType | ArrayType | StructType | StructArrayType | WildcardArrayType | StringType | TypeVar

# Convenience constants
F32 = ScalarType("f32")
F64 = ScalarType("f64")
BF16 = ScalarType("bf16")
I32 = ScalarType("i32")
I64 = ScalarType("i64")
BOOL = ScalarType("bool")
STRING = StringType()

# Base type sets — single source of truth for type classification
FLOAT_BASES = frozenset({"f32", "f64", "bf16"})

