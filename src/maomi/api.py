"""High-level Python API for Maomi.

Usage:
    import maomi
    mod = maomi.compile("examples/mlp.mao")
    result = mod.some_fn(arg1, arg2)
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import numpy as np

from .cli import compile_source
from .types import ScalarType, ArrayType, StructType, MaomiType

if TYPE_CHECKING:
    from .type_checker import FnSignature

# Maomi base types -> numpy dtypes
_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
}

# Maomi base types -> MLIR element types
_MLIR_ETYPE = {
    "f32": "f32",
    "f64": "f64",
    "i32": "i32",
    "i64": "i64",
    "bool": "i1",
}

# Builtins to filter out from user-facing function list
_BUILTIN_NAMES = frozenset(("mean", "sum", "exp", "log", "tanh", "sqrt", "abs"))


class MaomiStruct:
    """Python representation of a Maomi struct value."""

    def __init__(self, _name: str, _type: StructType, **fields):
        object.__setattr__(self, "_name", _name)
        object.__setattr__(self, "_type", _type)
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(f"'{object.__getattribute__(self, '_name')}' has no field '{name}'")

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_name")
        fields = object.__getattribute__(self, "_fields")
        parts = []
        for k, v in fields.items():
            if isinstance(v, np.ndarray):
                parts.append(f"{k}: {v.dtype.name}{list(v.shape) if v.ndim > 0 else ''}")
            elif isinstance(v, MaomiStruct):
                parts.append(f"{k}: {v!r}")
            else:
                parts.append(f"{k}: {v}")
        return f"{name} {{ {', '.join(parts)} }}"


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------

def flatten_type(t: MaomiType) -> list[MaomiType]:
    """Return the list of leaf (non-struct) types in a MaomiType."""
    if isinstance(t, StructType):
        result = []
        for _, ft in t.fields:
            result.extend(flatten_type(ft))
        return result
    return [t]


def flatten_value(value, t: MaomiType) -> list[np.ndarray]:
    """Flatten a Python value into a list of numpy arrays matching leaf types."""
    if isinstance(t, StructType):
        result = []
        for fname, ft in t.fields:
            fval = getattr(value, fname)
            result.extend(flatten_value(fval, ft))
        return result
    return [_coerce_to_numpy(value, t)]


def unflatten_value(arrays: list[np.ndarray], t: MaomiType, struct_defs: dict[str, StructType], offset: int = 0):
    """Unflatten a list of numpy arrays into a Python value matching the type."""
    if isinstance(t, StructType):
        fields = {}
        for fname, ft in t.fields:
            val, offset = unflatten_value(arrays, ft, struct_defs, offset)
            fields[fname] = val
        return MaomiStruct(t.name, t, **fields), offset
    return np.asarray(arrays[offset]), offset + 1


def _coerce_to_numpy(value, t: MaomiType) -> np.ndarray:
    """Convert a Python value to a numpy array matching the expected Maomi type."""
    dtype = _DTYPE_MAP[t.base]
    if isinstance(value, np.ndarray):
        if value.dtype != dtype:
            return value.astype(dtype)
        return value
    # Python scalar -> 0-d numpy array
    return np.array(value, dtype=dtype)


# ---------------------------------------------------------------------------
# MLIR wrapper generation
# ---------------------------------------------------------------------------

def _mlir_type(t: MaomiType) -> str:
    """Convert a MaomiType to an MLIR tensor type string."""
    if isinstance(t, ScalarType):
        return f"tensor<{_MLIR_ETYPE[t.base]}>"
    if isinstance(t, ArrayType):
        shape = "x".join(str(d) for d in t.dims)
        return f"tensor<{shape}x{_MLIR_ETYPE[t.base]}>"
    if isinstance(t, StructType):
        field_types = ", ".join(_mlir_type(ft) for _, ft in t.fields)
        return f"tuple<{field_types}>"
    raise TypeError(f"unknown type: {t}")


def _build_tuple_from_args(flat_types: list[MaomiType], t: MaomiType,
                           arg_names: list[str], offset: int,
                           lines: list[str], var_counter: list[int]) -> tuple[str, int]:
    """Generate MLIR ops to build a tuple from flat args. Returns (ssa_name, new_offset)."""
    if isinstance(t, StructType):
        field_ssas = []
        for _, ft in t.fields:
            ssa, offset = _build_tuple_from_args(flat_types, ft, arg_names, offset, lines, var_counter)
            field_ssas.append(ssa)
        tup_var = f"%__tup{var_counter[0]}"
        var_counter[0] += 1
        mlir_t = _mlir_type(t)
        field_refs = ", ".join(field_ssas)
        field_mlir_types = ", ".join(_mlir_type(ft) for _, ft in t.fields)
        lines.append(f"    {tup_var} = \"stablehlo.tuple\"({field_refs}) : ({field_mlir_types}) -> {mlir_t}")
        return tup_var, offset
    return arg_names[offset], offset + 1


def _destructure_tuple(t: MaomiType, src_ssa: str,
                       lines: list[str], var_counter: list[int],
                       result_ssas: list[str]):
    """Generate MLIR ops to destructure a tuple into flat tensors."""
    if isinstance(t, StructType):
        for i, (_, ft) in enumerate(t.fields):
            elem_var = f"%__gte{var_counter[0]}"
            var_counter[0] += 1
            elem_mlir = _mlir_type(ft)
            src_mlir = _mlir_type(t)
            lines.append(f"    {elem_var} = \"stablehlo.get_tuple_element\"({src_ssa}) {{index = {i} : i32}} : ({src_mlir}) -> {elem_mlir}")
            _destructure_tuple(ft, elem_var, lines, var_counter, result_ssas)
    else:
        result_ssas.append(src_ssa)


def _generate_wrapper(mlir_text: str, fn_name: str, fn_sig: FnSignature) -> str:
    """Generate a @main wrapper that bridges flat tensors <-> tuples for the target function."""
    # Compute flat param types and flat return types
    flat_param_types = []
    for pt in fn_sig.param_types:
        flat_param_types.extend(flatten_type(pt))

    flat_ret_types = flatten_type(fn_sig.return_type)

    # Build wrapper function signature
    wrapper_params = []
    arg_names = []
    for i, ft in enumerate(flat_param_types):
        name = f"%arg{i}"
        arg_names.append(name)
        wrapper_params.append(f"{name}: {_mlir_type(ft)}")

    flat_ret_mlir = [_mlir_type(rt) for rt in flat_ret_types]

    lines = []
    lines.append(f"  func.func @main({', '.join(wrapper_params)}) -> ({', '.join(flat_ret_mlir)}) {{")

    # Build call args: reconstruct tuples from flat args where needed
    var_counter = [0]
    call_args = []
    offset = 0
    for pt in fn_sig.param_types:
        if isinstance(pt, StructType):
            ssa, offset = _build_tuple_from_args(flat_param_types, pt, arg_names, offset, lines, var_counter)
            call_args.append(ssa)
        else:
            call_args.append(arg_names[offset])
            offset += 1

    # Call the original function
    call_arg_str = ", ".join(call_args)
    call_type_str = ", ".join(_mlir_type(pt) for pt in fn_sig.param_types)
    ret_mlir = _mlir_type(fn_sig.return_type)
    result_var = f"%__result{var_counter[0]}"
    var_counter[0] += 1
    lines.append(f"    {result_var} = func.call @{fn_name}({call_arg_str}) : ({call_type_str}) -> {ret_mlir}")

    # Destructure result if struct
    if isinstance(fn_sig.return_type, StructType):
        result_ssas: list[str] = []
        _destructure_tuple(fn_sig.return_type, result_var, lines, var_counter, result_ssas)
        lines.append(f"    return {', '.join(result_ssas)} : {', '.join(flat_ret_mlir)}")
    else:
        lines.append(f"    return {result_var} : {ret_mlir}")

    lines.append("  }")

    wrapper_text = "\n".join(lines)

    # Inject wrapper into module: before the closing `}`
    # Also add module @main sym_name
    mlir_text = mlir_text.replace("module {", "module @main {", 1)

    # Find the last `}` in the module and insert the wrapper before it
    last_brace = mlir_text.rfind("}")
    mlir_text = mlir_text[:last_brace] + wrapper_text + "\n" + mlir_text[last_brace:]

    return mlir_text


# ---------------------------------------------------------------------------
# MaomiFunction
# ---------------------------------------------------------------------------

class MaomiFunction:
    """Callable wrapper around a compiled Maomi function."""

    def __init__(self, fn_name: str, fn_sig: FnSignature, mlir_text: str, struct_defs: dict[str, StructType]):
        self._fn_name = fn_name
        self._fn_sig = fn_sig
        self._mlir_text = mlir_text
        self._struct_defs = struct_defs
        self._executable = None
        self._backend = None

    def __repr__(self) -> str:
        sig = self._fn_sig
        params = ", ".join(f"{n}: {t}" for n, t in zip(sig.param_names, sig.param_types))
        return f"<MaomiFunction {self._fn_name}({params}) -> {sig.return_type}>"

    def __call__(self, *args, _callbacks=None):
        sig = self._fn_sig
        if len(args) != len(sig.param_types):
            raise TypeError(
                f"{self._fn_name}() takes {len(sig.param_types)} argument(s), got {len(args)}"
            )

        # Flatten all args
        flat_inputs = []
        for arg, pt in zip(args, sig.param_types):
            flat_inputs.extend(flatten_value(arg, pt))

        # Compile on first call
        if self._executable is None:
            self._compile(_callbacks)

        # Execute
        buffers = [self._backend.buffer_from_pyval(arr) for arr in flat_inputs]
        results = self._executable.execute(buffers)

        # Unflatten output
        flat_ret_types = flatten_type(sig.return_type)
        result_arrays = [np.asarray(r) for r in results[:len(flat_ret_types)]]

        if isinstance(sig.return_type, StructType):
            value, _ = unflatten_value(result_arrays, sig.return_type, self._struct_defs)
            return value
        return result_arrays[0]

    def _compile(self, host_callbacks=None):
        import jax
        from jax._src import compiler, xla_bridge as xb
        from jax._src.interpreters import mlir
        from jax._src.lib import xla_client as xc
        from jax._src.lib.mlir import ir

        jax.devices()

        wrapped = _generate_wrapper(self._mlir_text, self._fn_name, self._fn_sig)

        ctx = mlir.make_ir_context()
        with ctx:
            module = ir.Module.parse(wrapped)

        self._backend = xb.get_backend()
        self._executable = compiler.backend_compile_and_load(
            self._backend,
            module,
            xc.DeviceList(tuple(jax.devices())),
            xc.CompileOptions(),
            host_callbacks or [],
        )


# ---------------------------------------------------------------------------
# MaomiModule
# ---------------------------------------------------------------------------

class MaomiModule:
    """Represents a compiled Maomi module with callable functions and struct constructors."""

    def __init__(self, mlir_text: str, fn_table: dict[str, FnSignature],
                 struct_defs: dict[str, StructType],
                 callback_count: int = 0, callback_labels: dict[int, list[str]] | None = None):
        self._mlir_text = mlir_text
        self._fn_table = fn_table
        self._struct_defs = struct_defs
        self._callback_count = callback_count
        self._callback_labels = callback_labels or {}
        self._functions: dict[str, MaomiFunction] = {}

    def __repr__(self) -> str:
        user_fns = [k for k in self._fn_table if k not in _BUILTIN_NAMES]
        structs = list(self._struct_defs.keys())
        parts = []
        if user_fns:
            parts.append(f"functions={user_fns}")
        if structs:
            parts.append(f"structs={structs}")
        return f"<MaomiModule {', '.join(parts)}>"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        # Check functions (user-defined only)
        fn_table = object.__getattribute__(self, "_fn_table")
        if name in fn_table and name not in _BUILTIN_NAMES:
            functions = object.__getattribute__(self, "_functions")
            if name not in functions:
                functions[name] = MaomiFunction(
                    name,
                    fn_table[name],
                    object.__getattribute__(self, "_mlir_text"),
                    object.__getattribute__(self, "_struct_defs"),
                )
            return functions[name]

        # Check struct constructors
        struct_defs = object.__getattribute__(self, "_struct_defs")
        if name in struct_defs:
            stype = struct_defs[name]
            def _constructor(**fields):
                expected = {fname for fname, _ in stype.fields}
                got = set(fields.keys())
                if got != expected:
                    missing = expected - got
                    extra = got - expected
                    parts = []
                    if missing:
                        parts.append(f"missing: {missing}")
                    if extra:
                        parts.append(f"unexpected: {extra}")
                    raise TypeError(f"{name}() field mismatch: {', '.join(parts)}")
                return MaomiStruct(name, stype, **fields)
            return _constructor

        raise AttributeError(f"module has no function or struct '{name}'")


# ---------------------------------------------------------------------------
# compile() entry point
# ---------------------------------------------------------------------------

def compile(source_or_path: str) -> MaomiModule:
    """Compile a Maomi source string or .mao file into a callable module.

    Args:
        source_or_path: Either a path to a .mao file or a Maomi source string.

    Returns:
        A MaomiModule with callable functions and struct constructors.
    """
    if source_or_path.endswith(".mao") and os.path.isfile(source_or_path):
        with open(source_or_path) as f:
            source = f.read()
        filename = source_or_path
    else:
        source = source_or_path
        filename = "<string>"

    result = compile_source(source, filename)
    return MaomiModule(
        result.mlir_text,
        result.fn_table,
        result.struct_defs,
        result.callback_count,
        result.callback_labels,
    )
