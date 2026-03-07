"""Tests for the high-level Python API (maomi.compile)."""

import numpy as np
import pytest

import maomi
from maomi.api import (
    MaomiStruct,
    MaomiModule,
    MaomiFunction,
    flatten_type,
    flatten_value,
    unflatten_value,
    _generate_wrapper,
)
from maomi.types import ScalarType, ArrayType, StructType
from maomi.cli import compile_source


class TestCompileEntryPoint:
    def test_compile_from_string(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        assert isinstance(mod, MaomiModule)

    def test_compile_from_file(self, tmp_path):
        src = "fn add(a: f32, b: f32) -> f32 { a + b }"
        p = tmp_path / "test.mao"
        p.write_text(src)
        mod = maomi.compile(str(p))
        assert isinstance(mod, MaomiModule)

    def test_compile_error(self):
        with pytest.raises(Exception):
            maomi.compile("fn bad(a: f32) -> i32 { a }")


class TestMaomiStruct:
    def test_field_access(self):
        stype = StructType("P", (("x", ScalarType("f32")), ("y", ScalarType("i32"))))
        s = MaomiStruct("P", stype, x=np.float32(1.0), y=np.int32(2))
        assert s.x == np.float32(1.0)
        assert s.y == np.int32(2)

    def test_missing_field(self):
        stype = StructType("P", (("x", ScalarType("f32")),))
        s = MaomiStruct("P", stype, x=np.float32(1.0))
        with pytest.raises(AttributeError, match="no field 'z'"):
            _ = s.z

    def test_repr(self):
        stype = StructType("P", (("w", ArrayType("f32", (4,))),))
        s = MaomiStruct("P", stype, w=np.zeros(4, dtype=np.float32))
        r = repr(s)
        assert "P" in r
        assert "w:" in r


class TestFlattenUnflatten:
    def test_flatten_scalar(self):
        t = ScalarType("f32")
        assert flatten_type(t) == [t]

    def test_flatten_array(self):
        t = ArrayType("f32", (3, 4))
        assert flatten_type(t) == [t]

    def test_flatten_struct(self):
        t = StructType("P", (
            ("w", ArrayType("f32", (4,))),
            ("b", ScalarType("f32")),
        ))
        flat = flatten_type(t)
        assert len(flat) == 2
        assert flat[0] == ArrayType("f32", (4,))
        assert flat[1] == ScalarType("f32")

    def test_flatten_nested_struct(self):
        inner = StructType("Inner", (("x", ScalarType("f32")),))
        outer = StructType("Outer", (
            ("a", inner),
            ("b", ScalarType("i32")),
        ))
        flat = flatten_type(outer)
        assert len(flat) == 2
        assert flat[0] == ScalarType("f32")
        assert flat[1] == ScalarType("i32")

    def test_flatten_value_struct(self):
        stype = StructType("P", (
            ("w", ArrayType("f32", (3,))),
            ("b", ScalarType("f32")),
        ))
        s = MaomiStruct("P", stype, w=np.ones(3, dtype=np.float32), b=np.float32(2.0))
        flat = flatten_value(s, stype)
        assert len(flat) == 2
        np.testing.assert_array_equal(flat[0], np.ones(3, dtype=np.float32))
        assert flat[1] == np.float32(2.0)

    def test_unflatten_struct(self):
        stype = StructType("P", (
            ("w", ArrayType("f32", (3,))),
            ("b", ScalarType("f32")),
        ))
        arrays = [np.ones(3, dtype=np.float32), np.float32(2.0)]
        val, offset = unflatten_value(arrays, stype, {"P": stype})
        assert offset == 2
        assert isinstance(val, MaomiStruct)
        np.testing.assert_array_equal(val.w, np.ones(3, dtype=np.float32))
        assert val.b == np.float32(2.0)

    def test_roundtrip(self):
        stype = StructType("P", (
            ("w", ArrayType("f32", (2, 3))),
            ("b", ScalarType("f32")),
        ))
        orig_w = np.arange(6, dtype=np.float32).reshape(2, 3)
        orig_b = np.float32(7.0)
        s = MaomiStruct("P", stype, w=orig_w, b=orig_b)
        flat = flatten_value(s, stype)
        restored, _ = unflatten_value(flat, stype, {"P": stype})
        np.testing.assert_array_equal(restored.w, orig_w)
        assert restored.b == orig_b


class TestWrapperGeneration:
    def test_no_struct_wrapper(self):
        src = "fn add(a: f32, b: f32) -> f32 { a + b }"
        result = compile_source(src)
        from maomi.type_checker import FnSignature
        sig = result.fn_table["add"]
        wrapped = _generate_wrapper(result.mlir_text, "add", sig)
        assert "@main" in wrapped
        assert "func.call @add" in wrapped
        assert "module @main" in wrapped

    def test_struct_param_wrapper(self):
        src = """
        struct P { w: f32[4], b: f32 }
        fn f(p: P) -> f32 { p.b }
        """
        result = compile_source(src)
        sig = result.fn_table["f"]
        wrapped = _generate_wrapper(result.mlir_text, "f", sig)
        assert "stablehlo.tuple" in wrapped
        assert "func.call @f" in wrapped

    def test_struct_return_wrapper(self):
        src = """
        struct P { x: f32, y: f32 }
        fn f(a: f32, b: f32) -> P { P { x: a, y: b } }
        """
        result = compile_source(src)
        sig = result.fn_table["f"]
        wrapped = _generate_wrapper(result.mlir_text, "f", sig)
        assert "get_tuple_element" in wrapped


class TestModuleRepr:
    def test_repr(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        r = repr(mod)
        assert "MaomiModule" in r
        assert "add" in r

    def test_unknown_attr(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        with pytest.raises(AttributeError, match="no function or struct"):
            _ = mod.nonexistent

    def test_builtin_not_exposed(self):
        mod = maomi.compile("fn f(a: f32) -> f32 { exp(a) }")
        with pytest.raises(AttributeError):
            _ = mod.exp


class TestFunctionRepr:
    def test_repr(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        r = repr(mod.add)
        assert "MaomiFunction" in r
        assert "add" in r


class TestStructConstructor:
    def test_construct(self):
        mod = maomi.compile("""
            struct P { x: f32, y: i32 }
            fn f(p: P) -> f32 { p.x }
        """)
        p = mod.P(x=np.float32(1.0), y=np.int32(2))
        assert isinstance(p, MaomiStruct)
        assert p.x == np.float32(1.0)
        assert p.y == np.int32(2)

    def test_construct_missing_field(self):
        mod = maomi.compile("""
            struct P { x: f32, y: i32 }
            fn f(p: P) -> f32 { p.x }
        """)
        with pytest.raises(TypeError, match="field mismatch"):
            mod.P(x=np.float32(1.0))

    def test_construct_extra_field(self):
        mod = maomi.compile("""
            struct P { x: f32 }
            fn f(p: P) -> f32 { p.x }
        """)
        with pytest.raises(TypeError, match="field mismatch"):
            mod.P(x=np.float32(1.0), z=np.float32(2.0))


class TestExecution:
    """Tests that require JAX for actual execution."""

    @pytest.fixture(autouse=True)
    def _skip_without_jax(self):
        pytest.importorskip("jax")

    def test_scalar_add(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        result = mod.add(np.float32(2.0), np.float32(3.0))
        assert float(result) == pytest.approx(5.0)

    def test_type_coercion_int(self):
        mod = maomi.compile("fn inc(a: i32) -> i32 { a + 1 }")
        result = mod.inc(41)
        assert int(result) == 42

    def test_type_coercion_float(self):
        mod = maomi.compile("fn dbl(a: f32) -> f32 { a + a }")
        result = mod.dbl(2.5)
        assert float(result) == pytest.approx(5.0)

    def test_array_function(self):
        mod = maomi.compile("fn add(a: f32[3], b: f32[3]) -> f32[3] { a + b }")
        a = np.array([1., 2., 3.], dtype=np.float32)
        b = np.array([4., 5., 6.], dtype=np.float32)
        result = mod.add(a, b)
        np.testing.assert_allclose(result, [5., 7., 9.])

    def test_wrong_arg_count(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        with pytest.raises(TypeError, match="takes 2"):
            mod.add(np.float32(1.0))

    def test_executable_caching(self):
        mod = maomi.compile("fn add(a: f32, b: f32) -> f32 { a + b }")
        fn = mod.add
        _ = fn(np.float32(1.0), np.float32(2.0))
        exe1 = fn._executable
        _ = fn(np.float32(3.0), np.float32(4.0))
        exe2 = fn._executable
        assert exe1 is exe2

    def test_struct_roundtrip(self):
        mod = maomi.compile("""
            struct P { w: f32[4], b: f32 }
            fn step(p: P, x: f32[4]) -> P {
                p with { w = p.w + x, b = p.b + 1.0 }
            }
        """)
        p = mod.P(w=np.zeros(4, dtype=np.float32), b=np.float32(0.0))
        p2 = mod.step(p, np.ones(4, dtype=np.float32))
        assert isinstance(p2, MaomiStruct)
        np.testing.assert_allclose(p2.w, [1., 1., 1., 1.])
        assert float(p2.b) == pytest.approx(1.0)

    def test_struct_param_scalar_return(self):
        mod = maomi.compile("""
            struct P { x: f32, y: f32 }
            fn sum_fields(p: P) -> f32 { p.x + p.y }
        """)
        p = mod.P(x=np.float32(3.0), y=np.float32(4.0))
        result = mod.sum_fields(p)
        assert float(result) == pytest.approx(7.0)

    def test_nested_struct(self):
        mod = maomi.compile("""
            struct Inner { v: f32 }
            struct Outer { a: Inner, b: f32 }
            fn get_v(o: Outer) -> f32 { o.a.v + o.b }
        """)
        inner = mod.Inner(v=np.float32(3.0))
        outer = mod.Outer(a=inner, b=np.float32(4.0))
        result = mod.get_v(outer)
        assert float(result) == pytest.approx(7.0)

    def test_multiple_functions(self):
        mod = maomi.compile("""
            fn add(a: f32, b: f32) -> f32 { a + b }
            fn mul(a: f32, b: f32) -> f32 { a * b }
        """)
        assert float(mod.add(np.float32(2.0), np.float32(3.0))) == pytest.approx(5.0)
        assert float(mod.mul(np.float32(2.0), np.float32(3.0))) == pytest.approx(6.0)

    def test_matmul(self):
        mod = maomi.compile("fn mm(a: f32[2, 3], b: f32[3, 4]) -> f32[2, 4] { a @ b }")
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        result = mod.mm(a, b)
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result, 3.0)
