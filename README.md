# simd.hpp

A single-header, C++20 portable SIMD library. The same source compiles to
native vector code where the target has a vector unit and to plain scalar code
where it does not, and gives the same answers either way.

```cpp
#include "simd.hpp"
```

Everything lives in `namespace simd`. The snippets below assume
`using namespace simd;`.

## Types

### Scalar aliases

| Alias | Type |
|---|---|
| `u8` `u16` `u32` `u64` | `uint8_t` … `uint64_t` |
| `s8` `s16` `s32` `s64` | `int8_t` … `int64_t` |
| `f32` `f64` | `float`, `double` |
| `usize` | `size_t` |
| `width_type` | `usize` |

### Vectors

`vec<N, W>` is a vector of `W` lanes of scalar type `N`, for
`N` in `f32`, `u32`, `f64`, `u64`. Each is named `<N>x<W>`, e.g. `f32x4`,
`u64x2`.

Which widths exist depends on the target, so portable code should not name a
width directly. Use:

| Name | Meaning |
|---|---|
| `plat_vec<N>` | The widest `vec` the target supports for `N`. Always exists; width 1 on targets with no vector unit. |
| `plat_width<N>::value` | Its width. |
| `plat_vector_bytes` | Width of the widest vector register in bytes (0 if none). |

Width 1 (`f32x1`, `u32x1`, `f64x1`, `u64x1`) is available on every target.

Every vec type provides:

| Member | Meaning |
|---|---|
| `scalar_type` | `N` |
| `mask_type` | The type comparisons return (see [Masks](#masks)) |
| `width` | `W`, as a `static constexpr width_type` |
| `simd_type` | The underlying native type |

Construction:

```cpp
f32x4 a;              // all lanes zero
f32x4 b = 1.5f;       // broadcast, same as splat<f32x4>(1.5f)
f32x4 c = raw;        // from the native simd_type; also converts back implicitly
```

### Concepts

| Concept | Satisfied by |
|---|---|
| `ScalarType<T>` | Integral or floating-point types other than `bool` |
| `VecType<V>` | Any type with `V::scalar_type` and `V::width` |

## Operators

All operators are lane-wise. Any binary arithmetic or comparison operator also
accepts a scalar on either side, which is broadcast.

| Operators | Float vecs | Unsigned vecs |
|---|---|---|
| `+` `-` `*` | ✓ | ✓ (wrapping) |
| `/` | ✓ | — |
| unary `-` | ✓ | — |
| `==` `!=` `<` `<=` `>` `>=` → `mask_type` | ✓ | ✓ |
| `&` `\|` `^` `~` | — | ✓ |
| `<<` `>>` by an `int` count (logical) | — | ✓ |
| `+=` `-=` `*=` | ✓ | ✓ |
| `/=` | ✓ | — |
| `&=` `\|=` `^=` `<<=` `>>=` | — | ✓ |

Shift counts must be less than the lane width in bits.

Float comparisons follow IEEE: any comparison with a NaN is false, except `!=`,
which is true.

## Functions

Where the table shows `<V>`, the vec type must be given explicitly. Everywhere
else it is deduced from the arguments.

### Construction and memory

| Function | Description |
|---|---|
| `splat<V>(s)` / `splat<N, W>(s)` | Every lane set to `s`. |
| `iota<V>()` | Lanes `0, 1, 2, …`. |
| `load<V>(const N* p)` | Load `V::width` elements from `p`. No alignment required. |
| `store(N* p, v)` | Store every lane to `p`. No alignment required. |

### Lane access

`i` is a compile-time lane index, checked with `static_assert`.

| Function | Description |
|---|---|
| `get_lane<i>(v)` | Lane `i` as a scalar. |
| `set_lane<i>(v, s)` | `v` with lane `i` replaced by `s`. |
| `load_lane<i>(v, const N* p)` | `v` with lane `i` replaced by `*p`. |
| `store_lane<i>(N* p, v)` | Write lane `i` to `*p`. |

Pass only the index: `get_lane<0>(v)` compiles, but `get_lane<0, f32x4>(v)`
does not.

### Masked memory access

| Function | Description |
|---|---|
| `load_masked(mask, src, const N* p)` | Lanes where `mask` is set are read from `p`; the rest keep `src`. Memory for unset lanes is never touched, so this is safe at the end of a buffer. |
| `load_masked_unsafe(mask, src, const N* p)` | Same result, but reads all `V::width` elements at `p`. Use only when that memory is known to be readable. |
| `store_masked(N* p, mask, v)` | Write only the lanes where `mask` is set. |

### Arithmetic (float vecs)

| Function | Result |
|---|---|
| `fma(a, b, c)` | `a * b + c` |
| `fms(a, b, c)` | `a * b - c` |
| `fms2(a, b, c)` | `-(a * b) + c` |
| `abs(v)` | Absolute value |
| `sqrt(v)` | Square root |
| `recip(v)` | `1 / v` |
| `rsqrt(v)` | `1 / sqrt(v)` |
| `round(v)` | Round to nearest, ties to even |
| `floor(v)` / `ceil(v)` / `trunc(v)` | Round down / up / toward zero |

### Min, max and reductions (all vecs)

| Function | Result |
|---|---|
| `min(a, b)` / `max(a, b)` | Lane-wise minimum / maximum |
| `clamp(v, lo, hi)` | `min(max(v, lo), hi)` |
| `hsum(v)` | Sum of all lanes, as a scalar (wraps for unsigned) |
| `hmin(v)` / `hmax(v)` | Smallest / largest lane, as a scalar |
| `dot(a, b)` | `hsum(a * b)` |

### Selection

| Function | Result per lane |
|---|---|
| `select(mask, a, b)` | `mask ? a : b` (all vecs) |
| `select_add(mask, src, a, b)` | `mask ? a + b : src` |
| `select_sub(mask, src, a, b)` | `mask ? a - b : src` |
| `select_mul(mask, src, a, b)` | `mask ? a * b : src` |
| `select_div(mask, src, a, b)` | `mask ? a / b : src` |
| `select_min(mask, src, a, b)` | `mask ? min(a, b) : src` |
| `select_max(mask, src, a, b)` | `mask ? max(a, b) : src` |
| `select_fma(mask, src, a, b, c)` | `mask ? a * b + c : src` |
| `select_fms(mask, src, a, b, c)` | `mask ? a * b - c : src` |
| `select_fms2(mask, src, a, b, c)` | `mask ? -(a * b) + c : src` |

The `select_*` family is available for float vecs only.

### Conversion

| Function | Description |
|---|---|
| `cast<To>(v)` | Convert each lane's value, between `f32`↔`u32` or `f64`↔`u64` vecs of the same width. Float to unsigned truncates toward zero and saturates: negatives and NaN give 0, values too large give the maximum. |
| `reinterpret<To>(v)` | Reinterpret the bits of each lane, between the same pairs. |

### Loops

| Function | Description |
|---|---|
| `simd_loop_count<V>(n)` | `n` rounded down to a multiple of `V::width`: the number of elements covered by whole vectors. |

## Masks

A comparison of two `vec<N, W>` values returns `vec<N, W>::mask_type`. For
vector widths this is the unsigned vec with the same lane size (`u32` lanes for
`f32` or `u32`, `u64` lanes for `f64` or `u64`). At width 1 it is `bool`.

A mask lane is **canonical** when all of its bits are set (true) or all are
clear (false). Everything that reads a mask assumes this. Comparisons and the
functions below always produce canonical masks. Do not build masks by hand, e.g.
with `splat(1)`.

Because the mask type differs between widths, write portable mask code with the
functions below rather than operators, and let the mask type be deduced (write
`any(m)`, not `any<M>(m)`).

### Building masks

| Function | Result |
|---|---|
| `mask_all<V>()` | Every lane true |
| `mask_none<V>()` | Every lane false |
| `mask_from_bool<V>(b)` | Every lane equal to `b` |
| `mask_first_n<V>(n)` | Lanes `[0, n)` true, the rest false; `n` is clamped to `V::width` |
| `mask_from_bits<V>(bits)` | Lane `i` takes bit `i` of `bits`; bits at or above `V::width` are ignored |
| `mask_from_bits_at<V>(bits, offset)` | Lane `j` takes bit `offset + j`; bits at index 32 or above read as false |

### Mask logic

| Function | Result |
|---|---|
| `mask_and(a, b)` | `a & b` |
| `mask_or(a, b)` | `a \| b` |
| `mask_xor(a, b)` | `a ^ b` |
| `mask_not(m)` | `~m` |
| `mask_andnot(a, b)` | `a & ~b` |

### Mask queries

| Function | Result |
|---|---|
| `any(m)` | True if any lane is set |
| `all(m)` | True if every lane is set |
| `none(m)` | True if no lane is set |
| `movemask(m)` | `u32` with bit `i` set for each true lane `i`; the inverse of `mask_from_bits` |

### Mask storage

`mask_storage<V>` builds a mask one lane at a time in memory and then loads it:

```cpp
using MS = mask_storage<V>;
MS::element buf[V::width];
buf[i] = cond ? MS::true_value : MS::false_value;
typename V::mask_type m = MS::load(buf);
MS::store(buf, m);
```

## Example

This scales an array of any length in place, including the tail that doesn't
fill a whole vector:

```cpp
#include "simd.hpp"
using namespace simd;

void scale(f32* data, usize n, f32 k) {
    using V = plat_vec<f32>;

    const usize body = simd_loop_count<V>(n);
    for (usize i = 0; i < body; i += V::width)
        store(data + i, load<V>(data + i) * k);

    const auto tail = mask_first_n<V>(n - body);
    const V v = load_masked(tail, V{}, data + body);
    store_masked(data + body, tail, v * k);
}
```

## Portability notes

Results are identical on every target except in these cases:

- **`fma`, `fms`, `fms2`** are fused (rounded once) where the hardware supports
  it, and otherwise rounded twice. Results can differ by one ulp.
- **Float `min` and `max`** may differ when an input is NaN, or for
  `min(+0, -0)` and `max(+0, -0)`. If that matters, compare and `select`
  explicitly.
- **`recip` and `rsqrt`** may be approximations. Treat them as accurate to
  within a few ulp.
