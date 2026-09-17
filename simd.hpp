#ifndef SIMD_HPP_
#define SIMD_HPP_

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

// SIMD_MAX_WIDTH_ names the widest tier of vector the target offers:
//   2 -> 256-bit, 1 -> 128-bit, 0 -> scalar fallback.
// Every branch below must define it, so the scalar path is always reachable,
// and each tier implies the ones below it: a tier-2 target also defines the
// 128-bit types.
//
// The tier no longer identifies the instruction set on its own -- tier 1 is
// NEON on ARM and SSE4.2 on x86 -- so each branch also defines exactly one
// SIMD_BACKEND_*_ macro, which is what the implementations below switch on.
#if defined(__aarch64__) || defined(_M_ARM64)

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_BACKEND_NEON_ 1
#define SIMD_MAX_WIDTH_ 1
#else
#define SIMD_MAX_WIDTH_ 0
#endif

#elif defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

// Note that -mavx2 implies __SSE4_2__ but NOT __FMA__, so fma/fms/fms2 are
// gated on __FMA__ separately from the tier. Build with -march=x86-64-v3 (or
// -mavx2 -mfma) to get genuinely fused multiply-add; see the divergence notes.
#if defined(__AVX2__)
#define SIMD_BACKEND_X86_ 1
#define SIMD_MAX_WIDTH_ 2
#elif defined(__SSE4_2__)
#define SIMD_BACKEND_X86_ 1
#define SIMD_MAX_WIDTH_ 1
#else
// Below SSE4.2 the integer compares, blends and unsigned min/max this backend
// is built on are unavailable, so drop to the scalar path rather than emulate.
#define SIMD_MAX_WIDTH_ 0
#endif

#else

#define SIMD_MAX_WIDTH_ 0

#endif

// Backends, build flags, and what still differs between them
//
// Tier 2  AVX2         f32x8 u32x8 f64x4 u64x4, plus everything in tier 1
// Tier 1  NEON         f32x4 u32x4 f64x2 u64x2   (aarch64, __ARM_NEON)
// Tier 1  SSE4.2       f32x4 u32x4 f64x2 u64x2   (x86-64)
// Tier 0  scalar       f32x1 u32x1 f64x1 u64x1   (always available)
//
// On x86 build with -march=x86-64-v3 (or -mavx2 -mfma) for the 256-bit tier.
// Note that -mavx2 on its own does NOT define __FMA__, and fma/fms/fms2 are
// gated on __FMA__ rather than on the tier, so that combination silently costs
// you fused multiply-add. Below SSE4.2 x86 drops to the scalar path.
//
// The same source gives the same answers on every backend, with these
// exceptions, all of which are properties of the hardware rather than choices:
//
//   fma, fms, fms2
//       Fused, with a single rounding, on NEON and on x86 with __FMA__.
//       Without __FMA__ the x86 path falls back to a * b + c, which rounds
//       twice. Results can differ by an ulp; nothing else changes.
//
//   min, max  (float only)
//       NEON's FMIN/FMAX propagate NaN and order signed zeros. x86 and the
//       scalar backend both return the second operand when either input is
//       NaN, and for min(+0, -0) return whichever operand came second. If a
//       lane can be NaN and the choice matters, compare and select explicitly.
//
//   recip, rsqrt
//       Approximations on NEON (an estimate refined by Newton-Raphson) and
//       correctly rounded on x86, which has no usable estimate: its rcp/rsqrt
//       flush denormal inputs to zero, so no amount of refinement recovers
//       them. Treat both as approximate to within a few ulp.
//
// Everything else -- including the saturating float-to-unsigned cast, the
// canonical form of every mask, and the masked loads and stores -- is defined
// to behave identically everywhere.

namespace simd {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

using f32 = float;
using f64 = double;

using usize = size_t;

using width_type = usize;

template <typename T>
concept ScalarType =
    (std::integral<T> || std::floating_point<T>) && !std::same_as<T, bool>;

template <typename V>
concept VecType = requires {
    typename V::scalar_type;
    { V::width } -> std::convertible_to<width_type>;
};

template <typename Number, usize Width>
struct vec;

template <VecType V>
constexpr V splat(typename V::scalar_type number);

template <typename N, width_type W>
constexpr vec<N, W> splat(N s) { return splat<vec<N, W>>(s); }

// a * b + c
template <VecType V>
inline V fma(V a, V b, V c);

// a * b - c
template <VecType V>
inline V fms(V a, V b, V c);

// -(a * b) + c
template <VecType V>
inline V fms2(V a, V b, V c);

template <VecType V>
inline V load(const typename V::scalar_type *ptr);

template <VecType V>
inline void store(typename V::scalar_type *ptr, V v);

// Masked memory access
//
// The mask says which lanes take part; mask_first_n is the one to reach for at
// the end of a loop whose length is not a multiple of the width, and
// simd_loop_count gives the count for the whole-vector part before it.
//
// Argument order follows select(mask, a, b) and select_add(mask, src, ...):
// the mask leads, and src -- the value inactive lanes keep -- comes next.

// Lanes where the mask is set are read from ptr; the rest keep src. No memory
// outside the set lanes is touched, so this is safe right up against the end
// of a page, which is the whole point of a tail mask. Backends with no
// hardware masked load walk the set lanes one at a time.
template <VecType V>
inline V load_masked(typename V::mask_type mask, V src,
                     const typename V::scalar_type *ptr);

// The same result, reading the whole vector and blending: one load and one
// select on every backend, but it touches all V::width elements at ptr. Use it
// only when that many are known to be readable -- a padded buffer, or a tail
// that is not the last thing in its allocation.
template <VecType V>
inline V load_masked_unsafe(typename V::mask_type mask, V src,
                            const typename V::scalar_type *ptr);

// Writes only the lanes where the mask is set. There is deliberately no
// unsafe counterpart: reading past the end returns junk you then discard,
// whereas writing past it destroys someone else's data.
template <VecType V>
inline void store_masked(typename V::scalar_type *ptr, typename V::mask_type mask,
                         V v);

// Lane access
//
// load_lane<i>, store_lane<i>, get_lane<i> and set_lane<i> each take the lane
// index as their first template argument and are provided by every backend,
// per vec type, with a static_assert that the index is in range.
//
// They are deliberately not declared here as a `template <width_type lane,
// VecType V>` primary. Such a primary can never be defined -- every backend
// reaches for a different intrinsic, and an intrinsic's lane argument must be
// a compile-time constant of that backend's own type -- so a primary would
// only ever be a declaration that swallows calls spelled get_lane<0, f32x4>(v)
// and turns them into link errors. Writing them as plain per-type overloads
// means get_lane<0>(v) resolves and get_lane<0, f32x4>(v) is a compile error,
// which is the honest pair of outcomes. The scalar backend forces the same
// shape on any/all/none/movemask, whose mask is a bool rather than a VecType.

template <VecType V>
constexpr V min(V a, V b);

template <VecType V>
constexpr V max(V a, V b);

template <VecType V>
constexpr V abs(V v);

template <VecType V>
constexpr V sqrt(V v);

template <VecType V>
constexpr V recip(V v);

template <VecType V>
constexpr V rsqrt(V v);

template <VecType V>
constexpr V round(V v);

template <VecType V>
constexpr V floor(V v);

template <VecType V>
constexpr V ceil(V v);

template <VecType V>
constexpr V trunc(V v);

template <VecType V>
constexpr V select(typename V::mask_type mask, V a, V b);

// Fused select-arithmetic: result lane = mask ? op(...) : src.
// On NEON these compute-then-blend; on x86 they can map to
// merge-masked arithmetic intrinsics (e.g. _mm_mask_add_ps).

// mask ? a + b : src
template <VecType V>
constexpr V select_add(typename V::mask_type mask, V src, V a, V b);

// mask ? a - b : src
template <VecType V>
constexpr V select_sub(typename V::mask_type mask, V src, V a, V b);

// mask ? a * b : src
template <VecType V>
constexpr V select_mul(typename V::mask_type mask, V src, V a, V b);

// mask ? a / b : src
template <VecType V>
constexpr V select_div(typename V::mask_type mask, V src, V a, V b);

// mask ? min(a, b) : src
template <VecType V>
constexpr V select_min(typename V::mask_type mask, V src, V a, V b);

// mask ? max(a, b) : src
template <VecType V>
constexpr V select_max(typename V::mask_type mask, V src, V a, V b);

// mask ? a * b + c : src
template <VecType V>
inline V select_fma(typename V::mask_type mask, V src, V a, V b, V c);

// mask ? a * b - c : src
template <VecType V>
inline V select_fms(typename V::mask_type mask, V src, V a, V b, V c);

// mask ? -(a * b) + c : src
template <VecType V>
inline V select_fms2(typename V::mask_type mask, V src, V a, V b, V c);

template <VecType V>
constexpr typename V::scalar_type hsum(V v);

template <VecType V>
constexpr typename V::scalar_type hmin(V v);

template <VecType V>
constexpr typename V::scalar_type hmax(V v);

// clamp(v, lo, hi) = min(max(v, lo), hi)
template <VecType V>
constexpr V clamp(V v, V lo, V hi);

// dot(a, b) = hsum(a * b)
template <VecType V>
constexpr typename V::scalar_type dot(V a, V b);

// True if any lane of the mask is set.
template <VecType M>
inline bool any(M mask);

// True if every lane of the mask is set.
template <VecType M>
inline bool all(M mask);

// True if no lane of the mask is set.
template <VecType M>
inline bool none(M mask);

// Packs lane i's truth value into bit i of the result. Expects a canonical
// mask (see below); mask_from_bits is the inverse.
template <VecType M>
inline u32 movemask(M mask);

// Number of elements covered by whole vector iterations: n rounded down to a
// multiple of V::width. The n - simd_loop_count<V>(n) elements left over are
// the tail, which mask_first_n can mask off.
template <VecType V>
constexpr usize simd_loop_count(usize n) {
    return n - (n % V::width);
}

// Masks
//
// A mask lane is *canonical* when all of its bits are set (true) or all are
// clear (false). Comparisons produce canonical masks, and everything that
// consumes a mask depends on it: select blends bit by bit, so a lane holding 1
// rather than ~0 mixes the two inputs instead of choosing one, and all() and
// movemask() would likewise read a half-set lane as false. Build masks with
// comparisons or with the helpers below, never with splat(1).

// Lane indices 0, 1, 2, ... as a vector of V's scalar type. Comparing against
// it is the general way to derive a mask that is canonical by construction,
// e.g. iota<u32x4>() < splat<u32x4>(n).
template <VecType V>
constexpr V iota();

// Lane i takes bit i of `bits`; bits at or above V::width are ignored.
// The inverse of movemask.
template <VecType V>
constexpr typename V::mask_type mask_from_bits(u32 bits);

// Lane-wise mask logic. These exist because a mask is not a vector on every
// backend -- the scalar one uses bool, where `~m` promotes to int and yields
// the wrong answer -- so no single operator spelling is portable.
template <VecType M>
inline M mask_and(M a, M b);

template <VecType M>
inline M mask_or(M a, M b);

template <VecType M>
inline M mask_xor(M a, M b);

template <VecType M>
inline M mask_not(M m);

// a & ~b
template <VecType M>
inline M mask_andnot(M a, M b);

// Every lane true / every lane false. Derived from a comparison, so the result
// is canonical whatever the backend's mask representation happens to be.
template <VecType V>
constexpr typename V::mask_type mask_all() {
    const V z = splat<V>(typename V::scalar_type{0});
    return z == z;
}

template <VecType V>
constexpr typename V::mask_type mask_none() {
    const V z = splat<V>(typename V::scalar_type{0});
    return z != z;
}

// Broadcast one bool to every lane.
template <VecType V>
constexpr typename V::mask_type mask_from_bool(bool b) {
    return b ? mask_all<V>() : mask_none<V>();
}

// Lanes [0, n) true, the rest false, with n clamped to V::width. This is the
// mask for the tail of a loop whose length is not a multiple of the width.
template <VecType V>
constexpr typename V::mask_type mask_first_n(usize n) {
    const usize k = n < V::width ? n : V::width;
    if constexpr (VecType<typename V::mask_type>) {
        // Compare in the mask's own integer domain: for a float V this avoids
        // converting k to float just to compare lane indices.
        using M = typename V::mask_type;
        return iota<M>() < splat<M>(static_cast<typename M::scalar_type>(k));
    } else {
        return k != 0;   // scalar backend: the mask is a bool
    }
}

// Lane j takes bit (offset + j) of `bits` -- mask_from_bits reading a window
// further up the word, so a dense bitset can be walked a vector at a time:
//
//     for (usize i = 0; i < n; i += V::width) {
//         auto m = mask_from_bits_at<V>(words[i / 32], i % 32);
//         ...
//     }
//
// Bit indices of 32 and above read as false, so an offset past the end of the
// word gives an all-false mask rather than shifting by an invalid amount. With
// a width that divides 32 (all of them here) an offset that is a multiple of
// the width never straddles two words; otherwise combine two calls yourself.
template <VecType V>
constexpr typename V::mask_type mask_from_bits_at(u32 bits, u32 offset) {
    return mask_from_bits<V>(offset >= 32 ? 0u : bits >> offset);
}

// How a single mask lane is stored in memory, for building a mask one lane at a
// time before loading it as a vector. The element type is not V::mask_type: a
// vector mask is an array of its own scalar type, and at width 1 the mask is a
// bool, which no load overload accepts -- mask_storage::load papers over both.
//
//     using MS = mask_storage<V>;
//     MS::element buf[N];
//     buf[i] = cond ? MS::true_value : MS::false_value;   // one lane at a time
//     typename V::mask_type m = MS::load(buf + i);        // then vector-load
//
// true_value is every bit set, the canonical form select and movemask expect.
template <VecType V, bool VectorMask = VecType<typename V::mask_type>>
struct mask_storage;

template <VecType V>
struct mask_storage<V, true> {
    using mask = typename V::mask_type;
    using element = typename mask::scalar_type;

    static constexpr element true_value =
        static_cast<element>(~static_cast<element>(0));
    static constexpr element false_value = element(0);

    static mask load(const element *ptr) { return simd::load<mask>(ptr); }
    static void store(element *ptr, mask m) { simd::store<mask>(ptr, m); }
};

template <VecType V>
struct mask_storage<V, false> {
    using mask = typename V::mask_type;   // bool at width 1
    using element = bool;

    static constexpr element true_value = true;
    static constexpr element false_value = false;

    static mask load(const element *ptr) { return *ptr; }
    static void store(element *ptr, mask m) { *ptr = m; }
};

// Value-preserving lane-wise conversion (e.g. f32 -> u32 truncation).
//
// Float to unsigned truncates toward zero and saturates on every backend:
// negatives and NaN give 0, and anything at or above the target's range gives
// its maximum. That costs a few instructions on x86, which has no unsigned
// convert below AVX-512, but it is the only way the same code gives the same
// answers everywhere -- a bare static_cast would be undefined behaviour.
template <VecType To, VecType From>
constexpr To cast(From v);

// Bit-preserving reinterpretation of the underlying lanes.
template <VecType To, VecType From>
constexpr To reinterpret(From v);

// Compound assignment
//
// Defined once for every vec type rather than per specialization, and each one
// is constrained to exist exactly where the matching binary operator does: so
// f32x4 gets /= but not &=, and u32x4 the reverse. The constraint demands that
// the binary operator yield V itself, which keeps a compiler's own vector
// extensions out of the overload set -- clang, unlike gcc, will happily divide
// two uint32x4_t and hand back a raw vector, and admitting that would make
// u32x4 /= compile on one compiler only.


template <VecType V>
    requires requires(V a, V b) { { a + b } -> std::same_as<V>; }
constexpr V &operator+=(V &a, V b) {
    return a = a + b;
}

template <VecType V>
    requires requires(V a, typename V::scalar_type b) { { a + b } -> std::same_as<V>; }
constexpr V &operator+=(V &a, typename V::scalar_type b) {
    return a = a + b;
}

template <VecType V>
    requires requires(V a, V b) { { a - b } -> std::same_as<V>; }
constexpr V &operator-=(V &a, V b) {
    return a = a - b;
}

template <VecType V>
    requires requires(V a, typename V::scalar_type b) { { a - b } -> std::same_as<V>; }
constexpr V &operator-=(V &a, typename V::scalar_type b) {
    return a = a - b;
}

template <VecType V>
    requires requires(V a, V b) { { a * b } -> std::same_as<V>; }
constexpr V &operator*=(V &a, V b) {
    return a = a * b;
}

template <VecType V>
    requires requires(V a, typename V::scalar_type b) { { a * b } -> std::same_as<V>; }
constexpr V &operator*=(V &a, typename V::scalar_type b) {
    return a = a * b;
}

template <VecType V>
    requires requires(V a, V b) { { a / b } -> std::same_as<V>; }
constexpr V &operator/=(V &a, V b) {
    return a = a / b;
}

template <VecType V>
    requires requires(V a, typename V::scalar_type b) { { a / b } -> std::same_as<V>; }
constexpr V &operator/=(V &a, typename V::scalar_type b) {
    return a = a / b;
}

template <VecType V>
    requires requires(V a, V b) { { a & b } -> std::same_as<V>; }
constexpr V &operator&=(V &a, V b) {
    return a = a & b;
}

template <VecType V>
    requires requires(V a, V b) { { a | b } -> std::same_as<V>; }
constexpr V &operator|=(V &a, V b) {
    return a = a | b;
}

template <VecType V>
    requires requires(V a, V b) { { a ^ b } -> std::same_as<V>; }
constexpr V &operator^=(V &a, V b) {
    return a = a ^ b;
}

template <VecType V>
    requires requires(V a, int b) { { a << b } -> std::same_as<V>; }
constexpr V &operator<<=(V &a, int b) {
    return a = a << b;
}

template <VecType V>
    requires requires(V a, int b) { { a >> b } -> std::same_as<V>; }
constexpr V &operator>>=(V &a, int b) {
    return a = a >> b;
}

// Composed operations
//
// Everything below is written purely in terms of other simd:: entry points, so
// the text is identical for every vec type on every backend -- only the type
// name changes. Emitting them from a macro once per type is what keeps adding
// a new width cheap; anything backed by an actual intrinsic stays spelled out
// so it remains greppable.

// clamp and dot: available for every vec type.
#define SIMD_DEFINE_COMPOSED_(V)                                               \
    template <>                                                                \
    inline V clamp(V v, V lo, V hi) {                                          \
        return min<V>(max<V>(v, lo), hi);                                      \
    }                                                                          \
    template <>                                                                \
    inline V::scalar_type dot(V a, V b) {                                       \
        return hsum<V>(a * b);                                                 \
    }

// The fused select-arithmetic family. Float types only: it includes division,
// which the integer vec types deliberately do not provide.
#define SIMD_DEFINE_SELECT_OPS_(V)                                             \
    template <>                                                                \
    inline V select_add(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, a + b, src);                                       \
    }                                                                          \
    template <>                                                                \
    inline V select_sub(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, a - b, src);                                       \
    }                                                                          \
    template <>                                                                \
    inline V select_mul(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, a * b, src);                                       \
    }                                                                          \
    template <>                                                                \
    inline V select_div(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, a / b, src);                                       \
    }                                                                          \
    template <>                                                                \
    inline V select_min(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, min<V>(a, b), src);                                \
    }                                                                          \
    template <>                                                                \
    inline V select_max(V::mask_type m, V src, V a, V b) {                     \
        return select<V>(m, max<V>(a, b), src);                                \
    }                                                                          \
    template <>                                                                \
    inline V select_fma(V::mask_type m, V src, V a, V b, V c) {                \
        return select<V>(m, fma<V>(a, b, c), src);                             \
    }                                                                          \
    template <>                                                                \
    inline V select_fms(V::mask_type m, V src, V a, V b, V c) {                \
        return select<V>(m, fms<V>(a, b, c), src);                             \
    }                                                                          \
    template <>                                                                \
    inline V select_fms2(V::mask_type m, V src, V a, V b, V c) {               \
        return select<V>(m, fms2<V>(a, b, c), src);                            \
    }

// Lane-wise mask logic, for an integer vec type used as a mask. none() lives
// here too: it is the one mask query with no intrinsic behind it.
#define SIMD_DEFINE_MASK_OPS_(M)                                               \
    template <>                                                                \
    inline bool none(M mask) {                                                 \
        return !any<M>(mask);                                                  \
    }                                                                          \
    template <>                                                                \
    inline M mask_and(M a, M b) {                                              \
        return a & b;                                                          \
    }                                                                          \
    template <>                                                                \
    inline M mask_or(M a, M b) {                                               \
        return a | b;                                                          \
    }                                                                          \
    template <>                                                                \
    inline M mask_xor(M a, M b) {                                              \
        return a ^ b;                                                          \
    }                                                                          \
    template <>                                                                \
    inline M mask_not(M m) {                                                   \
        return ~m;                                                             \
    }                                                                          \
    template <>                                                                \
    inline M mask_andnot(M a, M b) {                                           \
        return a & ~b;                                                         \
    }

// Implementation
//
// constexpr appears only where an implementation is genuinely constant-
// evaluable, which today means the scalar backend. Intrinsic-backed
// specializations are `inline` instead: compiler intrinsics are not constant
// expressions, so marking them constexpr would be a promise the code cannot
// keep. Explicit specializations may drop the primary template's constexpr.

#if SIMD_MAX_WIDTH_ >= 1

using f32x4 = vec<f32, 4>;
using u32x4 = vec<u32, 4>;

using f64x2 = vec<f64, 2>;
using u64x2 = vec<u64, 2>;

template <>
struct vec<f32, 4> {
    using scalar_type = f32;
    using mask_type = u32x4;
    static constexpr width_type width = 4;

#if defined(SIMD_BACKEND_NEON_)
    using simd_type = float32x4_t;
#elif defined(SIMD_BACKEND_X86_)
    using simd_type = __m128;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.f) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const {
#if defined(SIMD_BACKEND_NEON_)
        return vnegq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_ps(v, _mm_set1_ps(-0.f));
#endif
    }

    This operator+(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vaddq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_add_ps(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vsubq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sub_ps(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vmulq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_mul_ps(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vdivq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_div_ps(*this, o);
#endif
    }
    This operator/(scalar_type s) const;
    friend This operator/(scalar_type s, This o);

    mask_type operator==(This o) const;
    mask_type operator!=(This o) const;
    mask_type operator<(This o) const;
    mask_type operator<=(This o) const;
    mask_type operator>(This o) const;
    mask_type operator>=(This o) const;

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o);
    friend mask_type operator!=(scalar_type s, This o);
    friend mask_type operator<(scalar_type s, This o);
    friend mask_type operator<=(scalar_type s, This o);
    friend mask_type operator>(scalar_type s, This o);
    friend mask_type operator>=(scalar_type s, This o);
};

static_assert(VecType<f32x4>, "vec<f32, 4> does not satisfy VecType");

template <>
struct vec<u32, 4> {
    using scalar_type = u32;
    using mask_type = vec<u32, 4>;
    static constexpr width_type width = 4;

#if defined(SIMD_BACKEND_NEON_)
    using simd_type = uint32x4_t;
#elif defined(SIMD_BACKEND_X86_)
    using simd_type = __m128i;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vaddq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_add_epi32(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vsubq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sub_epi32(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vmulq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_mullo_epi32(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vceqq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_cmpeq_epi32(*this, o);
#endif
    }
    mask_type operator!=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vmvnq_u32(vceqq_u32(*this, o));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(_mm_cmpeq_epi32(*this, o), _mm_set1_epi32(-1));
#endif
    }
    mask_type operator<(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcltq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        // a < b  <=>  !(max(a, b) == a)
        return _mm_xor_si128(_mm_cmpeq_epi32(_mm_max_epu32(*this, o), *this),
                             _mm_set1_epi32(-1));
#endif
    }
    mask_type operator<=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcleq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        // a <= b  <=>  min(a, b) == a
        return _mm_cmpeq_epi32(_mm_min_epu32(*this, o), *this);
#endif
    }
    mask_type operator>(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcgtq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        // a > b  <=>  !(min(a, b) == a)
        return _mm_xor_si128(_mm_cmpeq_epi32(_mm_min_epu32(*this, o), *this),
                             _mm_set1_epi32(-1));
#endif
    }
    mask_type operator>=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcgeq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        // a >= b  <=>  max(a, b) == a
        return _mm_cmpeq_epi32(_mm_max_epu32(*this, o), *this);
#endif
    }

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o) { return o == s; }
    friend mask_type operator!=(scalar_type s, This o) { return o != s; }
    friend mask_type operator<(scalar_type s, This o) { return o > s; }
    friend mask_type operator<=(scalar_type s, This o) { return o >= s; }
    friend mask_type operator>(scalar_type s, This o) { return o < s; }
    friend mask_type operator>=(scalar_type s, This o) { return o <= s; }

    This operator&(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vandq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_and_si128(*this, o);
#endif
    }
    This operator|(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vorrq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_or_si128(*this, o);
#endif
    }
    This operator^(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return veorq_u32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(*this, o);
#endif
    }
    This operator~() const {
#if defined(SIMD_BACKEND_NEON_)
        return vmvnq_u32(v);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(v, _mm_set1_epi32(-1));
#endif
    }

    This operator<<(int n) const {
#if defined(SIMD_BACKEND_NEON_)
        return vshlq_u32(*this, vdupq_n_s32(n));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sll_epi32(*this, _mm_cvtsi32_si128(n));
#endif
    }
    This operator>>(int n) const {
#if defined(SIMD_BACKEND_NEON_)
        return vshlq_u32(*this, vdupq_n_s32(-n));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_srl_epi32(*this, _mm_cvtsi32_si128(n));
#endif
    }
};

static_assert(VecType<u32x4>, "vec<u32, 4> does not satisfy VecType");

inline f32x4::mask_type f32x4::operator==(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vceqq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmpeq_ps(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator!=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vmvnq_u32(vceqq_f32(*this, o));
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmpneq_ps(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator<(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcltq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmplt_ps(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator<=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcleq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmple_ps(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator>(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcgtq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmpgt_ps(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator>=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcgeq_f32(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(_mm_cmpge_ps(*this, o));
#endif
}

template <>
inline f32x4 splat<f32x4>(f32x4::scalar_type s) {
#if defined(SIMD_BACKEND_NEON_)
    return vdupq_n_f32(s);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_set1_ps(s);
#endif
}

inline f32x4::vec(scalar_type s) : v(splat<This>(s)) {}

inline f32x4 f32x4::operator+(scalar_type o) const {
    return *this + splat<This>(o);
}
inline f32x4 f32x4::operator-(scalar_type o) const {
    return *this - splat<This>(o);
}
inline f32x4 operator-(f32x4::scalar_type s, f32x4 o) {
    return splat<f32x4>(s) - o;
}
inline f32x4 f32x4::operator*(scalar_type o) const {
    return *this * splat<This>(o);
}
inline f32x4 f32x4::operator/(scalar_type o) const {
    return *this / splat<This>(o);
}
inline f32x4 operator/(f32x4::scalar_type s, f32x4 o) {
    return splat<f32x4>(s) / o;
}

inline f32x4::mask_type f32x4::operator==(scalar_type o) const {
    return *this == splat<This>(o);
}
inline f32x4::mask_type f32x4::operator!=(scalar_type o) const {
    return *this != splat<This>(o);
}
inline f32x4::mask_type f32x4::operator<(scalar_type o) const {
    return *this < splat<This>(o);
}
inline f32x4::mask_type f32x4::operator<=(scalar_type o) const {
    return *this <= splat<This>(o);
}
inline f32x4::mask_type f32x4::operator>(scalar_type o) const {
    return *this > splat<This>(o);
}
inline f32x4::mask_type f32x4::operator>=(scalar_type o) const {
    return *this >= splat<This>(o);
}

inline f32x4::mask_type operator==(f32x4::scalar_type s, f32x4 o) {
    return o == s;
}
inline f32x4::mask_type operator!=(f32x4::scalar_type s, f32x4 o) {
    return o != s;
}
inline f32x4::mask_type operator<(f32x4::scalar_type s, f32x4 o) {
    return o > s;
}
inline f32x4::mask_type operator<=(f32x4::scalar_type s, f32x4 o) {
    return o >= s;
}
inline f32x4::mask_type operator>(f32x4::scalar_type s, f32x4 o) {
    return o < s;
}
inline f32x4::mask_type operator>=(f32x4::scalar_type s, f32x4 o) {
    return o <= s;
}

template <>
inline u32x4 splat<u32x4>(u32x4::scalar_type s) {
#if defined(SIMD_BACKEND_NEON_)
    return vdupq_n_u32(s);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_set1_epi32(static_cast<int>(s));
#endif
}

inline u32x4::vec(scalar_type s) : v(splat<This>(s)) {}

inline u32x4 u32x4::operator+(scalar_type o) const {
    return *this + splat<This>(o);
}
inline u32x4 u32x4::operator-(scalar_type o) const {
    return *this - splat<This>(o);
}
inline u32x4 operator-(u32x4::scalar_type s, u32x4 o) {
    return splat<u32x4>(s) - o;
}
inline u32x4 u32x4::operator*(scalar_type o) const {
    return *this * splat<This>(o);
}

inline u32x4::mask_type u32x4::operator==(scalar_type o) const {
    return *this == splat<This>(o);
}
inline u32x4::mask_type u32x4::operator!=(scalar_type o) const {
    return *this != splat<This>(o);
}
inline u32x4::mask_type u32x4::operator<(scalar_type o) const {
    return *this < splat<This>(o);
}
inline u32x4::mask_type u32x4::operator<=(scalar_type o) const {
    return *this <= splat<This>(o);
}
inline u32x4::mask_type u32x4::operator>(scalar_type o) const {
    return *this > splat<This>(o);
}
inline u32x4::mask_type u32x4::operator>=(scalar_type o) const {
    return *this >= splat<This>(o);
}

// a * b + c
template <>
inline f32x4 fma(f32x4 a, f32x4 b, f32x4 c) {
#if defined(SIMD_BACKEND_NEON_)
    return vfmaq_f32(c, a, b);
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fmadd_ps(a, b, c);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f32x4 fms(f32x4 a, f32x4 b, f32x4 c) {
#if defined(SIMD_BACKEND_NEON_)
    // Negation is exact, so this keeps the single rounding of the fused op.
    return vnegq_f32(vfmsq_f32(c, a, b));
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fmsub_ps(a, b, c);
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f32x4 fms2(f32x4 a, f32x4 b, f32x4 c) {
#if defined(SIMD_BACKEND_NEON_)
    return vfmsq_f32(c, a, b);
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fnmadd_ps(a, b, c);
#else
    return c - a * b;
#endif
}

template <>
inline f32x4 load(const f32x4::scalar_type *ptr) {
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_f32(ptr);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_loadu_ps(ptr);
#endif
}

template <width_type lane>
inline f32x4 load_lane(f32x4 v, const f32x4::scalar_type *ptr) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_lane_f32(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_ps(v, _mm_load_ss(ptr), lane << 4);
#endif
}

template <>
inline void store(f32x4::scalar_type *ptr, f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    vst1q_f32(ptr, v);
#elif defined(SIMD_BACKEND_X86_)
    _mm_storeu_ps(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(f32x4::scalar_type *ptr, f32x4 v) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    vst1q_lane_f32(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    _mm_store_ss(ptr, _mm_shuffle_ps(v, v, _MM_SHUFFLE(lane, lane, lane, lane)));
#endif
}

template <width_type lane>
inline f32x4::scalar_type get_lane(f32x4 v) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vgetq_lane_f32(v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(lane, lane, lane, lane)));
#endif
}

template <width_type lane>
inline f32x4 set_lane(f32x4 v, f32x4::scalar_type s) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vsetq_lane_f32(s, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_ps(v, _mm_set_ss(s), lane << 4);
#endif
}

template <>
inline u32x4 load(const u32x4::scalar_type *ptr) {
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_u32(ptr);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
#endif
}

template <width_type lane>
inline u32x4 load_lane(u32x4 v, const u32x4::scalar_type *ptr) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_lane_u32(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_epi32(v, static_cast<int>(*ptr), lane);
#endif
}

template <>
inline void store(u32x4::scalar_type *ptr, u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    vst1q_u32(ptr, v);
#elif defined(SIMD_BACKEND_X86_)
    _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), v);
#endif
}

template <width_type lane>
inline void store_lane(u32x4::scalar_type *ptr, u32x4 v) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    vst1q_lane_u32(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    *ptr = static_cast<u32>(_mm_extract_epi32(v, lane));
#endif
}

template <width_type lane>
inline u32x4::scalar_type get_lane(u32x4 v) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vgetq_lane_u32(v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return static_cast<u32>(_mm_extract_epi32(v, lane));
#endif
}

template <width_type lane>
inline u32x4 set_lane(u32x4 v, u32x4::scalar_type s) {
    static_assert(lane < 4);
#if defined(SIMD_BACKEND_NEON_)
    return vsetq_lane_u32(s, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_epi32(v, static_cast<int>(s), lane);
#endif
}

template <>
inline f32x4 min(f32x4 a, f32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vminq_f32(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_min_ps(a, b);
#endif
}

template <>
inline u32x4 min(u32x4 a, u32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vminq_u32(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_min_epu32(a, b);
#endif
}

template <>
inline f32x4 max(f32x4 a, f32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxq_f32(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_max_ps(a, b);
#endif
}

template <>
inline u32x4 max(u32x4 a, u32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxq_u32(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_max_epu32(a, b);
#endif
}

template <>
inline f32x4 abs(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vabsq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_andnot_ps(_mm_set1_ps(-0.f), v);
#endif
}

template <>
inline f32x4 sqrt(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vsqrtq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_sqrt_ps(v);
#endif
}

template <>
inline f32x4 recip(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    f32x4 e = vrecpeq_f32(v);
    e = vmulq_f32(vrecpsq_f32(v, e), e);
    e = vmulq_f32(vrecpsq_f32(v, e), e);
    return e;
#elif defined(SIMD_BACKEND_X86_)
    // Deliberately a plain divide rather than _mm_rcp_ps plus Newton-Raphson.
    // The NEON path above refines to full f32 precision, so that is the
    // contract to meet, and the estimate cannot meet it: RCPPS flushes
    // denormal inputs to zero, and the refinement term v * e then comes out
    // non-finite, which poisons the lane rather than correcting it. A divide
    // is correctly rounded, handles zero, infinity and denormals, and on
    // current cores is within a few cycles of the estimate-plus-refine
    // sequence anyway. Same reasoning as the f64x2 path below.
    return _mm_div_ps(_mm_set1_ps(1.f), v);
#endif
}

template <>
inline f32x4 rsqrt(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    f32x4 e = vrsqrteq_f32(v);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(v, e), e), e);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(v, e), e), e);
    return e;
#elif defined(SIMD_BACKEND_X86_)
    // Same reasoning as recip: RSQRTPS flushes denormal inputs to zero, so for
    // those its estimate is infinity where the true result is an ordinary
    // finite number, and no amount of refinement recovers it.
    return _mm_div_ps(_mm_set1_ps(1.f), _mm_sqrt_ps(v));
#endif
}

template <>
inline f32x4 round(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndnq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
}

template <>
inline f32x4 floor(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndmq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_floor_ps(v);
#endif
}

template <>
inline f32x4 ceil(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndpq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_ceil_ps(v);
#endif
}

template <>
inline f32x4 trunc(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#endif
}

template <>
inline f32x4 select(f32x4::mask_type mask, f32x4 a, f32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vbslq_f32(mask, a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_blendv_ps(b, a, _mm_castsi128_ps(mask));
#endif
}

template <>
inline u32x4 select(u32x4::mask_type mask, u32x4 a, u32x4 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vbslq_u32(mask, a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_blendv_epi8(b, a, mask);
#endif
}

template <>
inline f32x4::scalar_type hsum(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vaddvq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128 t = _mm_add_ps(v, _mm_movehl_ps(v, v));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
#endif
}

template <>
inline u32x4::scalar_type hsum(u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vaddvq_u32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128i t = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
#endif
}

template <>
inline f32x4::scalar_type hmin(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vminvq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128 t = _mm_min_ps(v, _mm_movehl_ps(v, v));
    t = _mm_min_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
#endif
}

template <>
inline u32x4::scalar_type hmin(u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vminvq_u32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128i t = _mm_min_epu32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_min_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
#endif
}

template <>
inline f32x4::scalar_type hmax(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxvq_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128 t = _mm_max_ps(v, _mm_movehl_ps(v, v));
    t = _mm_max_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
#endif
}

template <>
inline u32x4::scalar_type hmax(u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxvq_u32(v);
#elif defined(SIMD_BACKEND_X86_)
    __m128i t = _mm_max_epu32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_max_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
#endif
}

template <>
inline u32x4 cast<u32x4, f32x4>(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vcvtq_u32_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    // x86 has no packed f32 -> u32 convert before AVX-512, only the signed
    // _mm_cvttps_epi32, so this reproduces NEON's saturating vcvtq_u32_f32 in
    // three parts: _mm_max_ps sends negatives and NaN to 0 (max returns its
    // second operand for NaN), inputs at or above 2^31 are converted against a
    // biased input to dodge the signed range, and the top end saturates.
    const __m128 bias = _mm_set1_ps(2147483648.f);
    const __m128 x = _mm_max_ps(v, _mm_setzero_ps());
    const __m128 high = _mm_cmpge_ps(x, bias);
    const __m128 over = _mm_cmpge_ps(x, _mm_set1_ps(4294967296.f));
    const __m128i lo = _mm_cvttps_epi32(x);
    const __m128i hi = _mm_add_epi32(_mm_cvttps_epi32(_mm_sub_ps(x, bias)),
                                     _mm_set1_epi32(INT32_MIN));
    const __m128 r = _mm_blendv_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), high);
    return _mm_castps_si128(
        _mm_blendv_ps(r, _mm_castsi128_ps(_mm_set1_epi32(-1)), over));
#endif
}

template <>
inline f32x4 cast<f32x4, u32x4>(u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vcvtq_f32_u32(v);
#elif defined(SIMD_BACKEND_X86_)
    // No packed u32 -> f32 either. Both 16-bit halves convert exactly through
    // the signed path, and the recombining add rounds once, so the result is
    // correctly rounded across the whole u32 range.
    const __m128 lo = _mm_cvtepi32_ps(_mm_and_si128(v, _mm_set1_epi32(0xFFFF)));
    const __m128 hi = _mm_cvtepi32_ps(_mm_srli_epi32(v, 16));
    return _mm_add_ps(lo, _mm_mul_ps(hi, _mm_set1_ps(65536.f)));
#endif
}

template <>
inline u32x4 reinterpret<u32x4, f32x4>(f32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vreinterpretq_u32_f32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castps_si128(v);
#endif
}

template <>
inline f32x4 reinterpret<f32x4, u32x4>(u32x4 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vreinterpretq_f32_u32(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castsi128_ps(v);
#endif
}

template <>
inline bool any(u32x4 mask) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxvq_u32(mask) != 0;
#elif defined(SIMD_BACKEND_X86_)
    return !_mm_testz_si128(mask, mask);
#endif
}

template <>
inline bool all(u32x4 mask) {
#if defined(SIMD_BACKEND_NEON_)
    return vminvq_u32(mask) == 0xFFFFFFFFu;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_test_all_ones(mask) != 0;
#endif
}

template <>
inline u32 movemask(u32x4 mask) {
#if defined(SIMD_BACKEND_NEON_)
    const uint32x4_t bits = {1u, 2u, 4u, 8u};
    return vaddvq_u32(vandq_u32(mask, bits));
#elif defined(SIMD_BACKEND_X86_)
    return static_cast<u32>(_mm_movemask_ps(_mm_castsi128_ps(mask)));
#endif
}


template <>
struct vec<f64, 2> {
    using scalar_type = f64;
    using mask_type = u64x2;
    static constexpr width_type width = 2;

#if defined(SIMD_BACKEND_NEON_)
    using simd_type = float64x2_t;
#elif defined(SIMD_BACKEND_X86_)
    using simd_type = __m128d;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const {
#if defined(SIMD_BACKEND_NEON_)
        return vnegq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_pd(v, _mm_set1_pd(-0.0));
#endif
    }

    This operator+(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vaddq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_add_pd(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vsubq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sub_pd(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vmulq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_mul_pd(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vdivq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_div_pd(*this, o);
#endif
    }
    This operator/(scalar_type s) const;
    friend This operator/(scalar_type s, This o);

    mask_type operator==(This o) const;
    mask_type operator!=(This o) const;
    mask_type operator<(This o) const;
    mask_type operator<=(This o) const;
    mask_type operator>(This o) const;
    mask_type operator>=(This o) const;

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o);
    friend mask_type operator!=(scalar_type s, This o);
    friend mask_type operator<(scalar_type s, This o);
    friend mask_type operator<=(scalar_type s, This o);
    friend mask_type operator>(scalar_type s, This o);
    friend mask_type operator>=(scalar_type s, This o);
};

static_assert(VecType<f64x2>, "vec<f64, 2> does not satisfy VecType");

template <>
struct vec<u64, 2> {
    using scalar_type = u64;
    using mask_type = vec<u64, 2>;
    static constexpr width_type width = 2;

#if defined(SIMD_BACKEND_NEON_)
    using simd_type = uint64x2_t;
#elif defined(SIMD_BACKEND_X86_)
    using simd_type = __m128i;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vaddq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_add_epi64(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vsubq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sub_epi64(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        // NEON has no 64-bit vector multiply; fall back to lane-wise scalars.
        scalar_type lo = vgetq_lane_u64(*this, 0) * vgetq_lane_u64(o, 0);
        scalar_type hi = vgetq_lane_u64(*this, 1) * vgetq_lane_u64(o, 1);
        return vsetq_lane_u64(hi, vdupq_n_u64(lo), 1);
#elif defined(SIMD_BACKEND_X86_)
        // x86 has no packed 64-bit multiply before AVX-512DQ either.
        scalar_type lo = static_cast<scalar_type>(_mm_cvtsi128_si64(*this)) *
                         static_cast<scalar_type>(_mm_cvtsi128_si64(o));
        scalar_type hi = static_cast<scalar_type>(_mm_extract_epi64(*this, 1)) *
                         static_cast<scalar_type>(_mm_extract_epi64(o, 1));
        return _mm_set_epi64x(static_cast<long long>(hi),
                              static_cast<long long>(lo));
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vceqq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_cmpeq_epi64(*this, o);
#endif
    }
    mask_type operator!=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        // No vmvnq_u64: invert through u32, which is bit-width agnostic.
        return vreinterpretq_u64_u32(
            vmvnq_u32(vreinterpretq_u32_u64(vceqq_u64(*this, o))));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(_mm_cmpeq_epi64(*this, o), _mm_set1_epi32(-1));
#endif
    }
    mask_type operator<(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcltq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        const __m128i bias = _mm_set1_epi64x(INT64_MIN);
        return _mm_cmpgt_epi64(_mm_xor_si128(o, bias), _mm_xor_si128(*this, bias));
#endif
    }
    mask_type operator<=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcleq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        const __m128i bias = _mm_set1_epi64x(INT64_MIN);
        return _mm_xor_si128(
            _mm_cmpgt_epi64(_mm_xor_si128(*this, bias), _mm_xor_si128(o, bias)),
            _mm_set1_epi32(-1));
#endif
    }
    mask_type operator>(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcgtq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        const __m128i bias = _mm_set1_epi64x(INT64_MIN);
        return _mm_cmpgt_epi64(_mm_xor_si128(*this, bias), _mm_xor_si128(o, bias));
#endif
    }
    mask_type operator>=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vcgeq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        const __m128i bias = _mm_set1_epi64x(INT64_MIN);
        return _mm_xor_si128(
            _mm_cmpgt_epi64(_mm_xor_si128(o, bias), _mm_xor_si128(*this, bias)),
            _mm_set1_epi32(-1));
#endif
    }

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o) { return o == s; }
    friend mask_type operator!=(scalar_type s, This o) { return o != s; }
    friend mask_type operator<(scalar_type s, This o) { return o > s; }
    friend mask_type operator<=(scalar_type s, This o) { return o >= s; }
    friend mask_type operator>(scalar_type s, This o) { return o < s; }
    friend mask_type operator>=(scalar_type s, This o) { return o <= s; }

    This operator&(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vandq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_and_si128(*this, o);
#endif
    }
    This operator|(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return vorrq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_or_si128(*this, o);
#endif
    }
    This operator^(This o) const {
#if defined(SIMD_BACKEND_NEON_)
        return veorq_u64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(*this, o);
#endif
    }
    This operator~() const {
#if defined(SIMD_BACKEND_NEON_)
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v)));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_xor_si128(v, _mm_set1_epi32(-1));
#endif
    }

    This operator<<(int n) const {
#if defined(SIMD_BACKEND_NEON_)
        return vshlq_u64(*this, vdupq_n_s64(n));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_sll_epi64(*this, _mm_cvtsi32_si128(n));
#endif
    }
    This operator>>(int n) const {
#if defined(SIMD_BACKEND_NEON_)
        return vshlq_u64(*this, vdupq_n_s64(-n));
#elif defined(SIMD_BACKEND_X86_)
        return _mm_srl_epi64(*this, _mm_cvtsi32_si128(n));
#endif
    }
};

static_assert(VecType<u64x2>, "vec<u64, 2> does not satisfy VecType");

inline f64x2::mask_type f64x2::operator==(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vceqq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmpeq_pd(*this, o));
#endif
}

inline f64x2::mask_type f64x2::operator!=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    // No vmvnq_u64: invert through u32, which is bit-width agnostic.
    return vreinterpretq_u64_u32(
        vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(*this, o))));
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmpneq_pd(*this, o));
#endif
}

inline f64x2::mask_type f64x2::operator<(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcltq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmplt_pd(*this, o));
#endif
}

inline f64x2::mask_type f64x2::operator<=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcleq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmple_pd(*this, o));
#endif
}

inline f64x2::mask_type f64x2::operator>(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcgtq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmpgt_pd(*this, o));
#endif
}

inline f64x2::mask_type f64x2::operator>=(This o) const {
#if defined(SIMD_BACKEND_NEON_)
    return vcgeq_f64(*this, o);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(_mm_cmpge_pd(*this, o));
#endif
}

template <>
inline f64x2 splat<f64x2>(f64x2::scalar_type s) {
#if defined(SIMD_BACKEND_NEON_)
    return vdupq_n_f64(s);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_set1_pd(s);
#endif
}

inline f64x2::vec(scalar_type s) : v(splat<This>(s)) {}

inline f64x2 f64x2::operator+(scalar_type o) const {
    return *this + splat<This>(o);
}

inline f64x2 f64x2::operator-(scalar_type o) const {
    return *this - splat<This>(o);
}

inline f64x2 operator-(f64x2::scalar_type s, f64x2 o) {
    return splat<f64x2>(s) - o;
}

inline f64x2 f64x2::operator*(scalar_type o) const {
    return *this * splat<This>(o);
}

inline f64x2 f64x2::operator/(scalar_type o) const {
    return *this / splat<This>(o);
}

inline f64x2 operator/(f64x2::scalar_type s, f64x2 o) {
    return splat<f64x2>(s) / o;
}

inline f64x2::mask_type f64x2::operator==(scalar_type o) const {
    return *this == splat<This>(o);
}

inline f64x2::mask_type f64x2::operator!=(scalar_type o) const {
    return *this != splat<This>(o);
}

inline f64x2::mask_type f64x2::operator<(scalar_type o) const {
    return *this < splat<This>(o);
}

inline f64x2::mask_type f64x2::operator<=(scalar_type o) const {
    return *this <= splat<This>(o);
}

inline f64x2::mask_type f64x2::operator>(scalar_type o) const {
    return *this > splat<This>(o);
}

inline f64x2::mask_type f64x2::operator>=(scalar_type o) const {
    return *this >= splat<This>(o);
}

inline f64x2::mask_type operator==(f64x2::scalar_type s, f64x2 o) {
    return o == s;
}

inline f64x2::mask_type operator!=(f64x2::scalar_type s, f64x2 o) {
    return o != s;
}

inline f64x2::mask_type operator<(f64x2::scalar_type s, f64x2 o) {
    return o > s;
}

inline f64x2::mask_type operator<=(f64x2::scalar_type s, f64x2 o) {
    return o >= s;
}

inline f64x2::mask_type operator>(f64x2::scalar_type s, f64x2 o) {
    return o < s;
}

inline f64x2::mask_type operator>=(f64x2::scalar_type s, f64x2 o) {
    return o <= s;
}

template <>
inline u64x2 splat<u64x2>(u64x2::scalar_type s) {
#if defined(SIMD_BACKEND_NEON_)
    return vdupq_n_u64(s);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_set1_epi64x(static_cast<long long>(s));
#endif
}

inline u64x2::vec(scalar_type s) : v(splat<This>(s)) {}

inline u64x2 u64x2::operator+(scalar_type o) const {
    return *this + splat<This>(o);
}

inline u64x2 u64x2::operator-(scalar_type o) const {
    return *this - splat<This>(o);
}

inline u64x2 operator-(u64x2::scalar_type s, u64x2 o) {
    return splat<u64x2>(s) - o;
}

inline u64x2 u64x2::operator*(scalar_type o) const {
    return *this * splat<This>(o);
}

inline u64x2::mask_type u64x2::operator==(scalar_type o) const {
    return *this == splat<This>(o);
}

inline u64x2::mask_type u64x2::operator!=(scalar_type o) const {
    return *this != splat<This>(o);
}

inline u64x2::mask_type u64x2::operator<(scalar_type o) const {
    return *this < splat<This>(o);
}

inline u64x2::mask_type u64x2::operator<=(scalar_type o) const {
    return *this <= splat<This>(o);
}

inline u64x2::mask_type u64x2::operator>(scalar_type o) const {
    return *this > splat<This>(o);
}

inline u64x2::mask_type u64x2::operator>=(scalar_type o) const {
    return *this >= splat<This>(o);
}

// a * b + c
template <>
inline f64x2 fma(f64x2 a, f64x2 b, f64x2 c) {
#if defined(SIMD_BACKEND_NEON_)
    return vfmaq_f64(c, a, b);
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fmadd_pd(a, b, c);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f64x2 fms(f64x2 a, f64x2 b, f64x2 c) {
#if defined(SIMD_BACKEND_NEON_)
    // Negation is exact, so this keeps the single rounding of the fused op.
    return vnegq_f64(vfmsq_f64(c, a, b));
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fmsub_pd(a, b, c);
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f64x2 fms2(f64x2 a, f64x2 b, f64x2 c) {
#if defined(SIMD_BACKEND_NEON_)
    return vfmsq_f64(c, a, b);
#elif defined(SIMD_BACKEND_X86_) && defined(__FMA__)
    return _mm_fnmadd_pd(a, b, c);
#else
    return c - a * b;
#endif
}

template <>
inline f64x2 load(const f64x2::scalar_type *ptr) {
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_f64(ptr);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_loadu_pd(ptr);
#endif
}

template <width_type lane>
inline f64x2 load_lane(f64x2 v, const f64x2::scalar_type *ptr) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_lane_f64(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    if constexpr (lane == 0) return _mm_loadl_pd(v, ptr);
    else return _mm_loadh_pd(v, ptr);
#endif
}

template <>
inline void store(f64x2::scalar_type *ptr, f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    vst1q_f64(ptr, v);
#elif defined(SIMD_BACKEND_X86_)
    _mm_storeu_pd(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(f64x2::scalar_type *ptr, f64x2 v) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    vst1q_lane_f64(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    if constexpr (lane == 0) _mm_storel_pd(ptr, v);
    else _mm_storeh_pd(ptr, v);
#endif
}

template <width_type lane>
inline f64x2::scalar_type get_lane(f64x2 v) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vgetq_lane_f64(v, lane);
#elif defined(SIMD_BACKEND_X86_)
    if constexpr (lane == 0) return _mm_cvtsd_f64(v);
    else return _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
#endif
}

template <width_type lane>
inline f64x2 set_lane(f64x2 v, f64x2::scalar_type s) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vsetq_lane_f64(s, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    if constexpr (lane == 0) return _mm_move_sd(v, _mm_set_sd(s));
    else return _mm_shuffle_pd(v, _mm_set_sd(s), 0);
#endif
}

template <>
inline u64x2 load(const u64x2::scalar_type *ptr) {
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_u64(ptr);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
#endif
}

template <width_type lane>
inline u64x2 load_lane(u64x2 v, const u64x2::scalar_type *ptr) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vld1q_lane_u64(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_epi64(v, static_cast<long long>(*ptr), lane);
#endif
}

template <>
inline void store(u64x2::scalar_type *ptr, u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    vst1q_u64(ptr, v);
#elif defined(SIMD_BACKEND_X86_)
    _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), v);
#endif
}

template <width_type lane>
inline void store_lane(u64x2::scalar_type *ptr, u64x2 v) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    vst1q_lane_u64(ptr, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    *ptr = static_cast<u64>(_mm_extract_epi64(v, lane));
#endif
}

template <width_type lane>
inline u64x2::scalar_type get_lane(u64x2 v) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vgetq_lane_u64(v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return static_cast<u64>(_mm_extract_epi64(v, lane));
#endif
}

template <width_type lane>
inline u64x2 set_lane(u64x2 v, u64x2::scalar_type s) {
    static_assert(lane < 2);
#if defined(SIMD_BACKEND_NEON_)
    return vsetq_lane_u64(s, v, lane);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_insert_epi64(v, static_cast<long long>(s), lane);
#endif
}

template <>
inline f64x2 min(f64x2 a, f64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vminq_f64(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_min_pd(a, b);
#endif
}

template <>
inline u64x2 min(u64x2 a, u64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    // No vminq_u64 at 64-bit width; blend on the comparison instead.
    return vbslq_u64(vcltq_u64(a, b), a, b);
#elif defined(SIMD_BACKEND_X86_)
    // No _mm_min_epu64 either; blend on the biased signed comparison.
    const __m128i bias = _mm_set1_epi64x(INT64_MIN);
    const __m128i lt =
        _mm_cmpgt_epi64(_mm_xor_si128(b, bias), _mm_xor_si128(a, bias));
    return _mm_blendv_epi8(b, a, lt);
#endif
}

template <>
inline f64x2 max(f64x2 a, f64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxq_f64(a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_max_pd(a, b);
#endif
}

template <>
inline u64x2 max(u64x2 a, u64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    // No vmaxq_u64 at 64-bit width; blend on the comparison instead.
    return vbslq_u64(vcgtq_u64(a, b), a, b);
#elif defined(SIMD_BACKEND_X86_)
    // No _mm_max_epu64 either; blend on the biased signed comparison.
    const __m128i bias = _mm_set1_epi64x(INT64_MIN);
    const __m128i gt =
        _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    return _mm_blendv_epi8(b, a, gt);
#endif
}

template <>
inline f64x2 abs(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vabsq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_andnot_pd(_mm_set1_pd(-0.0), v);
#endif
}

template <>
inline f64x2 sqrt(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vsqrtq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_sqrt_pd(v);
#endif
}

template <>
inline f64x2 round(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndnq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
}

template <>
inline f64x2 floor(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndmq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_floor_pd(v);
#endif
}

template <>
inline f64x2 ceil(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndpq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_ceil_pd(v);
#endif
}

template <>
inline f64x2 trunc(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vrndq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_round_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#endif
}

template <>
inline f64x2 recip(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    // The estimate is ~8 bits and each Newton-Raphson step doubles it, so
    // double's 53-bit significand needs three steps where f32 needed two.
    f64x2 e = vrecpeq_f64(v);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    return e;
#elif defined(SIMD_BACKEND_X86_)
    // x86 has no double-precision reciprocal estimate to refine, so this is
    // a plain divide -- correctly rounded, i.e. better than the NEON path.
    return _mm_div_pd(_mm_set1_pd(1.0), v);
#endif
}

template <>
inline f64x2 rsqrt(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    // Three refinement steps, for the same reason as recip above.
    f64x2 e = vrsqrteq_f64(v);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    return e;
#elif defined(SIMD_BACKEND_X86_)
    // No double-precision rsqrt estimate either; see recip above.
    return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v));
#endif
}

template <>
inline f64x2 select(f64x2::mask_type mask, f64x2 a, f64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vbslq_f64(mask, a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_blendv_pd(b, a, _mm_castsi128_pd(mask));
#endif
}

template <>
inline u64x2 select(u64x2::mask_type mask, u64x2 a, u64x2 b) {
#if defined(SIMD_BACKEND_NEON_)
    return vbslq_u64(mask, a, b);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_blendv_epi8(b, a, mask);
#endif
}

template <>
inline f64x2::scalar_type hsum(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vaddvq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)));
#endif
}

template <>
inline u64x2::scalar_type hsum(u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vaddvq_u64(v);
#elif defined(SIMD_BACKEND_X86_)
    return static_cast<u64>(
        _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_unpackhi_epi64(v, v))));
#endif
}

template <>
inline f64x2::scalar_type hmin(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vminvq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_cvtsd_f64(_mm_min_sd(v, _mm_unpackhi_pd(v, v)));
#endif
}

template <>
inline u64x2::scalar_type hmin(u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    // No vminvq_u64; only two lanes, so compare them directly.
    u64 a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
    return a < b ? a : b;
#elif defined(SIMD_BACKEND_X86_)
    // Only two lanes, so compare them directly.
    u64 a = static_cast<u64>(_mm_cvtsi128_si64(v));
    u64 b = static_cast<u64>(_mm_extract_epi64(v, 1));
    return a < b ? a : b;
#endif
}

template <>
inline f64x2::scalar_type hmax(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxvq_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_cvtsd_f64(_mm_max_sd(v, _mm_unpackhi_pd(v, v)));
#endif
}

template <>
inline u64x2::scalar_type hmax(u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    // No vmaxvq_u64; only two lanes, so compare them directly.
    u64 a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
    return a > b ? a : b;
#elif defined(SIMD_BACKEND_X86_)
    // Only two lanes, so compare them directly.
    u64 a = static_cast<u64>(_mm_cvtsi128_si64(v));
    u64 b = static_cast<u64>(_mm_extract_epi64(v, 1));
    return a > b ? a : b;
#endif
}

template <>
inline u64x2 cast<u64x2, f64x2>(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vcvtq_u64_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    // No packed f64 -> u64 before AVX-512DQ. Clamping into [0, 2^64) first
    // sends negatives and NaN to 0 the way NEON's saturating convert does and
    // keeps the per-lane scalar conversion out of undefined behaviour; the
    // top end is then saturated back to UINT64_MAX to complete the match.
    const __m128d x = _mm_max_pd(v, _mm_setzero_pd());
    const __m128d over = _mm_cmpge_pd(x, _mm_set1_pd(18446744073709551616.0));
    const __m128d y = _mm_min_pd(x, _mm_set1_pd(18446744073709549568.0));
    const u64 lo = static_cast<u64>(_mm_cvtsd_f64(y));
    const u64 hi = static_cast<u64>(_mm_cvtsd_f64(_mm_unpackhi_pd(y, y)));
    const __m128i r = _mm_set_epi64x(static_cast<long long>(hi),
                                     static_cast<long long>(lo));
    return _mm_blendv_epi8(r, _mm_set1_epi32(-1), _mm_castpd_si128(over));
#endif
}

template <>
inline f64x2 cast<f64x2, u64x2>(u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vcvtq_f64_u64(v);
#elif defined(SIMD_BACKEND_X86_)
    // No packed u64 -> f64 either; two lanes convert cheaply as scalars.
    const u64 lo = static_cast<u64>(_mm_cvtsi128_si64(v));
    const u64 hi = static_cast<u64>(_mm_extract_epi64(v, 1));
    return _mm_setr_pd(static_cast<f64>(lo), static_cast<f64>(hi));
#endif
}

template <>
inline u64x2 reinterpret<u64x2, f64x2>(f64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vreinterpretq_u64_f64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castpd_si128(v);
#endif
}

template <>
inline f64x2 reinterpret<f64x2, u64x2>(u64x2 v) {
#if defined(SIMD_BACKEND_NEON_)
    return vreinterpretq_f64_u64(v);
#elif defined(SIMD_BACKEND_X86_)
    return _mm_castsi128_pd(v);
#endif
}

template <>
inline bool any(u64x2 mask) {
#if defined(SIMD_BACKEND_NEON_)
    return vmaxvq_u32(vreinterpretq_u32_u64(mask)) != 0;
#elif defined(SIMD_BACKEND_X86_)
    return !_mm_testz_si128(mask, mask);
#endif
}

template <>
inline bool all(u64x2 mask) {
#if defined(SIMD_BACKEND_NEON_)
    return vminvq_u32(vreinterpretq_u32_u64(mask)) == 0xFFFFFFFFu;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_test_all_ones(mask) != 0;
#endif
}

template <>
inline u32 movemask(u64x2 mask) {
#if defined(SIMD_BACKEND_NEON_)
    const uint64x2_t bits = {1u, 2u};
    return static_cast<u32>(vaddvq_u64(vandq_u64(mask, bits)));
#elif defined(SIMD_BACKEND_X86_)
    return static_cast<u32>(_mm_movemask_pd(_mm_castsi128_pd(mask)));
#endif
}

template <>
inline f32x4 iota<f32x4>() {
#if defined(SIMD_BACKEND_NEON_)
    const float32x4_t r = {0.f, 1.f, 2.f, 3.f};
    return r;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_setr_ps(0.f, 1.f, 2.f, 3.f);
#endif
}

template <>
inline u32x4 iota<u32x4>() {
#if defined(SIMD_BACKEND_NEON_)
    const uint32x4_t r = {0u, 1u, 2u, 3u};
    return r;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_setr_epi32(0, 1, 2, 3);
#endif
}

template <>
inline f32x4::mask_type mask_from_bits<f32x4>(u32 bits) {
#if defined(SIMD_BACKEND_NEON_)
    const uint32x4_t bit = {1u, 2u, 4u, 8u};
    return vceqq_u32(vandq_u32(vdupq_n_u32(bits), bit), bit);
#elif defined(SIMD_BACKEND_X86_)
    const __m128i bit = _mm_setr_epi32(1, 2, 4, 8);
    return _mm_cmpeq_epi32(
        _mm_and_si128(_mm_set1_epi32(static_cast<int>(bits)), bit), bit);
#endif
}

template <>
inline u32x4::mask_type mask_from_bits<u32x4>(u32 bits) {
#if defined(SIMD_BACKEND_NEON_)
    const uint32x4_t bit = {1u, 2u, 4u, 8u};
    return vceqq_u32(vandq_u32(vdupq_n_u32(bits), bit), bit);
#elif defined(SIMD_BACKEND_X86_)
    const __m128i bit = _mm_setr_epi32(1, 2, 4, 8);
    return _mm_cmpeq_epi32(
        _mm_and_si128(_mm_set1_epi32(static_cast<int>(bits)), bit), bit);
#endif
}

template <>
inline f64x2 iota<f64x2>() {
#if defined(SIMD_BACKEND_NEON_)
    const float64x2_t r = {0.0, 1.0};
    return r;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_setr_pd(0.0, 1.0);
#endif
}

template <>
inline u64x2 iota<u64x2>() {
#if defined(SIMD_BACKEND_NEON_)
    const uint64x2_t r = {0u, 1u};
    return r;
#elif defined(SIMD_BACKEND_X86_)
    return _mm_set_epi64x(1, 0);
#endif
}

template <>
inline f64x2::mask_type mask_from_bits<f64x2>(u32 bits) {
#if defined(SIMD_BACKEND_NEON_)
    const uint64x2_t bit = {1u, 2u};
    return vceqq_u64(vandq_u64(vdupq_n_u64(bits), bit), bit);
#elif defined(SIMD_BACKEND_X86_)
    const __m128i bit = _mm_set_epi64x(2, 1);
    return _mm_cmpeq_epi64(
        _mm_and_si128(_mm_set1_epi64x(static_cast<long long>(bits)), bit), bit);
#endif
}

template <>
inline u64x2::mask_type mask_from_bits<u64x2>(u32 bits) {
#if defined(SIMD_BACKEND_NEON_)
    const uint64x2_t bit = {1u, 2u};
    return vceqq_u64(vandq_u64(vdupq_n_u64(bits), bit), bit);
#elif defined(SIMD_BACKEND_X86_)
    const __m128i bit = _mm_set_epi64x(2, 1);
    return _mm_cmpeq_epi64(
        _mm_and_si128(_mm_set1_epi64x(static_cast<long long>(bits)), bit), bit);
#endif
}

// The composed operations, emitted once per type. Every intrinsic they rest on
// is defined above, for whichever of the two tier-1 backends is in play.
SIMD_DEFINE_COMPOSED_(f32x4)
SIMD_DEFINE_COMPOSED_(u32x4)
SIMD_DEFINE_COMPOSED_(f64x2)
SIMD_DEFINE_COMPOSED_(u64x2)

SIMD_DEFINE_SELECT_OPS_(f32x4)
SIMD_DEFINE_SELECT_OPS_(f64x2)

SIMD_DEFINE_MASK_OPS_(u32x4)
SIMD_DEFINE_MASK_OPS_(u64x2)

#endif  // SIMD_MAX_WIDTH_ >= 1

#if SIMD_MAX_WIDTH_ >= 2

// 256-bit backend
//
// Tier 2 is AVX2 today, and the bodies below are written against it directly
// rather than through a SIMD_BACKEND_* switch the way tier 1 is: there is only
// one 256-bit backend so far. A second one (SVE, say) would add its own arms
// here exactly as NEON and SSE share tier 1.
//
// Tier 2 implies tier 1, so the 128-bit types above are also available, and a
// few of the operations x86 has no 256-bit instruction for simply split into
// two 128-bit halves and reuse them.

using f32x8 = vec<f32, 8>;
using u32x8 = vec<u32, 8>;

using f64x4 = vec<f64, 4>;
using u64x4 = vec<u64, 4>;

template <>
struct vec<f32, 8> {
    using scalar_type = f32;
    using mask_type = u32x8;
    static constexpr width_type width = 8;

    using simd_type = __m256;

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.f) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const { return _mm256_xor_ps(v, _mm256_set1_ps(-0.f)); }

    This operator+(This o) const { return _mm256_add_ps(*this, o); }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const { return _mm256_sub_ps(*this, o); }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const { return _mm256_mul_ps(*this, o); }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const { return _mm256_div_ps(*this, o); }
    This operator/(scalar_type s) const;
    friend This operator/(scalar_type s, This o);

    // _CMP_*_OQ are the ordered forms (false for NaN); _CMP_NEQ_UQ is the
    // unordered not-equal, so NaN != anything stays true. That matches what
    // the tier-1 backends produce.
    mask_type operator==(This o) const;
    mask_type operator!=(This o) const;
    mask_type operator<(This o) const;
    mask_type operator<=(This o) const;
    mask_type operator>(This o) const;
    mask_type operator>=(This o) const;

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o);
    friend mask_type operator!=(scalar_type s, This o);
    friend mask_type operator<(scalar_type s, This o);
    friend mask_type operator<=(scalar_type s, This o);
    friend mask_type operator>(scalar_type s, This o);
    friend mask_type operator>=(scalar_type s, This o);
};

static_assert(VecType<f32x8>, "vec<f32, 8> does not satisfy VecType");

template <>
struct vec<u32, 8> {
    using scalar_type = u32;
    using mask_type = vec<u32, 8>;
    static constexpr width_type width = 8;

    using simd_type = __m256i;

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const { return _mm256_add_epi32(*this, o); }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const { return _mm256_sub_epi32(*this, o); }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const { return _mm256_mullo_epi32(*this, o); }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const { return _mm256_cmpeq_epi32(*this, o); }
    mask_type operator!=(This o) const {
        return _mm256_xor_si256(_mm256_cmpeq_epi32(*this, o), _mm256_set1_epi32(-1));
    }
    // As at 128 bits, the unsigned compares come from the unsigned min/max
    // rather than a sign-bias xor: x86 has no unsigned integer compare.
    mask_type operator<(This o) const {
        return _mm256_xor_si256(
            _mm256_cmpeq_epi32(_mm256_max_epu32(*this, o), *this),
            _mm256_set1_epi32(-1));
    }
    mask_type operator<=(This o) const {
        return _mm256_cmpeq_epi32(_mm256_min_epu32(*this, o), *this);
    }
    mask_type operator>(This o) const {
        return _mm256_xor_si256(
            _mm256_cmpeq_epi32(_mm256_min_epu32(*this, o), *this),
            _mm256_set1_epi32(-1));
    }
    mask_type operator>=(This o) const {
        return _mm256_cmpeq_epi32(_mm256_max_epu32(*this, o), *this);
    }

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o) { return o == s; }
    friend mask_type operator!=(scalar_type s, This o) { return o != s; }
    friend mask_type operator<(scalar_type s, This o) { return o > s; }
    friend mask_type operator<=(scalar_type s, This o) { return o >= s; }
    friend mask_type operator>(scalar_type s, This o) { return o < s; }
    friend mask_type operator>=(scalar_type s, This o) { return o <= s; }

    This operator&(This o) const { return _mm256_and_si256(*this, o); }
    This operator|(This o) const { return _mm256_or_si256(*this, o); }
    This operator^(This o) const { return _mm256_xor_si256(*this, o); }
    This operator~() const { return _mm256_xor_si256(v, _mm256_set1_epi32(-1)); }

    This operator<<(int n) const {
        return _mm256_sll_epi32(*this, _mm_cvtsi32_si128(n));
    }
    This operator>>(int n) const {
        return _mm256_srl_epi32(*this, _mm_cvtsi32_si128(n));
    }
};

static_assert(VecType<u32x8>, "vec<u32, 8> does not satisfy VecType");

inline f32x8::mask_type f32x8::operator==(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_EQ_OQ));
}
inline f32x8::mask_type f32x8::operator!=(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_NEQ_UQ));
}
inline f32x8::mask_type f32x8::operator<(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_LT_OQ));
}
inline f32x8::mask_type f32x8::operator<=(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_LE_OQ));
}
inline f32x8::mask_type f32x8::operator>(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_GT_OQ));
}
inline f32x8::mask_type f32x8::operator>=(This o) const {
    return _mm256_castps_si256(_mm256_cmp_ps(*this, o, _CMP_GE_OQ));
}

template <>
inline f32x8 splat<f32x8>(f32x8::scalar_type s) { return _mm256_set1_ps(s); }

inline f32x8::vec(scalar_type s) : v(splat<This>(s)) {}

inline f32x8 f32x8::operator+(scalar_type o) const { return *this + splat<This>(o); }
inline f32x8 f32x8::operator-(scalar_type o) const { return *this - splat<This>(o); }
inline f32x8 operator-(f32x8::scalar_type s, f32x8 o) { return splat<f32x8>(s) - o; }
inline f32x8 f32x8::operator*(scalar_type o) const { return *this * splat<This>(o); }
inline f32x8 f32x8::operator/(scalar_type o) const { return *this / splat<This>(o); }
inline f32x8 operator/(f32x8::scalar_type s, f32x8 o) { return splat<f32x8>(s) / o; }

inline f32x8::mask_type f32x8::operator==(scalar_type o) const { return *this == splat<This>(o); }
inline f32x8::mask_type f32x8::operator!=(scalar_type o) const { return *this != splat<This>(o); }
inline f32x8::mask_type f32x8::operator<(scalar_type o) const { return *this < splat<This>(o); }
inline f32x8::mask_type f32x8::operator<=(scalar_type o) const { return *this <= splat<This>(o); }
inline f32x8::mask_type f32x8::operator>(scalar_type o) const { return *this > splat<This>(o); }
inline f32x8::mask_type f32x8::operator>=(scalar_type o) const { return *this >= splat<This>(o); }

inline f32x8::mask_type operator==(f32x8::scalar_type s, f32x8 o) { return o == s; }
inline f32x8::mask_type operator!=(f32x8::scalar_type s, f32x8 o) { return o != s; }
inline f32x8::mask_type operator<(f32x8::scalar_type s, f32x8 o) { return o > s; }
inline f32x8::mask_type operator<=(f32x8::scalar_type s, f32x8 o) { return o >= s; }
inline f32x8::mask_type operator>(f32x8::scalar_type s, f32x8 o) { return o < s; }
inline f32x8::mask_type operator>=(f32x8::scalar_type s, f32x8 o) { return o <= s; }

template <>
inline u32x8 splat<u32x8>(u32x8::scalar_type s) {
    return _mm256_set1_epi32(static_cast<int>(s));
}

inline u32x8::vec(scalar_type s) : v(splat<This>(s)) {}

inline u32x8 u32x8::operator+(scalar_type o) const { return *this + splat<This>(o); }
inline u32x8 u32x8::operator-(scalar_type o) const { return *this - splat<This>(o); }
inline u32x8 operator-(u32x8::scalar_type s, u32x8 o) { return splat<u32x8>(s) - o; }
inline u32x8 u32x8::operator*(scalar_type o) const { return *this * splat<This>(o); }

inline u32x8::mask_type u32x8::operator==(scalar_type o) const { return *this == splat<This>(o); }
inline u32x8::mask_type u32x8::operator!=(scalar_type o) const { return *this != splat<This>(o); }
inline u32x8::mask_type u32x8::operator<(scalar_type o) const { return *this < splat<This>(o); }
inline u32x8::mask_type u32x8::operator<=(scalar_type o) const { return *this <= splat<This>(o); }
inline u32x8::mask_type u32x8::operator>(scalar_type o) const { return *this > splat<This>(o); }
inline u32x8::mask_type u32x8::operator>=(scalar_type o) const { return *this >= splat<This>(o); }

// a * b + c
template <>
inline f32x8 fma(f32x8 a, f32x8 b, f32x8 c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f32x8 fms(f32x8 a, f32x8 b, f32x8 c) {
#if defined(__FMA__)
    return _mm256_fmsub_ps(a, b, c);
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f32x8 fms2(f32x8 a, f32x8 b, f32x8 c) {
#if defined(__FMA__)
    return _mm256_fnmadd_ps(a, b, c);
#else
    return c - a * b;
#endif
}

template <>
inline f32x8 load(const f32x8::scalar_type *ptr) { return _mm256_loadu_ps(ptr); }

template <>
inline void store(f32x8::scalar_type *ptr, f32x8 v) { _mm256_storeu_ps(ptr, v); }

// A 256-bit lane op works on the 128-bit half holding the lane, then puts the
// half back; there is no cross-half insert or extract to do it in one step.
template <width_type lane>
inline f32x8 load_lane(f32x8 v, const f32x8::scalar_type *ptr) {
    static_assert(lane < 8);
    __m128 h = _mm256_extractf128_ps(v, lane / 4);
    h = _mm_insert_ps(h, _mm_load_ss(ptr), (lane % 4) << 4);
    return _mm256_insertf128_ps(v, h, lane / 4);
}

template <width_type lane>
inline void store_lane(f32x8::scalar_type *ptr, f32x8 v) {
    static_assert(lane < 8);
    const __m128 h = _mm256_extractf128_ps(v, lane / 4);
    _mm_store_ss(ptr, _mm_shuffle_ps(h, h, _MM_SHUFFLE(lane % 4, lane % 4, lane % 4, lane % 4)));
}

template <width_type lane>
inline f32x8::scalar_type get_lane(f32x8 v) {
    static_assert(lane < 8);
    const __m128 h = _mm256_extractf128_ps(v, lane / 4);
    return _mm_cvtss_f32(_mm_shuffle_ps(h, h, _MM_SHUFFLE(lane % 4, lane % 4, lane % 4, lane % 4)));
}

template <width_type lane>
inline f32x8 set_lane(f32x8 v, f32x8::scalar_type s) {
    static_assert(lane < 8);
    __m128 h = _mm256_extractf128_ps(v, lane / 4);
    h = _mm_insert_ps(h, _mm_set_ss(s), (lane % 4) << 4);
    return _mm256_insertf128_ps(v, h, lane / 4);
}

template <>
inline u32x8 load(const u32x8::scalar_type *ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
}

template <>
inline void store(u32x8::scalar_type *ptr, u32x8 v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), v);
}

template <width_type lane>
inline u32x8 load_lane(u32x8 v, const u32x8::scalar_type *ptr) {
    static_assert(lane < 8);
    __m128i h = _mm256_extracti128_si256(v, lane / 4);
    h = _mm_insert_epi32(h, static_cast<int>(*ptr), lane % 4);
    return _mm256_inserti128_si256(v, h, lane / 4);
}

template <width_type lane>
inline void store_lane(u32x8::scalar_type *ptr, u32x8 v) {
    static_assert(lane < 8);
    *ptr = static_cast<u32>(_mm_extract_epi32(_mm256_extracti128_si256(v, lane / 4), lane % 4));
}

template <width_type lane>
inline u32x8::scalar_type get_lane(u32x8 v) {
    static_assert(lane < 8);
    return static_cast<u32>(_mm_extract_epi32(_mm256_extracti128_si256(v, lane / 4), lane % 4));
}

template <width_type lane>
inline u32x8 set_lane(u32x8 v, u32x8::scalar_type s) {
    static_assert(lane < 8);
    __m128i h = _mm256_extracti128_si256(v, lane / 4);
    h = _mm_insert_epi32(h, static_cast<int>(s), lane % 4);
    return _mm256_inserti128_si256(v, h, lane / 4);
}

template <>
inline f32x8 min(f32x8 a, f32x8 b) { return _mm256_min_ps(a, b); }
template <>
inline u32x8 min(u32x8 a, u32x8 b) { return _mm256_min_epu32(a, b); }
template <>
inline f32x8 max(f32x8 a, f32x8 b) { return _mm256_max_ps(a, b); }
template <>
inline u32x8 max(u32x8 a, u32x8 b) { return _mm256_max_epu32(a, b); }

template <>
inline f32x8 abs(f32x8 v) { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), v); }
template <>
inline f32x8 sqrt(f32x8 v) { return _mm256_sqrt_ps(v); }

// A divide rather than _mm256_rcp_ps plus refinement, for the reason spelled
// out on the 128-bit path: the estimate flushes denormal inputs to zero.
template <>
inline f32x8 recip(f32x8 v) { return _mm256_div_ps(_mm256_set1_ps(1.f), v); }
template <>
inline f32x8 rsqrt(f32x8 v) {
    return _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_sqrt_ps(v));
}

template <>
inline f32x8 round(f32x8 v) {
    return _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
template <>
inline f32x8 floor(f32x8 v) { return _mm256_floor_ps(v); }
template <>
inline f32x8 ceil(f32x8 v) { return _mm256_ceil_ps(v); }
template <>
inline f32x8 trunc(f32x8 v) {
    return _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

template <>
inline f32x8 select(f32x8::mask_type mask, f32x8 a, f32x8 b) {
    return _mm256_blendv_ps(b, a, _mm256_castsi256_ps(mask));
}
template <>
inline u32x8 select(u32x8::mask_type mask, u32x8 a, u32x8 b) {
    return _mm256_blendv_epi8(b, a, mask);
}

// Reductions fold the two 128-bit halves together first, then finish with the
// same shuffle-and-fold the tier-1 path uses.
template <>
inline f32x8::scalar_type hsum(f32x8 v) {
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
}
template <>
inline f32x8::scalar_type hmin(f32x8 v) {
    __m128 t = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    t = _mm_min_ps(t, _mm_movehl_ps(t, t));
    t = _mm_min_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
}
template <>
inline f32x8::scalar_type hmax(f32x8 v) {
    __m128 t = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    t = _mm_max_ps(t, _mm_movehl_ps(t, t));
    t = _mm_max_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(t);
}
template <>
inline u32x8::scalar_type hsum(u32x8 v) {
    __m128i t = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_add_epi32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
}
template <>
inline u32x8::scalar_type hmin(u32x8 v) {
    __m128i t = _mm_min_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    t = _mm_min_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_min_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
}
template <>
inline u32x8::scalar_type hmax(u32x8 v) {
    __m128i t = _mm_max_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    t = _mm_max_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2)));
    t = _mm_max_epu32(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<u32>(_mm_cvtsi128_si32(t));
}

// Saturating, exactly as the 128-bit path: see the comment there.
template <>
inline u32x8 cast<u32x8, f32x8>(f32x8 v) {
    const __m256 bias = _mm256_set1_ps(2147483648.f);
    const __m256 x = _mm256_max_ps(v, _mm256_setzero_ps());
    const __m256 high = _mm256_cmp_ps(x, bias, _CMP_GE_OQ);
    const __m256 over = _mm256_cmp_ps(x, _mm256_set1_ps(4294967296.f), _CMP_GE_OQ);
    const __m256i lo = _mm256_cvttps_epi32(x);
    const __m256i hi = _mm256_add_epi32(_mm256_cvttps_epi32(_mm256_sub_ps(x, bias)),
                                        _mm256_set1_epi32(INT32_MIN));
    const __m256 r = _mm256_blendv_ps(_mm256_castsi256_ps(lo), _mm256_castsi256_ps(hi), high);
    return _mm256_castps_si256(
        _mm256_blendv_ps(r, _mm256_castsi256_ps(_mm256_set1_epi32(-1)), over));
}

template <>
inline f32x8 cast<f32x8, u32x8>(u32x8 v) {
    const __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(v, _mm256_set1_epi32(0xFFFF)));
    const __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(v, 16));
    return _mm256_add_ps(lo, _mm256_mul_ps(hi, _mm256_set1_ps(65536.f)));
}

template <>
inline u32x8 reinterpret<u32x8, f32x8>(f32x8 v) { return _mm256_castps_si256(v); }
template <>
inline f32x8 reinterpret<f32x8, u32x8>(u32x8 v) { return _mm256_castsi256_ps(v); }

template <>
inline bool any(u32x8 mask) { return !_mm256_testz_si256(mask, mask); }
template <>
inline bool all(u32x8 mask) {
    return _mm256_testc_si256(mask, _mm256_set1_epi32(-1)) != 0;
}
template <>
inline u32 movemask(u32x8 mask) {
    return static_cast<u32>(_mm256_movemask_ps(_mm256_castsi256_ps(mask)));
}

template <>
inline f32x8 iota<f32x8>() {
    return _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f);
}
template <>
inline u32x8 iota<u32x8>() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }

template <>
inline f32x8::mask_type mask_from_bits<f32x8>(u32 bits) {
    const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    return _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32(static_cast<int>(bits)), bit), bit);
}
template <>
inline u32x8::mask_type mask_from_bits<u32x8>(u32 bits) {
    return mask_from_bits<f32x8>(bits);
}

template <>
struct vec<f64, 4> {
    using scalar_type = f64;
    using mask_type = u64x4;
    static constexpr width_type width = 4;

    using simd_type = __m256d;

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const { return _mm256_xor_pd(v, _mm256_set1_pd(-0.0)); }

    This operator+(This o) const { return _mm256_add_pd(*this, o); }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const { return _mm256_sub_pd(*this, o); }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const { return _mm256_mul_pd(*this, o); }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const { return _mm256_div_pd(*this, o); }
    This operator/(scalar_type s) const;
    friend This operator/(scalar_type s, This o);

    mask_type operator==(This o) const;
    mask_type operator!=(This o) const;
    mask_type operator<(This o) const;
    mask_type operator<=(This o) const;
    mask_type operator>(This o) const;
    mask_type operator>=(This o) const;

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o);
    friend mask_type operator!=(scalar_type s, This o);
    friend mask_type operator<(scalar_type s, This o);
    friend mask_type operator<=(scalar_type s, This o);
    friend mask_type operator>(scalar_type s, This o);
    friend mask_type operator>=(scalar_type s, This o);
};

static_assert(VecType<f64x4>, "vec<f64, 4> does not satisfy VecType");

template <>
struct vec<u64, 4> {
    using scalar_type = u64;
    using mask_type = vec<u64, 4>;
    static constexpr width_type width = 4;

    using simd_type = __m256i;

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const { return _mm256_add_epi64(*this, o); }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const { return _mm256_sub_epi64(*this, o); }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
        // Still no packed 64-bit multiply at 256 bits; reuse the 128-bit path
        // on each half rather than repeat its lane-wise scalar expansion.
        const u64x2 lo = u64x2(_mm256_castsi256_si128(*this)) *
                         u64x2(_mm256_castsi256_si128(o));
        const u64x2 hi = u64x2(_mm256_extracti128_si256(*this, 1)) *
                         u64x2(_mm256_extracti128_si256(o, 1));
        return _mm256_set_m128i(hi, lo);
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const { return _mm256_cmpeq_epi64(*this, o); }
    mask_type operator!=(This o) const {
        return _mm256_xor_si256(_mm256_cmpeq_epi64(*this, o), _mm256_set1_epi32(-1));
    }
    // No unsigned 64-bit compare and no unsigned 64-bit min/max to derive one
    // from, so these flip the sign bit and use the signed form.
    mask_type operator<(This o) const {
        const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
        return _mm256_cmpgt_epi64(_mm256_xor_si256(o, bias),
                                  _mm256_xor_si256(*this, bias));
    }
    mask_type operator<=(This o) const {
        const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
        return _mm256_xor_si256(
            _mm256_cmpgt_epi64(_mm256_xor_si256(*this, bias), _mm256_xor_si256(o, bias)),
            _mm256_set1_epi32(-1));
    }
    mask_type operator>(This o) const {
        const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
        return _mm256_cmpgt_epi64(_mm256_xor_si256(*this, bias),
                                  _mm256_xor_si256(o, bias));
    }
    mask_type operator>=(This o) const {
        const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
        return _mm256_xor_si256(
            _mm256_cmpgt_epi64(_mm256_xor_si256(o, bias), _mm256_xor_si256(*this, bias)),
            _mm256_set1_epi32(-1));
    }

    mask_type operator==(scalar_type s) const;
    mask_type operator!=(scalar_type s) const;
    mask_type operator<(scalar_type s) const;
    mask_type operator<=(scalar_type s) const;
    mask_type operator>(scalar_type s) const;
    mask_type operator>=(scalar_type s) const;

    friend mask_type operator==(scalar_type s, This o) { return o == s; }
    friend mask_type operator!=(scalar_type s, This o) { return o != s; }
    friend mask_type operator<(scalar_type s, This o) { return o > s; }
    friend mask_type operator<=(scalar_type s, This o) { return o >= s; }
    friend mask_type operator>(scalar_type s, This o) { return o < s; }
    friend mask_type operator>=(scalar_type s, This o) { return o <= s; }

    This operator&(This o) const { return _mm256_and_si256(*this, o); }
    This operator|(This o) const { return _mm256_or_si256(*this, o); }
    This operator^(This o) const { return _mm256_xor_si256(*this, o); }
    This operator~() const { return _mm256_xor_si256(v, _mm256_set1_epi32(-1)); }

    This operator<<(int n) const {
        return _mm256_sll_epi64(*this, _mm_cvtsi32_si128(n));
    }
    This operator>>(int n) const {
        return _mm256_srl_epi64(*this, _mm_cvtsi32_si128(n));
    }
};

static_assert(VecType<u64x4>, "vec<u64, 4> does not satisfy VecType");

inline f64x4::mask_type f64x4::operator==(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_EQ_OQ));
}
inline f64x4::mask_type f64x4::operator!=(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_NEQ_UQ));
}
inline f64x4::mask_type f64x4::operator<(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_LT_OQ));
}
inline f64x4::mask_type f64x4::operator<=(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_LE_OQ));
}
inline f64x4::mask_type f64x4::operator>(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_GT_OQ));
}
inline f64x4::mask_type f64x4::operator>=(This o) const {
    return _mm256_castpd_si256(_mm256_cmp_pd(*this, o, _CMP_GE_OQ));
}

template <>
inline f64x4 splat<f64x4>(f64x4::scalar_type s) { return _mm256_set1_pd(s); }

inline f64x4::vec(scalar_type s) : v(splat<This>(s)) {}

inline f64x4 f64x4::operator+(scalar_type o) const { return *this + splat<This>(o); }
inline f64x4 f64x4::operator-(scalar_type o) const { return *this - splat<This>(o); }
inline f64x4 operator-(f64x4::scalar_type s, f64x4 o) { return splat<f64x4>(s) - o; }
inline f64x4 f64x4::operator*(scalar_type o) const { return *this * splat<This>(o); }
inline f64x4 f64x4::operator/(scalar_type o) const { return *this / splat<This>(o); }
inline f64x4 operator/(f64x4::scalar_type s, f64x4 o) { return splat<f64x4>(s) / o; }

inline f64x4::mask_type f64x4::operator==(scalar_type o) const { return *this == splat<This>(o); }
inline f64x4::mask_type f64x4::operator!=(scalar_type o) const { return *this != splat<This>(o); }
inline f64x4::mask_type f64x4::operator<(scalar_type o) const { return *this < splat<This>(o); }
inline f64x4::mask_type f64x4::operator<=(scalar_type o) const { return *this <= splat<This>(o); }
inline f64x4::mask_type f64x4::operator>(scalar_type o) const { return *this > splat<This>(o); }
inline f64x4::mask_type f64x4::operator>=(scalar_type o) const { return *this >= splat<This>(o); }

inline f64x4::mask_type operator==(f64x4::scalar_type s, f64x4 o) { return o == s; }
inline f64x4::mask_type operator!=(f64x4::scalar_type s, f64x4 o) { return o != s; }
inline f64x4::mask_type operator<(f64x4::scalar_type s, f64x4 o) { return o > s; }
inline f64x4::mask_type operator<=(f64x4::scalar_type s, f64x4 o) { return o >= s; }
inline f64x4::mask_type operator>(f64x4::scalar_type s, f64x4 o) { return o < s; }
inline f64x4::mask_type operator>=(f64x4::scalar_type s, f64x4 o) { return o <= s; }

template <>
inline u64x4 splat<u64x4>(u64x4::scalar_type s) {
    return _mm256_set1_epi64x(static_cast<long long>(s));
}

inline u64x4::vec(scalar_type s) : v(splat<This>(s)) {}

inline u64x4 u64x4::operator+(scalar_type o) const { return *this + splat<This>(o); }
inline u64x4 u64x4::operator-(scalar_type o) const { return *this - splat<This>(o); }
inline u64x4 operator-(u64x4::scalar_type s, u64x4 o) { return splat<u64x4>(s) - o; }
inline u64x4 u64x4::operator*(scalar_type o) const { return *this * splat<This>(o); }

inline u64x4::mask_type u64x4::operator==(scalar_type o) const { return *this == splat<This>(o); }
inline u64x4::mask_type u64x4::operator!=(scalar_type o) const { return *this != splat<This>(o); }
inline u64x4::mask_type u64x4::operator<(scalar_type o) const { return *this < splat<This>(o); }
inline u64x4::mask_type u64x4::operator<=(scalar_type o) const { return *this <= splat<This>(o); }
inline u64x4::mask_type u64x4::operator>(scalar_type o) const { return *this > splat<This>(o); }
inline u64x4::mask_type u64x4::operator>=(scalar_type o) const { return *this >= splat<This>(o); }

// a * b + c
template <>
inline f64x4 fma(f64x4 a, f64x4 b, f64x4 c) {
#if defined(__FMA__)
    return _mm256_fmadd_pd(a, b, c);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f64x4 fms(f64x4 a, f64x4 b, f64x4 c) {
#if defined(__FMA__)
    return _mm256_fmsub_pd(a, b, c);
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f64x4 fms2(f64x4 a, f64x4 b, f64x4 c) {
#if defined(__FMA__)
    return _mm256_fnmadd_pd(a, b, c);
#else
    return c - a * b;
#endif
}

template <>
inline f64x4 load(const f64x4::scalar_type *ptr) { return _mm256_loadu_pd(ptr); }
template <>
inline void store(f64x4::scalar_type *ptr, f64x4 v) { _mm256_storeu_pd(ptr, v); }

template <width_type lane>
inline f64x4 load_lane(f64x4 v, const f64x4::scalar_type *ptr) {
    static_assert(lane < 4);
    __m128d h = _mm256_extractf128_pd(v, lane / 2);
    if constexpr (lane % 2 == 0) h = _mm_loadl_pd(h, ptr);
    else h = _mm_loadh_pd(h, ptr);
    return _mm256_insertf128_pd(v, h, lane / 2);
}

template <width_type lane>
inline void store_lane(f64x4::scalar_type *ptr, f64x4 v) {
    static_assert(lane < 4);
    const __m128d h = _mm256_extractf128_pd(v, lane / 2);
    if constexpr (lane % 2 == 0) _mm_storel_pd(ptr, h);
    else _mm_storeh_pd(ptr, h);
}

template <width_type lane>
inline f64x4::scalar_type get_lane(f64x4 v) {
    static_assert(lane < 4);
    const __m128d h = _mm256_extractf128_pd(v, lane / 2);
    if constexpr (lane % 2 == 0) return _mm_cvtsd_f64(h);
    else return _mm_cvtsd_f64(_mm_unpackhi_pd(h, h));
}

template <width_type lane>
inline f64x4 set_lane(f64x4 v, f64x4::scalar_type s) {
    static_assert(lane < 4);
    __m128d h = _mm256_extractf128_pd(v, lane / 2);
    if constexpr (lane % 2 == 0) h = _mm_move_sd(h, _mm_set_sd(s));
    else h = _mm_shuffle_pd(h, _mm_set_sd(s), 0);
    return _mm256_insertf128_pd(v, h, lane / 2);
}

template <>
inline u64x4 load(const u64x4::scalar_type *ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
}
template <>
inline void store(u64x4::scalar_type *ptr, u64x4 v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), v);
}

template <width_type lane>
inline u64x4 load_lane(u64x4 v, const u64x4::scalar_type *ptr) {
    static_assert(lane < 4);
    __m128i h = _mm256_extracti128_si256(v, lane / 2);
    h = _mm_insert_epi64(h, static_cast<long long>(*ptr), lane % 2);
    return _mm256_inserti128_si256(v, h, lane / 2);
}

template <width_type lane>
inline void store_lane(u64x4::scalar_type *ptr, u64x4 v) {
    static_assert(lane < 4);
    *ptr = static_cast<u64>(_mm_extract_epi64(_mm256_extracti128_si256(v, lane / 2), lane % 2));
}

template <width_type lane>
inline u64x4::scalar_type get_lane(u64x4 v) {
    static_assert(lane < 4);
    return static_cast<u64>(_mm_extract_epi64(_mm256_extracti128_si256(v, lane / 2), lane % 2));
}

template <width_type lane>
inline u64x4 set_lane(u64x4 v, u64x4::scalar_type s) {
    static_assert(lane < 4);
    __m128i h = _mm256_extracti128_si256(v, lane / 2);
    h = _mm_insert_epi64(h, static_cast<long long>(s), lane % 2);
    return _mm256_inserti128_si256(v, h, lane / 2);
}

template <>
inline f64x4 min(f64x4 a, f64x4 b) { return _mm256_min_pd(a, b); }
template <>
inline f64x4 max(f64x4 a, f64x4 b) { return _mm256_max_pd(a, b); }

template <>
inline u64x4 min(u64x4 a, u64x4 b) {
    const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
    const __m256i lt = _mm256_cmpgt_epi64(_mm256_xor_si256(b, bias),
                                          _mm256_xor_si256(a, bias));
    return _mm256_blendv_epi8(b, a, lt);
}
template <>
inline u64x4 max(u64x4 a, u64x4 b) {
    const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
    const __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias),
                                          _mm256_xor_si256(b, bias));
    return _mm256_blendv_epi8(b, a, gt);
}

template <>
inline f64x4 abs(f64x4 v) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v); }
template <>
inline f64x4 sqrt(f64x4 v) { return _mm256_sqrt_pd(v); }
template <>
inline f64x4 recip(f64x4 v) { return _mm256_div_pd(_mm256_set1_pd(1.0), v); }
template <>
inline f64x4 rsqrt(f64x4 v) {
    return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(v));
}
template <>
inline f64x4 round(f64x4 v) {
    return _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
template <>
inline f64x4 floor(f64x4 v) { return _mm256_floor_pd(v); }
template <>
inline f64x4 ceil(f64x4 v) { return _mm256_ceil_pd(v); }
template <>
inline f64x4 trunc(f64x4 v) {
    return _mm256_round_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

template <>
inline f64x4 select(f64x4::mask_type mask, f64x4 a, f64x4 b) {
    return _mm256_blendv_pd(b, a, _mm256_castsi256_pd(mask));
}
template <>
inline u64x4 select(u64x4::mask_type mask, u64x4 a, u64x4 b) {
    return _mm256_blendv_epi8(b, a, mask);
}

template <>
inline f64x4::scalar_type hsum(f64x4 v) {
    const __m128d t = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_add_sd(t, _mm_unpackhi_pd(t, t)));
}
template <>
inline f64x4::scalar_type hmin(f64x4 v) {
    const __m128d t = _mm_min_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_min_sd(t, _mm_unpackhi_pd(t, t)));
}
template <>
inline f64x4::scalar_type hmax(f64x4 v) {
    const __m128d t = _mm_max_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_max_sd(t, _mm_unpackhi_pd(t, t)));
}

// Fold the halves together, then hand the rest to the 128-bit reduction.
template <>
inline u64x4::scalar_type hsum(u64x4 v) {
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(v),
                                    _mm256_extracti128_si256(v, 1));
    return hsum<u64x2>(t);
}
template <>
inline u64x4::scalar_type hmin(u64x4 v) {
    return hmin<u64x2>(min<u64x2>(_mm256_castsi256_si128(v),
                                  _mm256_extracti128_si256(v, 1)));
}
template <>
inline u64x4::scalar_type hmax(u64x4 v) {
    return hmax<u64x2>(max<u64x2>(_mm256_castsi256_si128(v),
                                  _mm256_extracti128_si256(v, 1)));
}

// No packed f64 <-> u64 convert at 256 bits either; reuse the 128-bit halves.
template <>
inline u64x4 cast<u64x4, f64x4>(f64x4 v) {
    const u64x2 lo = cast<u64x2, f64x2>(_mm256_castpd256_pd128(v));
    const u64x2 hi = cast<u64x2, f64x2>(_mm256_extractf128_pd(v, 1));
    return _mm256_set_m128i(hi, lo);
}
template <>
inline f64x4 cast<f64x4, u64x4>(u64x4 v) {
    const f64x2 lo = cast<f64x2, u64x2>(_mm256_castsi256_si128(v));
    const f64x2 hi = cast<f64x2, u64x2>(_mm256_extracti128_si256(v, 1));
    return _mm256_set_m128d(hi, lo);
}

template <>
inline u64x4 reinterpret<u64x4, f64x4>(f64x4 v) { return _mm256_castpd_si256(v); }
template <>
inline f64x4 reinterpret<f64x4, u64x4>(u64x4 v) { return _mm256_castsi256_pd(v); }

template <>
inline bool any(u64x4 mask) { return !_mm256_testz_si256(mask, mask); }
template <>
inline bool all(u64x4 mask) {
    return _mm256_testc_si256(mask, _mm256_set1_epi32(-1)) != 0;
}
template <>
inline u32 movemask(u64x4 mask) {
    return static_cast<u32>(_mm256_movemask_pd(_mm256_castsi256_pd(mask)));
}

template <>
inline f64x4 iota<f64x4>() { return _mm256_setr_pd(0.0, 1.0, 2.0, 3.0); }
template <>
inline u64x4 iota<u64x4>() { return _mm256_setr_epi64x(0, 1, 2, 3); }

template <>
inline f64x4::mask_type mask_from_bits<f64x4>(u32 bits) {
    const __m256i bit = _mm256_setr_epi64x(1, 2, 4, 8);
    return _mm256_cmpeq_epi64(
        _mm256_and_si256(_mm256_set1_epi64x(static_cast<long long>(bits)), bit), bit);
}
template <>
inline u64x4::mask_type mask_from_bits<u64x4>(u32 bits) {
    return mask_from_bits<f64x4>(bits);
}

// The composed operations, once per 256-bit type.
SIMD_DEFINE_COMPOSED_(f32x8)
SIMD_DEFINE_COMPOSED_(u32x8)
SIMD_DEFINE_COMPOSED_(f64x4)
SIMD_DEFINE_COMPOSED_(u64x4)

SIMD_DEFINE_SELECT_OPS_(f32x8)
SIMD_DEFINE_SELECT_OPS_(f64x4)

SIMD_DEFINE_MASK_OPS_(u32x8)
SIMD_DEFINE_MASK_OPS_(u64x4)


// Masked memory access, in hardware
//
// AVX2's maskload/maskstore suppress faults on the lanes the mask excludes, so
// they give the safe contract in one instruction. maskload zeroes the excluded
// lanes rather than leaving them alone, hence the blend against src.
//
// Both the 128- and 256-bit types are covered here, in the tier-2 block,
// because that is where AVX2 is known to be available: a tier-1-only build
// (SSE4.2 without AVX) has no maskload at all and uses the lane-wise path.

template <>
inline f32x8 load_masked(f32x8::mask_type mask, f32x8 src, const f32 *ptr) {
    return _mm256_blendv_ps(src, _mm256_maskload_ps(ptr, mask),
                            _mm256_castsi256_ps(mask));
}
template <>
inline void store_masked(f32 *ptr, f32x8::mask_type mask, f32x8 v) {
    _mm256_maskstore_ps(ptr, mask, v);
}

template <>
inline u32x8 load_masked(u32x8::mask_type mask, u32x8 src, const u32 *ptr) {
    return _mm256_blendv_epi8(
        src, _mm256_maskload_epi32(reinterpret_cast<const int *>(ptr), mask), mask);
}
template <>
inline void store_masked(u32 *ptr, u32x8::mask_type mask, u32x8 v) {
    _mm256_maskstore_epi32(reinterpret_cast<int *>(ptr), mask, v);
}

template <>
inline f64x4 load_masked(f64x4::mask_type mask, f64x4 src, const f64 *ptr) {
    return _mm256_blendv_pd(src, _mm256_maskload_pd(ptr, mask),
                            _mm256_castsi256_pd(mask));
}
template <>
inline void store_masked(f64 *ptr, f64x4::mask_type mask, f64x4 v) {
    _mm256_maskstore_pd(ptr, mask, v);
}

template <>
inline u64x4 load_masked(u64x4::mask_type mask, u64x4 src, const u64 *ptr) {
    return _mm256_blendv_epi8(
        src, _mm256_maskload_epi64(reinterpret_cast<const long long *>(ptr), mask),
        mask);
}
template <>
inline void store_masked(u64 *ptr, u64x4::mask_type mask, u64x4 v) {
    _mm256_maskstore_epi64(reinterpret_cast<long long *>(ptr), mask, v);
}

template <>
inline f32x4 load_masked(f32x4::mask_type mask, f32x4 src, const f32 *ptr) {
    return _mm_blendv_ps(src, _mm_maskload_ps(ptr, mask), _mm_castsi128_ps(mask));
}
template <>
inline void store_masked(f32 *ptr, f32x4::mask_type mask, f32x4 v) {
    _mm_maskstore_ps(ptr, mask, v);
}

template <>
inline u32x4 load_masked(u32x4::mask_type mask, u32x4 src, const u32 *ptr) {
    return _mm_blendv_epi8(
        src, _mm_maskload_epi32(reinterpret_cast<const int *>(ptr), mask), mask);
}
template <>
inline void store_masked(u32 *ptr, u32x4::mask_type mask, u32x4 v) {
    _mm_maskstore_epi32(reinterpret_cast<int *>(ptr), mask, v);
}

template <>
inline f64x2 load_masked(f64x2::mask_type mask, f64x2 src, const f64 *ptr) {
    return _mm_blendv_pd(src, _mm_maskload_pd(ptr, mask), _mm_castsi128_pd(mask));
}
template <>
inline void store_masked(f64 *ptr, f64x2::mask_type mask, f64x2 v) {
    _mm_maskstore_pd(ptr, mask, v);
}

template <>
inline u64x2 load_masked(u64x2::mask_type mask, u64x2 src, const u64 *ptr) {
    return _mm_blendv_epi8(
        src, _mm_maskload_epi64(reinterpret_cast<const long long *>(ptr), mask), mask);
}
template <>
inline void store_masked(u64 *ptr, u64x2::mask_type mask, u64x2 v) {
    _mm_maskstore_epi64(reinterpret_cast<long long *>(ptr), mask, v);
}

#endif  // SIMD_MAX_WIDTH_ >= 2

#if SIMD_MAX_WIDTH_ >= 0

using f32x1 = vec<f32, 1>;
using u32x1 = vec<u32, 1>;

using f64x1 = vec<f64, 1>;
using u64x1 = vec<u64, 1>;

template <ScalarType Number>
struct scalar_vec {
    using scalar_type = Number;
    using mask_type = bool;
    static constexpr width_type width = 1;

    using simd_type = Number;

    simd_type v;

    using This = vec<Number, 1>;

    constexpr scalar_vec() : v(0) {}
    constexpr scalar_vec(simd_type v) : v(v) {}
    constexpr operator simd_type() const { return v; }

    constexpr This operator-() const { return -v; }

    constexpr This operator+(This o) const { return v + o.v; }
    constexpr This operator+(scalar_type s) const { return v + s; }
    friend constexpr This operator+(scalar_type s, This o) { return s + o.v; }

    constexpr This operator-(This o) const { return v - o.v; }
    constexpr This operator-(scalar_type s) const { return v - s; }
    friend constexpr This operator-(scalar_type s, This o) { return s - o.v; }

    constexpr This operator*(This o) const { return v * o.v; }
    constexpr This operator*(scalar_type s) const { return v * s; }
    friend constexpr This operator*(scalar_type s, This o) { return s * o.v; }

    constexpr This operator/(This o) const { return v / o.v; }
    constexpr This operator/(scalar_type s) const { return v / s; }
    friend constexpr This operator/(scalar_type s, This o) { return s / o.v; }

    constexpr mask_type operator==(This o) const { return v == o.v; }
    constexpr mask_type operator!=(This o) const { return v != o.v; }
    constexpr mask_type operator<(This o) const { return v < o.v; }
    constexpr mask_type operator<=(This o) const { return v <= o.v; }
    constexpr mask_type operator>(This o) const { return v > o.v; }
    constexpr mask_type operator>=(This o) const { return v >= o.v; }

    constexpr mask_type operator==(scalar_type s) const { return v == s; }
    constexpr mask_type operator!=(scalar_type s) const { return v != s; }
    constexpr mask_type operator<(scalar_type s) const { return v < s; }
    constexpr mask_type operator<=(scalar_type s) const { return v <= s; }
    constexpr mask_type operator>(scalar_type s) const { return v > s; }
    constexpr mask_type operator>=(scalar_type s) const { return v >= s; }

    // == and != need no reversed friend: mask_type is bool here, so the
    // compiler synthesizes `s == v` from the member overload above.

    friend constexpr mask_type operator<(scalar_type s, This o) { return o > s; }
    friend constexpr mask_type operator<=(scalar_type s, This o) { return o >= s; }
    friend constexpr mask_type operator>(scalar_type s, This o) { return o < s; }
    friend constexpr mask_type operator>=(scalar_type s, This o) { return o <= s; }
};

template <>
struct vec<f32, 1> : scalar_vec<f32> {
    using scalar_vec::scalar_vec;
    constexpr vec(scalar_vec<f32> b) : scalar_vec(b) {}
};

template <>
struct vec<u32, 1> : scalar_vec<u32> {
    using scalar_vec::scalar_vec;
    constexpr vec(scalar_vec<u32> b) : scalar_vec(b) {}

    constexpr This operator&(This o) const { return v & o.v; }
    constexpr This operator|(This o) const { return v | o.v; }
    constexpr This operator^(This o) const { return v ^ o.v; }
    constexpr This operator~() const { return ~v; }
    constexpr This operator<<(int n) const { return v << n; }
    constexpr This operator>>(int n) const { return v >> n; }
};

static_assert(VecType<f32x1>, "vec<f32, 1> does not satisfy VecType");
static_assert(VecType<u32x1>, "vec<u32, 1> does not satisfy VecType");

template <>
constexpr f32x1 splat<f32x1>(f32x1::scalar_type s) { return s; }

template <>
constexpr u32x1 splat<u32x1>(u32x1::scalar_type s) { return s; }

// std::fma is the genuinely fused, single-rounding form. It is not usable in a
// constant expression before C++26 (gcc folds it, clang does not), so unlike
// the rest of this backend these are inline rather than constexpr.
template <>
inline f32x1 fma(f32x1 a, f32x1 b, f32x1 c) {
    return std::fma(f32(a), f32(b), f32(c));
}

template <>
inline f32x1 fms(f32x1 a, f32x1 b, f32x1 c) {
    return std::fma(f32(a), f32(b), -f32(c));
}

template <>
inline f32x1 fms2(f32x1 a, f32x1 b, f32x1 c) {
    return std::fma(-f32(a), f32(b), f32(c));
}

template <>
inline f32x1 load(const f32x1::scalar_type *ptr) { return *ptr; }

template <>
inline u32x1 load(const u32x1::scalar_type *ptr) { return *ptr; }

template <width_type lane>
inline f32x1 load_lane(f32x1 /*v*/, const f32x1::scalar_type *ptr) {
    static_assert(lane < 1);
    return *ptr;
}

template <width_type lane>
inline u32x1 load_lane(u32x1 /*v*/, const u32x1::scalar_type *ptr) {
    static_assert(lane < 1);
    return *ptr;
}

template <>
inline void store(f32x1::scalar_type *ptr, f32x1 v) { *ptr = v; }

template <>
inline void store(u32x1::scalar_type *ptr, u32x1 v) { *ptr = v; }

template <width_type lane>
inline void store_lane(f32x1::scalar_type *ptr, f32x1 v) {
    static_assert(lane < 1);
    *ptr = v;
}

template <width_type lane>
inline void store_lane(u32x1::scalar_type *ptr, u32x1 v) {
    static_assert(lane < 1);
    *ptr = v;
}

template <width_type lane>
constexpr f32x1::scalar_type get_lane(f32x1 v) {
    static_assert(lane < 1);
    return v;
}

template <width_type lane>
constexpr u32x1::scalar_type get_lane(u32x1 v) {
    static_assert(lane < 1);
    return v;
}

template <width_type lane>
constexpr f32x1 set_lane(f32x1 /*v*/, f32x1::scalar_type s) {
    static_assert(lane < 1);
    return s;
}

template <width_type lane>
constexpr u32x1 set_lane(u32x1 /*v*/, u32x1::scalar_type s) {
    static_assert(lane < 1);
    return s;
}

template <>
constexpr f32x1 min(f32x1 a, f32x1 b) { return f32(a) < f32(b) ? a : b; }

template <>
constexpr u32x1 min(u32x1 a, u32x1 b) { return u32(a) < u32(b) ? a : b; }

template <>
constexpr f32x1 max(f32x1 a, f32x1 b) { return f32(a) > f32(b) ? a : b; }

template <>
constexpr u32x1 max(u32x1 a, u32x1 b) { return u32(a) > u32(b) ? a : b; }

template <>
constexpr f32x1 abs(f32x1 v) { return std::fabs(f32(v)); }

template <>
constexpr f32x1 sqrt(f32x1 v) { return std::sqrt(f32(v)); }

template <>
constexpr f32x1 recip(f32x1 v) { return 1.f / f32(v); }

template <>
constexpr f32x1 rsqrt(f32x1 v) { return 1.f / std::sqrt(f32(v)); }

template <>
constexpr f32x1 round(f32x1 v) { return std::nearbyint(f32(v)); }

template <>
constexpr f32x1 floor(f32x1 v) { return std::floor(f32(v)); }

template <>
constexpr f32x1 ceil(f32x1 v) { return std::ceil(f32(v)); }

template <>
constexpr f32x1 trunc(f32x1 v) { return std::trunc(f32(v)); }

template <>
constexpr f32x1 select(bool mask, f32x1 a, f32x1 b) { return mask ? a : b; }

template <>
constexpr u32x1 select(bool mask, u32x1 a, u32x1 b) { return mask ? a : b; }

template <>
constexpr f32x1 select_add(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? a + b : src;
}
template <>
constexpr f32x1 select_sub(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? a - b : src;
}
template <>
constexpr f32x1 select_mul(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? a * b : src;
}
template <>
constexpr f32x1 select_div(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? a / b : src;
}
template <>
constexpr f32x1 select_min(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? min<f32x1>(a, b) : src;
}
template <>
constexpr f32x1 select_max(bool mask, f32x1 src, f32x1 a, f32x1 b) {
    return mask ? max<f32x1>(a, b) : src;
}
template <>
inline f32x1 select_fma(bool mask, f32x1 src, f32x1 a, f32x1 b, f32x1 c) {
    return mask ? fma<f32x1>(a, b, c) : src;
}
template <>
inline f32x1 select_fms(bool mask, f32x1 src, f32x1 a, f32x1 b, f32x1 c) {
    return mask ? fms<f32x1>(a, b, c) : src;
}
template <>
inline f32x1 select_fms2(bool mask, f32x1 src, f32x1 a, f32x1 b, f32x1 c) {
    return mask ? fms2<f32x1>(a, b, c) : src;
}

template <>
constexpr f32x1::scalar_type hsum(f32x1 v) { return v; }
template <>
constexpr u32x1::scalar_type hsum(u32x1 v) { return v; }
template <>
constexpr f32x1::scalar_type hmin(f32x1 v) { return v; }
template <>
constexpr u32x1::scalar_type hmin(u32x1 v) { return v; }
template <>
constexpr f32x1::scalar_type hmax(f32x1 v) { return v; }
template <>
constexpr u32x1::scalar_type hmax(u32x1 v) { return v; }

template <>
constexpr f32x1 clamp(f32x1 v, f32x1 lo, f32x1 hi) {
    return min<f32x1>(max<f32x1>(v, lo), hi);
}
template <>
constexpr u32x1 clamp(u32x1 v, u32x1 lo, u32x1 hi) {
    return min<u32x1>(max<u32x1>(v, lo), hi);
}

template <>
constexpr f32x1::scalar_type dot(f32x1 a, f32x1 b) { return hsum<f32x1>(a * b); }
template <>
constexpr u32x1::scalar_type dot(u32x1 a, u32x1 b) { return hsum<u32x1>(a * b); }

constexpr bool any(bool mask) { return mask; }
constexpr bool all(bool mask) { return mask; }
constexpr bool none(bool mask) { return !mask; }
constexpr u32 movemask(bool mask) { return mask ? 1u : 0u; }

template <>
constexpr u32x1 cast<u32x1, f32x1>(f32x1 v) {
    // Saturating, to agree with the vector backends: a bare static_cast is
    // undefined behaviour for negatives, NaN, and anything at or above 2^32.
    const f32 x = v;
    if (!(x > 0.f)) return u32(0);              // negatives and NaN
    if (x >= 4294967296.f) return ~u32(0);
    return static_cast<u32>(x);
}
template <>
constexpr f32x1 cast<f32x1, u32x1>(u32x1 v) {
    return static_cast<f32>(u32(v));
}

template <>
constexpr u32x1 reinterpret<u32x1, f32x1>(f32x1 v) {
    f32 x = v;
    return __builtin_bit_cast(u32, x);
}
template <>
constexpr f32x1 reinterpret<f32x1, u32x1>(u32x1 v) {
    u32 x = v;
    return __builtin_bit_cast(f32, x);
}

template <>
struct vec<f64, 1> : scalar_vec<f64> {
    using scalar_vec::scalar_vec;
    constexpr vec(scalar_vec<f64> b) : scalar_vec(b) {}
};

template <>
struct vec<u64, 1> : scalar_vec<u64> {
    using scalar_vec::scalar_vec;
    constexpr vec(scalar_vec<u64> b) : scalar_vec(b) {}

    constexpr This operator&(This o) const { return v & o.v; }
    constexpr This operator|(This o) const { return v | o.v; }
    constexpr This operator^(This o) const { return v ^ o.v; }
    constexpr This operator~() const { return ~v; }
    constexpr This operator<<(int n) const { return v << n; }
    constexpr This operator>>(int n) const { return v >> n; }
};

static_assert(VecType<f64x1>, "vec<f64, 1> does not satisfy VecType");
static_assert(VecType<u64x1>, "vec<u64, 1> does not satisfy VecType");

template <>
constexpr f64x1 splat<f64x1>(f64x1::scalar_type s) { return s; }

template <>
constexpr u64x1 splat<u64x1>(u64x1::scalar_type s) { return s; }

// See the f32x1 note above: the fused form is not a constant expression.
template <>
inline f64x1 fma(f64x1 a, f64x1 b, f64x1 c) {
    return std::fma(f64(a), f64(b), f64(c));
}

template <>
inline f64x1 fms(f64x1 a, f64x1 b, f64x1 c) {
    return std::fma(f64(a), f64(b), -f64(c));
}

template <>
inline f64x1 fms2(f64x1 a, f64x1 b, f64x1 c) {
    return std::fma(-f64(a), f64(b), f64(c));
}

template <>
inline f64x1 load(const f64x1::scalar_type *ptr) { return *ptr; }

template <width_type lane>
inline f64x1 load_lane(f64x1 /*v*/, const f64x1::scalar_type *ptr) {
    static_assert(lane < 1);
    return *ptr;
}

template <>
inline void store(f64x1::scalar_type *ptr, f64x1 v) { *ptr = v; }

template <width_type lane>
inline void store_lane(f64x1::scalar_type *ptr, f64x1 v) {
    static_assert(lane < 1);
    *ptr = v;
}

template <width_type lane>
constexpr f64x1::scalar_type get_lane(f64x1 v) {
    static_assert(lane < 1);
    return v;
}

template <width_type lane>
constexpr f64x1 set_lane(f64x1 /*v*/, f64x1::scalar_type s) {
    static_assert(lane < 1);
    return s;
}

template <>
inline u64x1 load(const u64x1::scalar_type *ptr) { return *ptr; }

template <width_type lane>
inline u64x1 load_lane(u64x1 /*v*/, const u64x1::scalar_type *ptr) {
    static_assert(lane < 1);
    return *ptr;
}

template <>
inline void store(u64x1::scalar_type *ptr, u64x1 v) { *ptr = v; }

template <width_type lane>
inline void store_lane(u64x1::scalar_type *ptr, u64x1 v) {
    static_assert(lane < 1);
    *ptr = v;
}

template <width_type lane>
constexpr u64x1::scalar_type get_lane(u64x1 v) {
    static_assert(lane < 1);
    return v;
}

template <width_type lane>
constexpr u64x1 set_lane(u64x1 /*v*/, u64x1::scalar_type s) {
    static_assert(lane < 1);
    return s;
}

template <>
constexpr f64x1 min(f64x1 a, f64x1 b) { return f64(a) < f64(b) ? a : b; }

template <>
constexpr u64x1 min(u64x1 a, u64x1 b) { return u64(a) < u64(b) ? a : b; }

template <>
constexpr f64x1 max(f64x1 a, f64x1 b) { return f64(a) > f64(b) ? a : b; }

template <>
constexpr u64x1 max(u64x1 a, u64x1 b) { return u64(a) > u64(b) ? a : b; }

template <>
constexpr f64x1 abs(f64x1 v) { return std::fabs(f64(v)); }

template <>
constexpr f64x1 sqrt(f64x1 v) { return std::sqrt(f64(v)); }

template <>
constexpr f64x1 recip(f64x1 v) { return 1.0 / f64(v); }

template <>
constexpr f64x1 rsqrt(f64x1 v) { return 1.0 / std::sqrt(f64(v)); }

template <>
constexpr f64x1 round(f64x1 v) { return std::nearbyint(f64(v)); }

template <>
constexpr f64x1 floor(f64x1 v) { return std::floor(f64(v)); }

template <>
constexpr f64x1 ceil(f64x1 v) { return std::ceil(f64(v)); }

template <>
constexpr f64x1 trunc(f64x1 v) { return std::trunc(f64(v)); }

template <>
constexpr f64x1 select(bool mask, f64x1 a, f64x1 b) { return mask ? a : b; }

template <>
constexpr u64x1 select(bool mask, u64x1 a, u64x1 b) { return mask ? a : b; }

template <>
constexpr f64x1 select_add(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? a + b : src;
}

template <>
constexpr f64x1 select_sub(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? a - b : src;
}

template <>
constexpr f64x1 select_mul(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? a * b : src;
}

template <>
constexpr f64x1 select_div(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? a / b : src;
}

template <>
constexpr f64x1 select_min(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? min<f64x1>(a, b) : src;
}

template <>
constexpr f64x1 select_max(bool mask, f64x1 src, f64x1 a, f64x1 b) {
    return mask ? max<f64x1>(a, b) : src;
}

template <>
inline f64x1 select_fma(bool mask, f64x1 src, f64x1 a, f64x1 b, f64x1 c) {
    return mask ? fma<f64x1>(a, b, c) : src;
}

template <>
inline f64x1 select_fms(bool mask, f64x1 src, f64x1 a, f64x1 b, f64x1 c) {
    return mask ? fms<f64x1>(a, b, c) : src;
}

template <>
inline f64x1 select_fms2(bool mask, f64x1 src, f64x1 a, f64x1 b, f64x1 c) {
    return mask ? fms2<f64x1>(a, b, c) : src;
}

template <>
constexpr f64x1::scalar_type hsum(f64x1 v) { return v; }
template <>
constexpr u64x1::scalar_type hsum(u64x1 v) { return v; }
template <>
constexpr f64x1::scalar_type hmin(f64x1 v) { return v; }
template <>
constexpr u64x1::scalar_type hmin(u64x1 v) { return v; }
template <>
constexpr f64x1::scalar_type hmax(f64x1 v) { return v; }
template <>
constexpr u64x1::scalar_type hmax(u64x1 v) { return v; }

template <>
constexpr f64x1 clamp(f64x1 v, f64x1 lo, f64x1 hi) {
    return min<f64x1>(max<f64x1>(v, lo), hi);
}
template <>
constexpr u64x1 clamp(u64x1 v, u64x1 lo, u64x1 hi) {
    return min<u64x1>(max<u64x1>(v, lo), hi);
}

template <>
constexpr f64x1::scalar_type dot(f64x1 a, f64x1 b) { return hsum<f64x1>(a * b); }
template <>
constexpr u64x1::scalar_type dot(u64x1 a, u64x1 b) { return hsum<u64x1>(a * b); }

template <>
constexpr u64x1 cast<u64x1, f64x1>(f64x1 v) {
    // Saturating; see the f32x1 note above.
    const f64 x = v;
    if (!(x > 0.0)) return u64(0);              // negatives and NaN
    if (x >= 18446744073709551616.0) return ~u64(0);
    return static_cast<u64>(x);
}
template <>
constexpr f64x1 cast<f64x1, u64x1>(u64x1 v) {
    return static_cast<f64>(u64(v));
}

template <>
constexpr u64x1 reinterpret<u64x1, f64x1>(f64x1 v) {
    f64 x = v;
    return __builtin_bit_cast(u64, x);
}
template <>
constexpr f64x1 reinterpret<f64x1, u64x1>(u64x1 v) {
    u64 x = v;
    return __builtin_bit_cast(f64, x);
}

template <>
constexpr f32x1 iota<f32x1>() { return 0.f; }

template <>
constexpr u32x1 iota<u32x1>() { return 0u; }

template <>
constexpr f64x1 iota<f64x1>() { return 0.0; }

template <>
constexpr u64x1 iota<u64x1>() { return 0u; }

template <>
constexpr f32x1::mask_type mask_from_bits<f32x1>(u32 bits) {
    return (bits & 1u) != 0u;
}

template <>
constexpr u32x1::mask_type mask_from_bits<u32x1>(u32 bits) {
    return (bits & 1u) != 0u;
}

template <>
constexpr f64x1::mask_type mask_from_bits<f64x1>(u32 bits) {
    return (bits & 1u) != 0u;
}

template <>
constexpr u64x1::mask_type mask_from_bits<u64x1>(u32 bits) {
    return (bits & 1u) != 0u;
}

// The scalar mask is a bool, so these are plain overloads rather than
// specializations -- and the reason the mask_* spellings exist at all.
constexpr bool mask_and(bool a, bool b) { return a && b; }
constexpr bool mask_or(bool a, bool b) { return a || b; }
constexpr bool mask_xor(bool a, bool b) { return a != b; }
constexpr bool mask_not(bool m) { return !m; }
constexpr bool mask_andnot(bool a, bool b) { return a && !b; }

#endif  // SIMD_MAX_WIDTH_ >= 0

// Masked memory access: the portable implementation
//
// These sit after every backend so that ordinary lookup sees all the per-type
// load_lane/store_lane overloads they are built from. Only AVX2 replaces them,
// with real maskload/maskstore instructions; NEON, SSE4.2 and the scalar
// fallback all use exactly this code.

namespace detail {

// Walking the set bits of movemask and touching one lane at a time is what
// makes the safe form safe: a lane the mask excludes is never addressed.
template <VecType V, usize... I>
inline V load_masked_lanewise(typename V::mask_type mask, V src,
                              const typename V::scalar_type *ptr,
                              std::index_sequence<I...>) {
    const u32 bits = movemask(mask);
    V r = src;
    ((((bits >> I) & 1u) ? (void)(r = load_lane<I>(r, ptr + I)) : (void)0), ...);
    return r;
}

template <VecType V, usize... I>
inline void store_masked_lanewise(typename V::scalar_type *ptr,
                                  typename V::mask_type mask, V v,
                                  std::index_sequence<I...>) {
    const u32 bits = movemask(mask);
    ((((bits >> I) & 1u) ? (void)store_lane<I>(ptr + I, v) : (void)0), ...);
}

}  // namespace detail

template <VecType V>
inline V load_masked(typename V::mask_type mask, V src,
                     const typename V::scalar_type *ptr) {
    return detail::load_masked_lanewise<V>(mask, src, ptr,
                                           std::make_index_sequence<V::width>{});
}

template <VecType V>
inline void store_masked(typename V::scalar_type *ptr, typename V::mask_type mask,
                         V v) {
    detail::store_masked_lanewise<V>(ptr, mask, v,
                                     std::make_index_sequence<V::width>{});
}

template <VecType V>
inline V load_masked_unsafe(typename V::mask_type mask, V src,
                            const typename V::scalar_type *ptr) {
    return select<V>(mask, load<V>(ptr), src);
}


// Widest vector register this target offers, in bytes.
#if SIMD_MAX_WIDTH_ == 2
inline constexpr usize plat_vector_bytes = 32;  // AVX2 ymm registers
#elif SIMD_MAX_WIDTH_ == 1
inline constexpr usize plat_vector_bytes = 16;  // NEON q / SSE xmm registers
#elif SIMD_MAX_WIDTH_ == 0
inline constexpr usize plat_vector_bytes = 0;   // no vector unit
#endif

// The widest width this target supports for scalar type N, at least 1 so the
// scalar fallback is always reachable.
template <ScalarType N>
struct plat_width {
    static constexpr width_type value =
        plat_vector_bytes / sizeof(N) > 1 ? plat_vector_bytes / sizeof(N) : 1;

    // A scalar type the backend has no vec for fails here, rather than quietly
    // naming a vec<N, W> that was never implemented.
    static_assert(VecType<vec<N, value>>,
                  "simd: no vec implementation for this scalar type on this target");
};

// The widest vec type this target supports for scalar type N: plat_vec<f32> is
// vec<f32, 4> under NEON and vec<f32, 1> with no vector unit. Reach the
// matching mask with plat_vec<f32>::mask_type.
template <ScalarType N>
using plat_vec = vec<N, plat_width<N>::value>;

}  // namespace simd

#endif  // SIMD_HPP_
