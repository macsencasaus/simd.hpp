#ifndef SIMD_HPP_
#define SIMD_HPP_

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>

// SIMD_MAX_WIDTH_ selects the widest available backend:
//   1 -> 128-bit vectors (NEON), 0 -> scalar fallback.
// Every branch below must define it, so the scalar path is always reachable.
#if defined(__aarch64__) || defined(_M_ARM64)

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_MAX_WIDTH_ 1
#else
#define SIMD_MAX_WIDTH_ 0
#endif

#elif defined(__x86_64__) || defined(_M_X64)

// No x86 backend yet; the scalar path covers it.
#define SIMD_MAX_WIDTH_ 0

#else

#define SIMD_MAX_WIDTH_ 0

#endif

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

template <width_type lane, VecType V>
inline V load_lane(V v, const typename V::scalar_type *ptr);

template <VecType V>
inline void store(typename V::scalar_type *ptr, V v);

template <width_type lane, VecType V>
inline void store_lane(typename V::scalar_type *ptr, V v);

template <width_type lane, VecType V>
inline typename V::scalar_type get_lane(V v);

template <width_type lane, VecType V>
inline V set_lane(V v, typename V::scalar_type s);

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

#if defined(__ARM_NEON)
    using simd_type = float32x4_t;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.f) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const {
#if defined(__ARM_NEON)
        return vnegq_f32(v);
#endif
    }

    This operator+(This o) const {
#if defined(__ARM_NEON)
        return vaddq_f32(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(__ARM_NEON)
        return vsubq_f32(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(__ARM_NEON)
        return vmulq_f32(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const {
#if defined(__ARM_NEON)
        return vdivq_f32(*this, o);
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

#if defined(__ARM_NEON)
    using simd_type = uint32x4_t;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const {
#if defined(__ARM_NEON)
        return vaddq_u32(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(__ARM_NEON)
        return vsubq_u32(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(__ARM_NEON)
        return vmulq_u32(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const {
#if defined(__ARM_NEON)
        return vceqq_u32(*this, o);
#endif
    }
    mask_type operator!=(This o) const {
#if defined(__ARM_NEON)
        return vmvnq_u32(vceqq_u32(*this, o));
#endif
    }
    mask_type operator<(This o) const {
#if defined(__ARM_NEON)
        return vcltq_u32(*this, o);
#endif
    }
    mask_type operator<=(This o) const {
#if defined(__ARM_NEON)
        return vcleq_u32(*this, o);
#endif
    }
    mask_type operator>(This o) const {
#if defined(__ARM_NEON)
        return vcgtq_u32(*this, o);
#endif
    }
    mask_type operator>=(This o) const {
#if defined(__ARM_NEON)
        return vcgeq_u32(*this, o);
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
#if defined(__ARM_NEON)
        return vandq_u32(*this, o);
#endif
    }
    This operator|(This o) const {
#if defined(__ARM_NEON)
        return vorrq_u32(*this, o);
#endif
    }
    This operator^(This o) const {
#if defined(__ARM_NEON)
        return veorq_u32(*this, o);
#endif
    }
    This operator~() const {
#if defined(__ARM_NEON)
        return vmvnq_u32(v);
#endif
    }

    This operator<<(int n) const {
#if defined(__ARM_NEON)
        return vshlq_u32(*this, vdupq_n_s32(n));
#endif
    }
    This operator>>(int n) const {
#if defined(__ARM_NEON)
        return vshlq_u32(*this, vdupq_n_s32(-n));
#endif
    }
};

static_assert(VecType<u32x4>, "vec<u32, 4> does not satisfy VecType");

inline f32x4::mask_type f32x4::operator==(This o) const {
#if defined(__ARM_NEON)
    return vceqq_f32(*this, o);
#endif
}
inline f32x4::mask_type f32x4::operator!=(This o) const {
#if defined(__ARM_NEON)
    return vmvnq_u32(vceqq_f32(*this, o));
#endif
}
inline f32x4::mask_type f32x4::operator<(This o) const {
#if defined(__ARM_NEON)
    return vcltq_f32(*this, o);
#endif
}
inline f32x4::mask_type f32x4::operator<=(This o) const {
#if defined(__ARM_NEON)
    return vcleq_f32(*this, o);
#endif
}
inline f32x4::mask_type f32x4::operator>(This o) const {
#if defined(__ARM_NEON)
    return vcgtq_f32(*this, o);
#endif
}
inline f32x4::mask_type f32x4::operator>=(This o) const {
#if defined(__ARM_NEON)
    return vcgeq_f32(*this, o);
#endif
}

template <>
inline f32x4 splat<f32x4>(f32x4::scalar_type s) {
#if defined(__ARM_NEON)
    return vdupq_n_f32(s);
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
#if defined(__ARM_NEON)
    return vdupq_n_u32(s);
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
#if defined(__ARM_NEON)
    return vfmaq_f32(c, a, b);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f32x4 fms(f32x4 a, f32x4 b, f32x4 c) {
#if defined(__ARM_NEON)
    // Negation is exact, so this keeps the single rounding of the fused op.
    return vnegq_f32(vfmsq_f32(c, a, b));
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f32x4 fms2(f32x4 a, f32x4 b, f32x4 c) {
#if defined(__ARM_NEON)
    return vfmsq_f32(c, a, b);
#else
    return c - a * b;
#endif
}

template <>
inline f32x4 load(const f32x4::scalar_type *ptr) {
#if defined(__ARM_NEON)
    return vld1q_f32(ptr);
#endif
}

template <width_type lane>
inline f32x4 load_lane(f32x4 v, const f32x4::scalar_type *ptr) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vld1q_lane_f32(ptr, v, lane);
#endif
}

template <>
inline void store(f32x4::scalar_type *ptr, f32x4 v) {
#if defined(__ARM_NEON)
    vst1q_f32(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(f32x4::scalar_type *ptr, f32x4 v) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    vst1q_lane_f32(ptr, v, lane);
#endif
}

template <width_type lane>
inline f32x4::scalar_type get_lane(f32x4 v) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vgetq_lane_f32(v, lane);
#endif
}

template <width_type lane>
inline f32x4 set_lane(f32x4 v, f32x4::scalar_type s) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vsetq_lane_f32(s, v, lane);
#endif
}

template <>
inline u32x4 load(const u32x4::scalar_type *ptr) {
#if defined(__ARM_NEON)
    return vld1q_u32(ptr);
#endif
}

template <width_type lane>
inline u32x4 load_lane(u32x4 v, const u32x4::scalar_type *ptr) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vld1q_lane_u32(ptr, v, lane);
#endif
}

template <>
inline void store(u32x4::scalar_type *ptr, u32x4 v) {
#if defined(__ARM_NEON)
    vst1q_u32(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(u32x4::scalar_type *ptr, u32x4 v) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    vst1q_lane_u32(ptr, v, lane);
#endif
}

template <width_type lane>
inline u32x4::scalar_type get_lane(u32x4 v) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vgetq_lane_u32(v, lane);
#endif
}

template <width_type lane>
inline u32x4 set_lane(u32x4 v, u32x4::scalar_type s) {
    static_assert(lane < 4);
#if defined(__ARM_NEON)
    return vsetq_lane_u32(s, v, lane);
#endif
}

template <>
inline f32x4 min(f32x4 a, f32x4 b) {
#if defined(__ARM_NEON)
    return vminq_f32(a, b);
#endif
}

template <>
inline u32x4 min(u32x4 a, u32x4 b) {
#if defined(__ARM_NEON)
    return vminq_u32(a, b);
#endif
}

template <>
inline f32x4 max(f32x4 a, f32x4 b) {
#if defined(__ARM_NEON)
    return vmaxq_f32(a, b);
#endif
}

template <>
inline u32x4 max(u32x4 a, u32x4 b) {
#if defined(__ARM_NEON)
    return vmaxq_u32(a, b);
#endif
}

template <>
inline f32x4 abs(f32x4 v) {
#if defined(__ARM_NEON)
    return vabsq_f32(v);
#endif
}

template <>
inline f32x4 sqrt(f32x4 v) {
#if defined(__ARM_NEON)
    return vsqrtq_f32(v);
#endif
}

template <>
inline f32x4 recip(f32x4 v) {
#if defined(__ARM_NEON)
    f32x4 e = vrecpeq_f32(v);
    e = vmulq_f32(vrecpsq_f32(v, e), e);
    e = vmulq_f32(vrecpsq_f32(v, e), e);
    return e;
#endif
}

template <>
inline f32x4 rsqrt(f32x4 v) {
#if defined(__ARM_NEON)
    f32x4 e = vrsqrteq_f32(v);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(v, e), e), e);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(v, e), e), e);
    return e;
#endif
}

template <>
inline f32x4 round(f32x4 v) {
#if defined(__ARM_NEON)
    return vrndnq_f32(v);
#endif
}

template <>
inline f32x4 floor(f32x4 v) {
#if defined(__ARM_NEON)
    return vrndmq_f32(v);
#endif
}

template <>
inline f32x4 ceil(f32x4 v) {
#if defined(__ARM_NEON)
    return vrndpq_f32(v);
#endif
}

template <>
inline f32x4 trunc(f32x4 v) {
#if defined(__ARM_NEON)
    return vrndq_f32(v);
#endif
}

template <>
inline f32x4 select(f32x4::mask_type mask, f32x4 a, f32x4 b) {
#if defined(__ARM_NEON)
    return vbslq_f32(mask, a, b);
#endif
}

template <>
inline u32x4 select(u32x4::mask_type mask, u32x4 a, u32x4 b) {
#if defined(__ARM_NEON)
    return vbslq_u32(mask, a, b);
#endif
}

template <>
inline f32x4 select_add(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, a + b, src);
}

template <>
inline f32x4 select_sub(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, a - b, src);
}

template <>
inline f32x4 select_mul(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, a * b, src);
}

template <>
inline f32x4 select_div(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, a / b, src);
}

template <>
inline f32x4 select_min(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, min<f32x4>(a, b), src);
}

template <>
inline f32x4 select_max(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b) {
    return select<f32x4>(mask, max<f32x4>(a, b), src);
}

template <>
inline f32x4 select_fma(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b,
                           f32x4 c) {
    return select<f32x4>(mask, fma<f32x4>(a, b, c), src);
}

template <>
inline f32x4 select_fms(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b,
                           f32x4 c) {
    return select<f32x4>(mask, fms<f32x4>(a, b, c), src);
}

template <>
inline f32x4 select_fms2(f32x4::mask_type mask, f32x4 src, f32x4 a, f32x4 b,
                            f32x4 c) {
    return select<f32x4>(mask, fms2<f32x4>(a, b, c), src);
}

template <>
inline f32x4::scalar_type hsum(f32x4 v) {
#if defined(__ARM_NEON)
    return vaddvq_f32(v);
#endif
}

template <>
inline u32x4::scalar_type hsum(u32x4 v) {
#if defined(__ARM_NEON)
    return vaddvq_u32(v);
#endif
}

template <>
inline f32x4::scalar_type hmin(f32x4 v) {
#if defined(__ARM_NEON)
    return vminvq_f32(v);
#endif
}

template <>
inline u32x4::scalar_type hmin(u32x4 v) {
#if defined(__ARM_NEON)
    return vminvq_u32(v);
#endif
}

template <>
inline f32x4::scalar_type hmax(f32x4 v) {
#if defined(__ARM_NEON)
    return vmaxvq_f32(v);
#endif
}

template <>
inline u32x4::scalar_type hmax(u32x4 v) {
#if defined(__ARM_NEON)
    return vmaxvq_u32(v);
#endif
}

template <>
inline u32x4 cast<u32x4, f32x4>(f32x4 v) {
#if defined(__ARM_NEON)
    return vcvtq_u32_f32(v);
#endif
}

template <>
inline f32x4 cast<f32x4, u32x4>(u32x4 v) {
#if defined(__ARM_NEON)
    return vcvtq_f32_u32(v);
#endif
}

template <>
inline u32x4 reinterpret<u32x4, f32x4>(f32x4 v) {
#if defined(__ARM_NEON)
    return vreinterpretq_u32_f32(v);
#endif
}

template <>
inline f32x4 reinterpret<f32x4, u32x4>(u32x4 v) {
#if defined(__ARM_NEON)
    return vreinterpretq_f32_u32(v);
#endif
}

template <>
inline f32x4 clamp(f32x4 v, f32x4 lo, f32x4 hi) {
    return min<f32x4>(max<f32x4>(v, lo), hi);
}

template <>
inline u32x4 clamp(u32x4 v, u32x4 lo, u32x4 hi) {
    return min<u32x4>(max<u32x4>(v, lo), hi);
}

template <>
inline f32x4::scalar_type dot(f32x4 a, f32x4 b) {
    return hsum<f32x4>(a * b);
}

template <>
inline u32x4::scalar_type dot(u32x4 a, u32x4 b) {
    return hsum<u32x4>(a * b);
}

template <>
inline bool any(u32x4 mask) {
#if defined(__ARM_NEON)
    return vmaxvq_u32(mask) != 0;
#endif
}

template <>
inline bool all(u32x4 mask) {
#if defined(__ARM_NEON)
    return vminvq_u32(mask) == 0xFFFFFFFFu;
#endif
}

template <>
inline bool none(u32x4 mask) {
    return !any<u32x4>(mask);
}

template <>
inline u32 movemask(u32x4 mask) {
#if defined(__ARM_NEON)
    const uint32x4_t bits = {1u, 2u, 4u, 8u};
    return vaddvq_u32(vandq_u32(mask, bits));
#endif
}


template <>
struct vec<f64, 2> {
    using scalar_type = f64;
    using mask_type = u64x2;
    static constexpr width_type width = 2;

#if defined(__ARM_NEON)
    using simd_type = float64x2_t;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0.0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator-() const {
#if defined(__ARM_NEON)
        return vnegq_f64(v);
#endif
    }

    This operator+(This o) const {
#if defined(__ARM_NEON)
        return vaddq_f64(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(__ARM_NEON)
        return vsubq_f64(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(__ARM_NEON)
        return vmulq_f64(*this, o);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    This operator/(This o) const {
#if defined(__ARM_NEON)
        return vdivq_f64(*this, o);
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

#if defined(__ARM_NEON)
    using simd_type = uint64x2_t;
#endif

    simd_type v;

    using This = vec<scalar_type, width>;

    vec() : vec(0) {}
    constexpr vec(simd_type v) : v(v) {}
    vec(scalar_type s);
    constexpr operator simd_type() const { return v; }

    This operator+(This o) const {
#if defined(__ARM_NEON)
        return vaddq_u64(*this, o);
#endif
    }
    This operator+(scalar_type s) const;
    friend This operator+(scalar_type s, This o) { return o + s; }

    This operator-(This o) const {
#if defined(__ARM_NEON)
        return vsubq_u64(*this, o);
#endif
    }
    This operator-(scalar_type s) const;
    friend This operator-(scalar_type s, This o);

    This operator*(This o) const {
#if defined(__ARM_NEON)
        // NEON has no 64-bit vector multiply; fall back to lane-wise scalars.
        scalar_type lo = vgetq_lane_u64(*this, 0) * vgetq_lane_u64(o, 0);
        scalar_type hi = vgetq_lane_u64(*this, 1) * vgetq_lane_u64(o, 1);
        return vsetq_lane_u64(hi, vdupq_n_u64(lo), 1);
#endif
    }
    This operator*(scalar_type s) const;
    friend This operator*(scalar_type s, This o) { return o * s; }

    mask_type operator==(This o) const {
#if defined(__ARM_NEON)
        return vceqq_u64(*this, o);
#endif
    }
    mask_type operator!=(This o) const {
#if defined(__ARM_NEON)
        // No vmvnq_u64: invert through u32, which is bit-width agnostic.
        return vreinterpretq_u64_u32(
            vmvnq_u32(vreinterpretq_u32_u64(vceqq_u64(*this, o))));
#endif
    }
    mask_type operator<(This o) const {
#if defined(__ARM_NEON)
        return vcltq_u64(*this, o);
#endif
    }
    mask_type operator<=(This o) const {
#if defined(__ARM_NEON)
        return vcleq_u64(*this, o);
#endif
    }
    mask_type operator>(This o) const {
#if defined(__ARM_NEON)
        return vcgtq_u64(*this, o);
#endif
    }
    mask_type operator>=(This o) const {
#if defined(__ARM_NEON)
        return vcgeq_u64(*this, o);
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
#if defined(__ARM_NEON)
        return vandq_u64(*this, o);
#endif
    }
    This operator|(This o) const {
#if defined(__ARM_NEON)
        return vorrq_u64(*this, o);
#endif
    }
    This operator^(This o) const {
#if defined(__ARM_NEON)
        return veorq_u64(*this, o);
#endif
    }
    This operator~() const {
#if defined(__ARM_NEON)
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v)));
#endif
    }

    This operator<<(int n) const {
#if defined(__ARM_NEON)
        return vshlq_u64(*this, vdupq_n_s64(n));
#endif
    }
    This operator>>(int n) const {
#if defined(__ARM_NEON)
        return vshlq_u64(*this, vdupq_n_s64(-n));
#endif
    }
};

static_assert(VecType<u64x2>, "vec<u64, 2> does not satisfy VecType");

inline f64x2::mask_type f64x2::operator==(This o) const {
#if defined(__ARM_NEON)
    return vceqq_f64(*this, o);
#endif
}

inline f64x2::mask_type f64x2::operator!=(This o) const {
#if defined(__ARM_NEON)
    // No vmvnq_u64: invert through u32, which is bit-width agnostic.
    return vreinterpretq_u64_u32(
        vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(*this, o))));
#endif
}

inline f64x2::mask_type f64x2::operator<(This o) const {
#if defined(__ARM_NEON)
    return vcltq_f64(*this, o);
#endif
}

inline f64x2::mask_type f64x2::operator<=(This o) const {
#if defined(__ARM_NEON)
    return vcleq_f64(*this, o);
#endif
}

inline f64x2::mask_type f64x2::operator>(This o) const {
#if defined(__ARM_NEON)
    return vcgtq_f64(*this, o);
#endif
}

inline f64x2::mask_type f64x2::operator>=(This o) const {
#if defined(__ARM_NEON)
    return vcgeq_f64(*this, o);
#endif
}

template <>
inline f64x2 splat<f64x2>(f64x2::scalar_type s) {
#if defined(__ARM_NEON)
    return vdupq_n_f64(s);
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
#if defined(__ARM_NEON)
    return vdupq_n_u64(s);
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
#if defined(__ARM_NEON)
    return vfmaq_f64(c, a, b);
#else
    return a * b + c;
#endif
}

// a * b - c
template <>
inline f64x2 fms(f64x2 a, f64x2 b, f64x2 c) {
#if defined(__ARM_NEON)
    // Negation is exact, so this keeps the single rounding of the fused op.
    return vnegq_f64(vfmsq_f64(c, a, b));
#else
    return a * b - c;
#endif
}

// -(a * b) + c
template <>
inline f64x2 fms2(f64x2 a, f64x2 b, f64x2 c) {
#if defined(__ARM_NEON)
    return vfmsq_f64(c, a, b);
#else
    return c - a * b;
#endif
}

template <>
inline f64x2 load(const f64x2::scalar_type *ptr) {
#if defined(__ARM_NEON)
    return vld1q_f64(ptr);
#endif
}

template <width_type lane>
inline f64x2 load_lane(f64x2 v, const f64x2::scalar_type *ptr) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vld1q_lane_f64(ptr, v, lane);
#endif
}

template <>
inline void store(f64x2::scalar_type *ptr, f64x2 v) {
#if defined(__ARM_NEON)
    vst1q_f64(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(f64x2::scalar_type *ptr, f64x2 v) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    vst1q_lane_f64(ptr, v, lane);
#endif
}

template <width_type lane>
inline f64x2::scalar_type get_lane(f64x2 v) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vgetq_lane_f64(v, lane);
#endif
}

template <width_type lane>
inline f64x2 set_lane(f64x2 v, f64x2::scalar_type s) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vsetq_lane_f64(s, v, lane);
#endif
}

template <>
inline u64x2 load(const u64x2::scalar_type *ptr) {
#if defined(__ARM_NEON)
    return vld1q_u64(ptr);
#endif
}

template <width_type lane>
inline u64x2 load_lane(u64x2 v, const u64x2::scalar_type *ptr) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vld1q_lane_u64(ptr, v, lane);
#endif
}

template <>
inline void store(u64x2::scalar_type *ptr, u64x2 v) {
#if defined(__ARM_NEON)
    vst1q_u64(ptr, v);
#endif
}

template <width_type lane>
inline void store_lane(u64x2::scalar_type *ptr, u64x2 v) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    vst1q_lane_u64(ptr, v, lane);
#endif
}

template <width_type lane>
inline u64x2::scalar_type get_lane(u64x2 v) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vgetq_lane_u64(v, lane);
#endif
}

template <width_type lane>
inline u64x2 set_lane(u64x2 v, u64x2::scalar_type s) {
    static_assert(lane < 2);
#if defined(__ARM_NEON)
    return vsetq_lane_u64(s, v, lane);
#endif
}

template <>
inline f64x2 min(f64x2 a, f64x2 b) {
#if defined(__ARM_NEON)
    return vminq_f64(a, b);
#endif
}

template <>
inline u64x2 min(u64x2 a, u64x2 b) {
#if defined(__ARM_NEON)
    // No vminq_u64 at 64-bit width; blend on the comparison instead.
    return vbslq_u64(vcltq_u64(a, b), a, b);
#endif
}

template <>
inline f64x2 max(f64x2 a, f64x2 b) {
#if defined(__ARM_NEON)
    return vmaxq_f64(a, b);
#endif
}

template <>
inline u64x2 max(u64x2 a, u64x2 b) {
#if defined(__ARM_NEON)
    // No vmaxq_u64 at 64-bit width; blend on the comparison instead.
    return vbslq_u64(vcgtq_u64(a, b), a, b);
#endif
}

template <>
inline f64x2 abs(f64x2 v) {
#if defined(__ARM_NEON)
    return vabsq_f64(v);
#endif
}

template <>
inline f64x2 sqrt(f64x2 v) {
#if defined(__ARM_NEON)
    return vsqrtq_f64(v);
#endif
}

template <>
inline f64x2 round(f64x2 v) {
#if defined(__ARM_NEON)
    return vrndnq_f64(v);
#endif
}

template <>
inline f64x2 floor(f64x2 v) {
#if defined(__ARM_NEON)
    return vrndmq_f64(v);
#endif
}

template <>
inline f64x2 ceil(f64x2 v) {
#if defined(__ARM_NEON)
    return vrndpq_f64(v);
#endif
}

template <>
inline f64x2 trunc(f64x2 v) {
#if defined(__ARM_NEON)
    return vrndq_f64(v);
#endif
}

template <>
inline f64x2 recip(f64x2 v) {
#if defined(__ARM_NEON)
    // The estimate is ~8 bits and each Newton-Raphson step doubles it, so
    // double's 53-bit significand needs three steps where f32 needed two.
    f64x2 e = vrecpeq_f64(v);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    e = vmulq_f64(vrecpsq_f64(v, e), e);
    return e;
#endif
}

template <>
inline f64x2 rsqrt(f64x2 v) {
#if defined(__ARM_NEON)
    // Three refinement steps, for the same reason as recip above.
    f64x2 e = vrsqrteq_f64(v);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    e = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, e), e), e);
    return e;
#endif
}

template <>
inline f64x2 select(f64x2::mask_type mask, f64x2 a, f64x2 b) {
#if defined(__ARM_NEON)
    return vbslq_f64(mask, a, b);
#endif
}

template <>
inline u64x2 select(u64x2::mask_type mask, u64x2 a, u64x2 b) {
#if defined(__ARM_NEON)
    return vbslq_u64(mask, a, b);
#endif
}

template <>
inline f64x2 select_add(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, a + b, src);
}

template <>
inline f64x2 select_sub(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, a - b, src);
}

template <>
inline f64x2 select_mul(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, a * b, src);
}

template <>
inline f64x2 select_div(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, a / b, src);
}

template <>
inline f64x2 select_min(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, min<f64x2>(a, b), src);
}

template <>
inline f64x2 select_max(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b) {
    return select<f64x2>(mask, max<f64x2>(a, b), src);
}

template <>
inline f64x2 select_fma(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b, f64x2 c) {
    return select<f64x2>(mask, fma<f64x2>(a, b, c), src);
}

template <>
inline f64x2 select_fms(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b, f64x2 c) {
    return select<f64x2>(mask, fms<f64x2>(a, b, c), src);
}

template <>
inline f64x2 select_fms2(f64x2::mask_type mask, f64x2 src, f64x2 a,
                        f64x2 b, f64x2 c) {
    return select<f64x2>(mask, fms2<f64x2>(a, b, c), src);
}

template <>
inline f64x2::scalar_type hsum(f64x2 v) {
#if defined(__ARM_NEON)
    return vaddvq_f64(v);
#endif
}

template <>
inline u64x2::scalar_type hsum(u64x2 v) {
#if defined(__ARM_NEON)
    return vaddvq_u64(v);
#endif
}

template <>
inline f64x2::scalar_type hmin(f64x2 v) {
#if defined(__ARM_NEON)
    return vminvq_f64(v);
#endif
}

template <>
inline u64x2::scalar_type hmin(u64x2 v) {
#if defined(__ARM_NEON)
    // No vminvq_u64; only two lanes, so compare them directly.
    u64 a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
    return a < b ? a : b;
#endif
}

template <>
inline f64x2::scalar_type hmax(f64x2 v) {
#if defined(__ARM_NEON)
    return vmaxvq_f64(v);
#endif
}

template <>
inline u64x2::scalar_type hmax(u64x2 v) {
#if defined(__ARM_NEON)
    // No vmaxvq_u64; only two lanes, so compare them directly.
    u64 a = vgetq_lane_u64(v, 0), b = vgetq_lane_u64(v, 1);
    return a > b ? a : b;
#endif
}

template <>
inline u64x2 cast<u64x2, f64x2>(f64x2 v) {
#if defined(__ARM_NEON)
    return vcvtq_u64_f64(v);
#endif
}

template <>
inline f64x2 cast<f64x2, u64x2>(u64x2 v) {
#if defined(__ARM_NEON)
    return vcvtq_f64_u64(v);
#endif
}

template <>
inline u64x2 reinterpret<u64x2, f64x2>(f64x2 v) {
#if defined(__ARM_NEON)
    return vreinterpretq_u64_f64(v);
#endif
}

template <>
inline f64x2 reinterpret<f64x2, u64x2>(u64x2 v) {
#if defined(__ARM_NEON)
    return vreinterpretq_f64_u64(v);
#endif
}

template <>
inline f64x2 clamp(f64x2 v, f64x2 lo, f64x2 hi) {
    return min<f64x2>(max<f64x2>(v, lo), hi);
}

template <>
inline u64x2 clamp(u64x2 v, u64x2 lo, u64x2 hi) {
    return min<u64x2>(max<u64x2>(v, lo), hi);
}

template <>
inline f64x2::scalar_type dot(f64x2 a, f64x2 b) {
    return hsum<f64x2>(a * b);
}

template <>
inline u64x2::scalar_type dot(u64x2 a, u64x2 b) {
    return hsum<u64x2>(a * b);
}

template <>
inline bool any(u64x2 mask) {
#if defined(__ARM_NEON)
    return vmaxvq_u32(vreinterpretq_u32_u64(mask)) != 0;
#endif
}

template <>
inline bool all(u64x2 mask) {
#if defined(__ARM_NEON)
    return vminvq_u32(vreinterpretq_u32_u64(mask)) == 0xFFFFFFFFu;
#endif
}

template <>
inline bool none(u64x2 mask) {
    return !any<u64x2>(mask);
}

template <>
inline u32 movemask(u64x2 mask) {
#if defined(__ARM_NEON)
    const uint64x2_t bits = {1u, 2u};
    return static_cast<u32>(vaddvq_u64(vandq_u64(mask, bits)));
#endif
}

template <>
inline f32x4 iota<f32x4>() {
#if defined(__ARM_NEON)
    const float32x4_t r = {0.f, 1.f, 2.f, 3.f};
    return r;
#endif
}

template <>
inline u32x4 iota<u32x4>() {
#if defined(__ARM_NEON)
    const uint32x4_t r = {0u, 1u, 2u, 3u};
    return r;
#endif
}

template <>
inline f32x4::mask_type mask_from_bits<f32x4>(u32 bits) {
#if defined(__ARM_NEON)
    const uint32x4_t bit = {1u, 2u, 4u, 8u};
    return vceqq_u32(vandq_u32(vdupq_n_u32(bits), bit), bit);
#endif
}

template <>
inline u32x4::mask_type mask_from_bits<u32x4>(u32 bits) {
#if defined(__ARM_NEON)
    const uint32x4_t bit = {1u, 2u, 4u, 8u};
    return vceqq_u32(vandq_u32(vdupq_n_u32(bits), bit), bit);
#endif
}

template <>
inline u32x4 mask_and(u32x4 a, u32x4 b) { return a & b; }

template <>
inline u32x4 mask_or(u32x4 a, u32x4 b) { return a | b; }

template <>
inline u32x4 mask_xor(u32x4 a, u32x4 b) { return a ^ b; }

template <>
inline u32x4 mask_not(u32x4 m) { return ~m; }

template <>
inline u32x4 mask_andnot(u32x4 a, u32x4 b) { return a & ~b; }

template <>
inline f64x2 iota<f64x2>() {
#if defined(__ARM_NEON)
    const float64x2_t r = {0.0, 1.0};
    return r;
#endif
}

template <>
inline u64x2 iota<u64x2>() {
#if defined(__ARM_NEON)
    const uint64x2_t r = {0u, 1u};
    return r;
#endif
}

template <>
inline f64x2::mask_type mask_from_bits<f64x2>(u32 bits) {
#if defined(__ARM_NEON)
    const uint64x2_t bit = {1u, 2u};
    return vceqq_u64(vandq_u64(vdupq_n_u64(bits), bit), bit);
#endif
}

template <>
inline u64x2::mask_type mask_from_bits<u64x2>(u32 bits) {
#if defined(__ARM_NEON)
    const uint64x2_t bit = {1u, 2u};
    return vceqq_u64(vandq_u64(vdupq_n_u64(bits), bit), bit);
#endif
}

template <>
inline u64x2 mask_and(u64x2 a, u64x2 b) { return a & b; }

template <>
inline u64x2 mask_or(u64x2 a, u64x2 b) { return a | b; }

template <>
inline u64x2 mask_xor(u64x2 a, u64x2 b) { return a ^ b; }

template <>
inline u64x2 mask_not(u64x2 m) { return ~m; }

template <>
inline u64x2 mask_andnot(u64x2 a, u64x2 b) { return a & ~b; }

#endif  // SIMD_MAX_WIDTH_ >= 1

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
    return static_cast<u32>(f32(v));
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
    return static_cast<u64>(f64(v));
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

// Widest vector register this target offers, in bytes. Adding a wider backend
// (AVX2, SVE) means changing this one number, not a table of aliases.
#if SIMD_MAX_WIDTH_ == 1
inline constexpr usize plat_vector_bytes = 16;  // NEON q registers
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
