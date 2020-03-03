// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_INTERNAL_SIMD_AVX_H
#define COMP_CORE_INTERNAL_SIMD_AVX_H 1

#include <climits>
#include <cstdint>
#include <limits>

#ifndef COMP_EMU_MARCH_AVX
# include <immintrin.h>
#endif

#define HAS_SIMD_INSTRUCTIONS

//------------------------------------------------------------------------------
// determine the capabilities of the host.
//------------------------------------------------------------------------------
#if defined(__AVX512DQ__)
# define pp_qword 1
#else
# define pp_qword 0
#endif
#if defined(__AVX512VL__)
# define pp_vlext 1
#else
# define pp_vlext 0
#endif
#if defined(__AVX512_VBMI2__)
# define pp_vbmi2 1
#else
# define pp_vbmi2 0
#endif

//------------------------------------------------------------------------------
// internal::simd::avx<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {
namespace simd {

template <int arity>
class avx;

template <>
class avx<4> {
  public:
    // AVX-512 ops won't work w/ YMM (256-bit) registers, so use ZMM
    // (512-bit) registers, but only the lower half.
#if pp_qword && pp_vlext
    using baseT = __m256i;
#else
    using baseT = __m512i;
#endif

    using maskT = __mmask8;
    using unitT = std::uint64_t;

    static constexpr maskT mask_max = 0xf;
};

template <>
class avx<8> {
  public:
    using baseT = __m512i;
    using maskT = __mmask8;
    using unitT = std::uint64_t;

    static constexpr maskT mask_max = 0xff;
};

} // namespace simd
} // namespace internal
} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// abi specialization for internal::simd::avx<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

template <int ARITY>
class abi<simd::avx<ARITY>> {
  private:
    using unitT = typename simd::avx<ARITY>::unitT;
    using baseT = typename simd::avx<ARITY>::baseT;
    using maskT = typename simd::avx<ARITY>::maskT;

  public:
    using unit_type = unitT;
    using base_type = baseT;
    using mask_type = maskT;

    static constexpr auto arity = ARITY;

    static constexpr auto unit_max = std::numeric_limits<unitT>::max();
    static constexpr auto mask_max = simd::avx<ARITY>::mask_max;

    static constexpr auto unit_width = static_cast<int>(sizeof(unitT) * CHAR_BIT);

    static auto set(const unitT &a) -> baseT;
    static auto get(void const *a) -> baseT;
    static auto put(void *base_addr, baseT const& a) -> void;
    static auto cpy(void *base_addr, baseT const& a) -> void;

    static auto mget(maskT const &k, void const *mem_addr) -> baseT;
    static auto mput(void *base_addr, maskT const &k, baseT const &a) -> void;

    static auto gather(baseT const &vindex, void const *base_addr) -> baseT;
    static auto scatter(void *base_addr, baseT const &vindex, baseT const &a) -> void;

    static auto neg(baseT const &a) -> baseT;

    static auto add(baseT const &a, baseT const &b) -> baseT;
    static auto sub(baseT const &a, baseT const &b) -> baseT;
    static auto mul(baseT const &a, baseT const &b) -> baseT;
    static auto div(baseT const &a, baseT const &b) -> baseT;

    static auto bsr(baseT const &a, unsigned int const &imm8) -> baseT;
    static auto bsl(baseT const &a, unsigned int const &imm8) -> baseT;

    static auto mbsl(maskT const &k, baseT const &a, unsigned int const &imm8) -> baseT;

    static auto lor(baseT const &a, baseT const &b) -> baseT;
    static auto eor(baseT const &a, baseT const &b) -> baseT;

    static auto mcnt(maskT const &k) -> int;

    static auto cmpgt(baseT const &a, baseT const &b) -> maskT;
    static auto cmple(baseT const &a, baseT const &b) -> maskT;
    static auto cmpeq(baseT const &a, baseT const &b) -> maskT;
    static auto cmpne(baseT const &a, baseT const &b) -> maskT;
};

} // namespace internal
} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// abi implementation for internal::simd::avx<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

template <int arity>
inline auto abi<simd::avx<arity>>::set(unitT const& a) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_set_epi64x(a, a, a, a);
#   else
      return _mm512_set_epi64(0, 0, 0, 0, a, a, a, a);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_set_epi64(a, a, a, a, a, a, a, a);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::get(void const *mem_addr) -> baseT {
# if pp_vbmi2
    if constexpr (4 == arity) {
#     if pp_qword && pp_vlext
        return _mm256_cvtepu8_epi64(_mm_maskz_expandloadu_epi8(mask_max, mem_addr));
#     elif pp_vlext
        return _mm512_cvtepu8_epi64(_mm_maskz_expandloadu_epi8(mask_max, mem_addr));
#     else
        return _mm512_cvtepu8_epi64(
          _mm512_castsi512_si128(_mm512_maskz_expandloadu_epi8(mask_max, mem_addr)));
#     endif
    } else if constexpr (8 == arity) {
#     if pp_vlext
        return _mm512_cvtepu8_epi64(_mm_maskz_expandloadu_epi8(mask_max, mem_addr));
#     else
        return _mm512_cvtepu8_epi64(
          _mm512_castsi512_si128(_mm512_maskz_expandloadu_epi8(mask_max, mem_addr)));
#     endif
    }
# endif

  unsigned char ap[16] = { 0 };
  auto const *m = static_cast<unsigned char const*>(mem_addr);
  for (int j = 0; j < arity; j++)
    ap[j] = m[j];
  auto const b = _mm_loadu_si128(reinterpret_cast<__m128i*>(ap));

  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cvtepu8_epi64(b);
#   else
      return _mm512_maskz_cvtepu8_epi64(mask_max, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cvtepu8_epi64(b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::put(void *base_addr, baseT const& a) -> void {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      _mm256_cvtepi64_storeu_epi8(base_addr, a);
#   else
      _mm512_mask_cvtepi64_storeu_epi8(base_addr, mask_max, a);
#   endif
  } else if constexpr (8 == arity) {
    _mm512_cvtepi64_storeu_epi8(base_addr, a);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::cpy(void *base_addr, baseT const& a) -> void {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      _mm256_storeu_epi64(base_addr, a);
#   else
      _mm512_mask_storeu_epi64(base_addr, mask_max, a);
#   endif
  } else if constexpr (8 == arity) {
    _mm512_storeu_epi64(base_addr, a);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::mget(maskT const &k, void const *mem_addr) -> baseT {
# if pp_vbmi2
    if constexpr (4 == arity) {
#     if pp_qword && pp_vlext
        return _mm256_cvtepu8_epi64(_mm_maskz_expandloadu_epi8(k, mem_addr));
#     else
        return _mm512_cvtepu8_epi64(
          _mm512_castsi512_si128(_mm512_maskz_expandloadu_epi8(k, mem_addr)));
#     endif
    } else if constexpr (8 == arity) {
#     if pp_vlext
        return _mm512_cvtepu8_epi64(_mm_maskz_expandloadu_epi8(k, mem_addr));
#     else
        return _mm512_cvtepu8_epi64(
          _mm512_castsi512_si128(_mm512_maskz_expandloadu_epi8(k, mem_addr)));
#     endif
    }
# endif

  unsigned char ap[16] = { 0 };
  auto const *m = static_cast<unsigned char const*>(mem_addr);
  for (int j = 0; j < arity; j++)
    if (k & (1 << j))
      ap[j] = *m++;
  auto const b = _mm_loadu_si128(reinterpret_cast<__m128i*>(ap));

  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cvtepu8_epi64(b);
#   else
      return _mm512_maskz_cvtepu8_epi64(mask_max, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cvtepu8_epi64(b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::mput(void *base_addr, maskT const &k, baseT const &a) -> void {
# if pp_vbmi2
    if constexpr (4 == arity) {
#     if pp_qword && pp_vlext
        _mm_mask_compressstoreu_epi8(base_addr, k, _mm256_cvtepi64_epi8(a));
#     else
        _mm512_mask_compressstoreu_epi8(base_addr, k,
            _mm512_castsi128_si512(_mm512_cvtepi64_epi8(a)));
#     endif
    } else if constexpr (8 == arity) {
#     if pp_vlext
        _mm_mask_compressstoreu_epi8(base_addr, k, _mm512_cvtepi64_epi8(a));
#     else
        _mm512_mask_compressstoreu_epi8(base_addr, k,
            _mm512_castsi128_si512(_mm512_cvtepi64_epi8(a)));
#     endif
    }
# else
    __m128i b;
    unsigned char ap[16];
    auto *m = static_cast<unsigned char*>(base_addr);

    for (int j = 0; j < arity; j++)
      if (k & (1 << j))
        *m++ = ap[j];

    if constexpr (4 == arity) {
#     if pp_qword && pp_vlext
        b = _mm256_cvtepi64_epi8(a);
#     else
        b = _mm512_maskz_cvtepi64_epi8(mask_max, a);
#     endif
    } else if constexpr (8 == arity) {
#     if pp_vlext
        b = _mm512_cvtepi64_epi8(a);
#     else
        b = _mm512_maskz_cvtepi64_epi8(mask_max, a);
#     endif
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ap), b);
# endif
}

template <int arity>
inline auto abi<simd::avx<arity>>::gather(baseT const &vindex, void const *base_addr) -> baseT {
  auto const zero = set(0);
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_i64gather_epi64(base_addr, vindex, 8);
#   else
      return _mm512_mask_i64gather_epi64(zero, mask_max, vindex, base_addr, 8);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_i64gather_epi64(vindex, base_addr, 8);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::scatter(void *base_addr, baseT const &vindex, baseT const &a) -> void {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      _mm256_i64scatter_epi64(base_addr, vindex, a, 8);
#   else
      _mm512_mask_i64scatter_epi64(base_addr, mask_max, vindex, a, 8);
#   endif
  } else if constexpr (8 == arity) {
    _mm512_i64scatter_epi64(base_addr, vindex, a, 8);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::neg(baseT const &a) -> baseT {
  return sub(set(0), a);
}

template <int arity>
inline auto abi<simd::avx<arity>>::add(baseT const &a, baseT const &b) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_add_epi64(a, b);
#   else
      return _mm512_add_epi64(a, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_add_epi64(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::sub(baseT const &a, baseT const &b) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_sub_epi64(a, b);
#   else
      return _mm512_sub_epi64(a, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_sub_epi64(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::mul(baseT const &a, baseT const &b) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_mullo_epi64(a, b);
#   else
      return _mm512_mullox_epi64(a, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_mullox_epi64(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::div(baseT const &a, baseT const &b) -> baseT {
  alignas(64) unitT p[arity];
  alignas(64) unitT q[arity];
  alignas(64) unitT r[arity];

  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      _mm256_store_epi64(reinterpret_cast<__m256i*>(p), a);
      _mm256_store_epi64(reinterpret_cast<__m256i*>(q), b);
#   else
      _mm512_mask_store_epi64(reinterpret_cast<__m512i*>(p), mask_max, a);
      _mm512_mask_store_epi64(reinterpret_cast<__m512i*>(q), mask_max, b);
#   endif
  } else if constexpr (8 == arity) {
    _mm512_store_epi64(reinterpret_cast<__m512i*>(p), a);
    _mm512_store_epi64(reinterpret_cast<__m512i*>(q), b);
  }

  for (auto i = 0; i < arity; i++)
    r[i] = p[i] / q[i];

  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_load_epi64(r);
#   else
      return _mm512_maskz_load_epi64(mask_max, r);
#   endif
  } else {
    return _mm512_load_epi64(r);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::bsr(baseT const &a, unsigned int const &imm8) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_srli_epi64(a, imm8);
#   else
      return _mm512_srli_epi64(a, imm8);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_srli_epi64(a, imm8);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::bsl(baseT const &a, unsigned int const &imm8) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_slli_epi64(a, imm8);
#   else
      return _mm512_slli_epi64(a, imm8);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_slli_epi64(a, imm8);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::mbsl(maskT const &k, baseT const &a, unsigned int const &imm8) -> baseT {
  if constexpr (4 == arity) {
#    if pp_qword && pp_vlext
       return _mm256_mask_slli_epi64(a, k, a, imm8);
#    else
       return _mm512_mask_slli_epi64(a, k, a, imm8);
#    endif
  } else if constexpr (8 == arity) {
    return _mm512_mask_slli_epi64(a, k, a, imm8);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::lor(baseT const &a, baseT const &b) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_or_epi64(a, b);
#   else
      return _mm512_or_epi64(a, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_or_epi64(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::eor(baseT const &a, baseT const &b) -> baseT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_xor_epi64(a, b);
#   else
      return _mm512_xor_epi64(a, b);
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_xor_epi64(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::mcnt(maskT const &k) -> int {
  return __builtin_popcount(_cvtmask8_u32(k));
}

template <int arity>
inline auto abi<simd::avx<arity>>::cmpgt(baseT const& a, baseT const& b) -> maskT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cmpgt_epu64_mask(a, b);
#   else
      return _mm512_cmpgt_epu64_mask(a, b) & mask_max;
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cmpgt_epu64_mask(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::cmple(baseT const& a, baseT const& b) -> maskT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cmple_epu64_mask(a, b);
#   else
      return _mm512_cmple_epu64_mask(a, b) & mask_max;
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cmple_epu64_mask(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::cmpeq(baseT const& a, baseT const& b) -> maskT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cmpeq_epu64_mask(a, b);
#   else
      return _mm512_cmpeq_epu64_mask(a, b) & mask_max;
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cmpeq_epu64_mask(a, b);
  }
}

template <int arity>
inline auto abi<simd::avx<arity>>::cmpne(baseT const& a, baseT const& b) -> maskT {
  if constexpr (4 == arity) {
#   if pp_qword && pp_vlext
      return _mm256_cmpneq_epu64_mask(a, b);
#   else
      return _mm512_cmpneq_epu64_mask(a, b) & mask_max;
#   endif
  } else if constexpr (8 == arity) {
    return _mm512_cmpneq_epu64_mask(a, b);
  }
}

} // namespace internal
} // namespace core
} // namespace comp

#undef pp_qword
#undef pp_vlext
#undef pp_vbmi2

#endif // COMP_CORE_INTERNAL_SIMD_AVX_H
