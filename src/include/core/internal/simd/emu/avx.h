// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_INTERNAL_SIMD_EMU_AVX_H
#define COMP_CORE_INTERNAL_SIMD_EMU_AVX_H 1

#include <cstdint>

#define COMP_EMU_MARCH_AVX 1

typedef union {
  std::int64_t  i64[2];
  std::int32_t  i32[4];
  std::int16_t  i16[8];
  std::int8_t   i8[16];
  std::uint64_t u64[2];
  std::uint32_t u32[4];
  std::uint16_t u16[8];
  std::uint8_t  u8[16];
} __m128i;

typedef union {
  std::int64_t  i64[4];
  std::int32_t  i32[8];
  std::int16_t  i16[16];
  std::int8_t   i8[32];
  std::uint64_t u64[4];
  std::uint32_t u32[8];
  std::uint16_t u16[16];
  std::uint8_t  u8[32];
} __m256i;

typedef union {
  std::int64_t  i64[8];
  std::int32_t  i32[16];
  std::int16_t  i16[32];
  std::int8_t   i8[64];
  std::uint64_t u64[8];
  std::uint32_t u32[16];
  std::uint16_t u16[32];
  std::uint8_t  u8[64];
} __m512i;

using __int64  = std::int64_t;
using __mmask8 = std::uint8_t;

#if defined(__SSE2__)
inline __m128i _mm_loadu_si128(__m128i const *mem_addr) {
  return *mem_addr;
}

inline void _mm_storeu_si128(__m128i *mem_addr, __m128i a) {
  *mem_addr = a;
}
#endif

#if defined(__AVX512F__)
inline __m512i _mm512_set_epi64(__int64 e7, __int64 e6, __int64 e5, __int64 e4, __int64 e3, __int64 e2, __int64 e1, __int64 e0) {
  __m512i retval;
  retval.i64[0] = e0;
  retval.i64[1] = e1;
  retval.i64[2] = e2;
  retval.i64[3] = e3;
  retval.i64[4] = e4;
  retval.i64[5] = e5;
  retval.i64[6] = e6;
  retval.i64[7] = e7;
  return retval;
}

inline __m512i _mm512_cvtepu8_epi64(__m128i a) {
  __m512i dst;
  for (auto i = 0; i < 8; i++)
    dst.i64[i] = a.u8[i];
  return dst;
}

inline __m512i _mm512_maskz_cvtepu8_epi64(__mmask8 k, __m128i a) {
  __m512i dst;
  for (auto i = 0; i < 8; i++) {
    if (k & (1 << i))
      dst.i64[i] = a.u8[i];
    else
      dst.i64[i] = 0;
  }
  return dst;
}

inline void _mm512_mask_cvtepi64_storeu_epi8(void *base_addr, __mmask8 k, __m512i a) {
  auto *mem = static_cast<uint8_t*>(base_addr);
  for (auto i = 0; i < 8; i++) {
    if (k & (1 << i))
      mem[i] = a.i64[i]; // TODO: Truncate_Int64_To_Int8
  }
}

inline void _mm512_storeu_epi64(void *mem_addr, __m512i a) {
  // FIXME: mem_addr does not need to be aligned on any particular boundary.
  auto *mem = static_cast<__int64*>(mem_addr);
  for (auto i = 0; i < 8; i++)
    mem[i] = a.i64[i];
}

inline void _mm512_mask_storeu_epi64(void *mem_addr, __mmask8 k, __m512i a) {
  // FIXME: mem_addr does not need to be aligned on any particular boundary.
  auto *mem = static_cast<__int64*>(mem_addr);
  for (auto i = 0; i < 8; i++) {
    if (k & (1 << i))
      mem[i] = a.i64[i];
  }
}

inline __m512i _mm512_i64gather_epi64(__m512i vindex, void const *base_addr, int scale) {
  __m512i dst;
  auto const *mem = static_cast<char const*>(base_addr);
  for (auto i = 0; i < 8; i++)
    dst.i64[i] = *reinterpret_cast<__int64 const*>(mem + vindex.i64[i] * scale);
  return dst;
}

inline __m512i _mm512_mask_i64gather_epi64(__m512i src, __mmask8 k, __m512i vindex, void const *base_addr, int scale) {
  __m512i dst;
  auto const *mem = static_cast<char const*>(base_addr);
  for (auto i = 0; i < 8; i++) {
    if (k & (1 << i))
      dst.i64[i] = *reinterpret_cast<__int64 const*>(mem + vindex.i64[i] * scale);
    else
      dst.i64[i] = src.i64[i];
  }
  return dst;
}

inline void _mm512_i64scatter_epi64(void *base_addr, __m512i vindex, __m512i a, int scale) {
  // FIXME: base_addr does not need to be aligned on any particular boundary.
  auto *mem = static_cast<char*>(base_addr);
  for (auto i = 0; i < 8; i++) {
    *reinterpret_cast<__int64*>(mem + vindex.i64[i] * scale) = a.i64[i];
  }
}

inline void _mm512_mask_i64scatter_epi64(void *base_addr, __mmask8 k, __m512i vindex, __m512i a, int scale) {
  // FIXME: base_addr does not need to be aligned on any particular boundary.
  auto *mem = static_cast<char*>(base_addr);
  for (auto i = 0; i < 8; i++)
    if (k & (1 << i))
      *reinterpret_cast<__int64*>(mem + vindex.i64[i] * scale) = a.i64[i];
}

inline __m512i _mm512_add_epi64(__m512i a, __m512i b) {
  __m512i dst;
  for (auto i = 0; i < 8; i++)
    dst.i64[i] = a.i64[i] + b.i64[i];
  return dst;
}

inline __m512i _mm512_sub_epi64(__m512i a, __m512i b) {
  __m512i dst;
  for (auto i = 0; i < 8; i++)
    dst.i64[i] = a.i64[i] - b.i64[i];
  return dst;
}

inline __m512i _mm512_mullox_epi64(__m512i a, __m512i b) {
  __m512i dst;
  for (auto i = 0; i < 8; i++)
    dst.i64[i] = a.i64[i] * b.i64[i];
  return dst;
}

inline __mmask8 _mm512_cmpneq_epu64_mask(__m512i a, __m512i b) {
  __mmask8 k = 0;
  for (auto i = 0; i < 8; i++)
    k |= (a.i64[i] != b.i64[i] ? 1 : 0) << i;
  return k;
}

inline __mmask8 _mm512_cmpeq_epu64_mask(__m512i a, __m512i b) {
  __mmask8 k = 0;
  for (auto i = 0; i < 8; i++)
    k |= (a.i64[i] == b.i64[i] ? 1 : 0) << i;
  return k;
}
#endif

#endif // COMP_CORE_INTERNAL_SIMD_EMU_AVX_H
