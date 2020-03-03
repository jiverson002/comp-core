// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_SIMD_AVX_H
#define COMP_CORE_SIMD_AVX_H 1

#include "core/internal/abi.h"
#include "core/internal/data.h"
#include "core/internal/simd/avx.h"

//------------------------------------------------------------------------------
// exported simd<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! simd type
template <int arity>
using simd = internal::data<internal::simd::avx<arity>>;

} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// data_traits specialization for exported simd<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! Specialization for simd type.
template <int ARITY>
class data_traits<simd<ARITY>> {
  public:
    using abi       = typename internal::abi<internal::simd::avx<ARITY>>;
    using unit_type = typename abi::unit_type;
    using base_type = typename abi::base_type;
    using mask_type = typename abi::mask_type;

    static constexpr auto arity      = abi::arity;
    static constexpr auto unit_max   = abi::unit_max;
    static constexpr auto mask_max   = abi::mask_max;
    static constexpr auto unit_width = abi::unit_width;
};

} // namespace core
} // namespace comp

#endif // COMP_CORE_SIMD_AVX_H
