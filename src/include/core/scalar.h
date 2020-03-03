// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_SCALAR_H
#define COMP_CORE_SCALAR_H 1

#include "core/internal/abi.h"
#include "core/internal/data.h"
#include "core/internal/scalar.h"
#include "core/traits.h"

//------------------------------------------------------------------------------
// exported scalar type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! scalar type
using scalar = internal::data<internal::scalar>;

} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// data_traits specialization for exported scalar<> type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! Specialization for scalar type.
template <>
class data_traits<scalar> {
  public:
    using abi       = typename internal::abi<internal::scalar>;
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

#endif // COMP_CORE_SCALAR_H
