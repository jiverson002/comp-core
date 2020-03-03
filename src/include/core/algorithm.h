// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_ALGORITHM_H
#define COMP_CORE_ALGORITHM_H 1

#include "traits.h"

//------------------------------------------------------------------------------
// algorithms on containers.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//==============================================================================
// Algorithms
//==============================================================================
template < template <class, class> class contT
         , class alocT
         , class dataT
         , class unitT = typename data_traits<dataT>::unit_type >
inline auto gather(contT<unitT, alocT> const &c, dataT const &vindex) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::gather(static_cast<base>(vindex), c.data());
}

template < template <class, class> class contT
         , class alocT
         , class dataT
         , class unitT = typename data_traits<dataT>::unit_type >
inline auto scatter(contT<unitT, alocT> &c, dataT const &vindex, dataT const &a) -> void {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  abi::scatter(c.data(), static_cast<base>(vindex), static_cast<base>(a));
}

} // namespace core
} // namespace comp

#endif // COMP_CORE_ALGORITHM_H
