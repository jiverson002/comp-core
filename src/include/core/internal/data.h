// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_INTERNAL_DATA_H
#define COMP_CORE_INTERNAL_DATA_H 1

#include "core/internal/abi.h"
#include "core/traits.h"

//------------------------------------------------------------------------------
// internal data wrapper type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

//==============================================================================
//
//==============================================================================
template <class impl>
class data {
  private:
    using unitT = typename abi<impl>::unit_type;
    using baseT = typename abi<impl>::base_type;
    using maskT = typename abi<impl>::mask_type;

    static constexpr auto arity    = abi<impl>::arity;
    static constexpr auto mask_max = abi<impl>::mask_max;

    baseT a_;
    maskT k_;

  public:
    // Ctors
    data(baseT const &a) : a_(a), k_(mask_max) { }
    template<class T1 = unitT, typename T2 = baseT>
    data(unitT const &a, typename std::enable_if<!std::is_same<T1, T2>::value>::type* = nullptr) : a_(abi<impl>::set(a)), k_(mask_max) { }

    // Type conversion
    explicit operator baseT const&() const { return a_; }

    // Bitwise arithmetic operators
    auto operator<<=(int const &imm8) -> data<impl>&;

    // Mask operators
    auto operator[](maskT const &k) -> data<impl>&;
};

} // namespace internal
} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// internal data wrapper type functions.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

//==============================================================================
// Unary arithmetic operators
//==============================================================================
template <class dataT>
inline auto operator-(dataT const &a) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::neg(static_cast<base>(a));
}

//==============================================================================
// Binary arithmetic operators
//==============================================================================
template <class dataT>
inline auto operator+(dataT const &a, dataT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::add(static_cast<base>(a), static_cast<base>(b));
}

template <class dataT ,class unitT = typename data_traits<dataT>::unit_type>
inline auto operator+(dataT const &a, unitT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::add(static_cast<base>(a), abi::set(b));
}

template <class dataT>
inline auto operator-(dataT const &a, dataT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::sub(static_cast<base>(a), static_cast<base>(b));
}

template <class dataT>
inline auto operator*(dataT const &a, dataT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::mul(static_cast<base>(a), static_cast<base>(b));
}

template <class dataT>
inline auto operator/(dataT const &a, dataT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::div(static_cast<base>(a), static_cast<base>(b));
}

template <class dataT>
inline auto operator--(dataT &a, int) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::sub(static_cast<base>(a), abi::set(1)));
}

template <class dataT>
inline auto operator+=(dataT &a, dataT const &b) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::add(static_cast<base>(a), static_cast<base>(b)));
}

template <class dataT>
inline auto operator*=(dataT &a, dataT const &b) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::mul(static_cast<base>(a), static_cast<base>(b)));
}

template <class dataT>
inline auto operator/=(dataT &a, dataT const &b) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::div(static_cast<base>(a), static_cast<base>(b)));
}

//==============================================================================
// Bitwise arithmetic operators
//==============================================================================
template <class dataT>
inline auto operator^(dataT const &a, dataT const &b) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::eor(static_cast<base>(a), static_cast<base>(b));
}

template <class dataT>
inline auto operator>>(dataT const &a, int const &imm8) -> dataT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::bsr(static_cast<base>(a), imm8);
}

template <class dataT>
inline auto operator|=(dataT &a, dataT const &b) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::lor(static_cast<base>(a), static_cast<base>(b)));
}

template <class dataT>
inline auto operator>>=(dataT &a, int const &imm8) -> dataT& {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return (a = abi::bsr(static_cast<base>(a), imm8));
}

template <class impl>
inline auto data<impl>::operator<<=(int const &imm8) -> data<impl>& {
  if (mask_max == k_) {
    a_ = abi<impl>::bsl(a_, imm8);
  } else {
    a_ = abi<impl>::mbsl(k_, a_, imm8);
    k_ = mask_max;
  }
  return *this;
}

//==============================================================================
// Relational operators
//==============================================================================
template < class dataT
         , class maskT = typename data_traits<dataT>::mask_type >
inline auto operator>(dataT const &a, dataT const &b) -> maskT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::cmpgt(static_cast<base>(a), static_cast<base>(b));
}

template < class dataT
         , class unitT = typename data_traits<dataT>::unit_type
         , class maskT = typename data_traits<dataT>::mask_type >
inline auto operator<=(dataT const &a, unitT const &b) -> maskT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::cmple(static_cast<base>(a), abi::set(b));
}

template < class dataT
         , class maskT = typename data_traits<dataT>::mask_type >
inline auto operator==(dataT const &a, dataT const &b) -> maskT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::cmpeq(static_cast<base>(a), static_cast<base>(b));
}

template < class dataT
         , class maskT = typename data_traits<dataT>::mask_type >
inline auto operator!=(dataT const &a, dataT const &b) -> maskT {
  using abi  = typename data_traits<dataT>::abi;
  using base = typename data_traits<dataT>::base_type;
  return abi::cmpne(static_cast<base>(a), static_cast<base>(b));
}

//==============================================================================
// Mask operators
//==============================================================================
template <class impl>
inline auto data<impl>::operator[](maskT const &k) -> data<impl>& {
  k_ = k;
  return *this;
}

} // namespace internal
} // namespace core
} // namespace comp

#endif // COMP_CORE_INTERNAL_DATA_H
