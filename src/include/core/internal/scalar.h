// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_INTERNAL_SCALAR_H
#define COMP_CORE_INTERNAL_SCALAR_H 1

#include <climits>
#include <limits>

#include "core/internal/abi.h"

//------------------------------------------------------------------------------
// internal::scalar type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

class scalar {
  public:
    using unitT = std::uint64_t;
    using baseT = unitT;
    using maskT = bool;
};

} // namespace internal
} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// abi specialization for internal::scalar type.
//------------------------------------------------------------------------------
namespace comp {
namespace core {
namespace internal {

template <>
class abi<scalar> {
  private:
    using unitT = typename scalar::unitT;
    using baseT = typename scalar::baseT;
    using maskT = typename scalar::maskT;

  public:
    using unit_type = unitT;
    using base_type = baseT;
    using mask_type = maskT;

    static constexpr auto arity = 1;

    static constexpr auto unit_max = std::numeric_limits<unitT>::max();
    static constexpr auto mask_max = true;

    static constexpr auto unit_width = static_cast<int>(sizeof(unitT) * CHAR_BIT);

    static auto set(unitT const &a) -> baseT {
      return a;
    }
    static auto get(void const *a) -> baseT {
      return *reinterpret_cast<unsigned char const*>(a);
    }
    static auto mget(maskT const &k, void const *a) -> baseT {
      return k ? *reinterpret_cast<unsigned char const*>(a) : 0;
    }

    static auto put(void *base_addr, baseT const &a) -> void {
      *static_cast<unsigned char*>(base_addr) = a;
    }
    static auto mput(void *base_addr, maskT const &k, baseT const &a) -> void {
      if (k)
        *static_cast<unsigned char*>(base_addr) = a;
    }

    static auto cpy(void *base_addr, baseT const &a) -> void {
      *static_cast<baseT*>(base_addr) = a;
    }

    static auto gather(baseT const &vindex, void const *base_addr) -> baseT {
      return static_cast<baseT const*>(base_addr)[vindex];
    }
    static auto scatter(void *base_addr, baseT const &vindex, baseT const &a) -> void {
      static_cast<baseT*>(base_addr)[vindex] = a;
    }

    static auto neg(baseT const &a) -> baseT { return -a; }
    static auto add(baseT const &a, baseT const & b) -> baseT { return a + b; }
    static auto mul(baseT const &a, baseT const & b) -> baseT { return a * b; }
    static auto sub(baseT const &a, baseT const & b) -> baseT { return a - b; }
    static auto div(baseT const &a, baseT const & b) -> baseT { return a / b; }
    static auto lor(baseT const &a, baseT const & b) -> baseT { return a | b; }
    static auto eor(baseT const &a, baseT const & b) -> baseT { return a ^ b; }

    static auto bsl(baseT const &a, unsigned int const &imm8) -> baseT { return a << imm8; }
    static auto bsr(baseT const &a, unsigned int const &imm8) -> baseT { return a >> imm8; }

    static auto mbsl(maskT const& k, baseT const &a, unsigned int const &imm8) -> baseT {
      return a << (k ? imm8 : 0);
    }

    static auto mcnt(maskT const &k) -> int { return k ? 1 : 0; }

    static auto cmpgt(baseT const &a, baseT const &b) -> maskT { return a > b; }
    static auto cmple(baseT const &a, baseT const &b) -> maskT { return a <= b; }
    static auto cmpeq(baseT const &a, baseT const &b) -> maskT { return a == b; }
    static auto cmpne(baseT const &a, baseT const &b) -> maskT { return a != b; }
};

} // namespace internal
} // namespace core
} // namespace comp

#endif // COMP_CORE_INTERNAL_SCALAR_H
