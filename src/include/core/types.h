// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_TYPES_H
#define COMP_CORE_TYPES_H 1

#include <cstdint>

//------------------------------------------------------------------------------
// Capability agnostic types.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! byte type
using byte = std::uint8_t;
//! symbol type
using symb = std::uint32_t;

} // namespace core
} // namespace comp

//------------------------------------------------------------------------------
// exported scalar type.
//------------------------------------------------------------------------------
#include "scalar.h"

//------------------------------------------------------------------------------
// enable architecture emulation -- if requested.
//------------------------------------------------------------------------------
#if defined(COMP_EMU_MARCH)
# include "core/internal/simd/emu.h"
#endif

//------------------------------------------------------------------------------
// exported simd type -- if available.
//------------------------------------------------------------------------------
#if defined(__AVX512F__)
# include "core/simd/avx.h"
#endif

//------------------------------------------------------------------------------
// convenience types.
//------------------------------------------------------------------------------
namespace comp {
namespace core {

//! frequency type
using freq = std::uint64_t;

} // namespace cors
} // namespace comp

#endif // COMP_CORE_TYPES_H
