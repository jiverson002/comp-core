// SPDX-License-Identifier: MIT
#ifndef COMP_CORE_INTERNAL_SIMD_EMU_H
#define COMP_CORE_INTERNAL_SIMD_EMU_H 1

#define COMP_EMU_MARCH_knl 0

#define COMP_EMU_MARCH_VAL_impl_x(a) COMP_EMU_MARCH_ ## a
#define COMP_EMU_MARCH_VAL_impl(a)   COMP_EMU_MARCH_VAL_impl_x(a)
#define COMP_EMU_MARCH_VAL           COMP_EMU_MARCH_VAL_impl(COMP_EMU_MARCH)

#if COMP_EMU_MARCH_VAL == COMP_EMU_MARCH_knl
# include "core/internal/simd/emu/march/knl.h"
#endif

#endif // COMP_CORE_INTERNAL_SIMD_EMU_H
