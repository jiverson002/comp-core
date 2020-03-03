cmake_minimum_required(VERSION 3.10)

project(comp/core LANGUAGES CXX)

#-------------------------------------------------------------------------------
# INTERFACE configuration -- this is the genesis of the comp library target
#-------------------------------------------------------------------------------
add_library(comp INTERFACE)

target_include_directories(comp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
)

#-------------------------------------------------------------------------------
# COMP_EMU_MARCH -- Allow an architecture emulator to be requested
#-------------------------------------------------------------------------------
set(COMP_EMU_MARCH "" CACHE STRING "Architecture to emulate, as if -march=XXX were used")
if (NOT "${COMP_EMU_MARCH}" STREQUAL "")
  string(TOLOWER ${COMP_EMU_MARCH} lcCOMP_EMU_MARCH)

  message(STATUS "Emulating architecture: ${lcCOMP_EMU_MARCH}")

  target_compile_options(comp INTERFACE -DCOMP_EMU_MARCH=${lcCOMP_EMU_MARCH})

  unset(lcCOMP_EMU_MARCH)
endif ()
