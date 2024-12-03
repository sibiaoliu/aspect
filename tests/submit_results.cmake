## ---------------------------------------------------------------------
##
## Copyright (C) 2013 - 2024 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# Submit existing test results to CDash.
#
# Usage:
#   Invoke this script in a directory with test results already present
#   under ./Testing, i.e. valid ./Testing/TAG pointing to test results:
#
#   ctest -S ../tests/submit_results.cmake
#
# You may specify CTEST_SOURCE_DIRECTORY to point to a directory containing
# CTestConfig.cmake
#

if("${CTEST_SOURCE_DIRECTORY}" STREQUAL "")
  # Get parent directory:
  get_filename_component(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}" PATH)

  if(NOT EXISTS ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
    set(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()

message("-- CTEST_SOURCE_DIRECTORY: ${CTEST_SOURCE_DIRECTORY}")

set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

message("-- CTEST_BINARY_DIRECTORY: ${CTEST_BINARY_DIRECTORY}")

file(STRINGS ${CTEST_BINARY_DIRECTORY}/Testing/TAG _tag)
list(GET _tag 1 _track)

if("${_track}" STREQUAL "")
  message(FATAL_ERROR "
No test results found. Bailing out.
"
    )
endif()

ctest_start(Experimental TRACK ${_track} APPEND)
ctest_submit()
