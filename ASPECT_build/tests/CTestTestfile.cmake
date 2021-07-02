# CMake generated Testfile for 
# Source directory: /Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/tests
# Build directory: /Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/ASPECT_build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(quick_mpi "/Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake" "-DBINARY_DIR=/Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/ASPECT_build/tests" "-DTESTNAME=tests.quick_mpi" "-DERROR=\"Test quick_mpi failed\"" "-P" "/Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/tests/run_test.cmake")
set_tests_properties(quick_mpi PROPERTIES  TIMEOUT "600" WORKING_DIRECTORY "/Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/ASPECT_build/tests" _BACKTRACE_TRIPLES "/Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/tests/CMakeLists.txt;389;ADD_TEST;/Users/sliu/GoogleDrive/1_Research/Aspect/aspect_July21/tests/CMakeLists.txt;0;")
