# CMake generated Testfile for 
# Source directory: /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/tests
# Build directory: /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/tests/tests_include-b858cb2.cmake")
add_test([=[tests]=] "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/tests/tests")
set_tests_properties([=[tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/tests/CMakeLists.txt;44;add_test;/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/tests/CMakeLists.txt;0;")
add_test([=[teardown]=] "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/tests/test_teardown")
set_tests_properties([=[teardown]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/tests/CMakeLists.txt;52;add_test;/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/tests/CMakeLists.txt;0;")
subdirs("../../_deps/doctest-build")
