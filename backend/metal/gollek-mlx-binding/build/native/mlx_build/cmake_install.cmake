# Install script for directory: /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/_deps/json-build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/jaccl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/mlx/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/tests/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/examples/cpp/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/libmlx.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/mlx" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/backend\\/metal\\/kernels\\.h$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "metal_cpp_source" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/metal_cpp" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/_deps/metal_cpp-src/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake"
         "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES
    "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/MLXConfig.cmake"
    "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/MLXConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/3rdparty/mlx/cmake/")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/backend/metal/gollek-mlx-binding/build/native/mlx_build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
