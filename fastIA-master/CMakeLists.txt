cmake_minimum_required(VERSION 2.8)

PROJECT( fastIA )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

message(STATUS ${CMAKE_MODULE_PATH})



# The version number
set(fastIA_VERSION_MAJOR 1 CACHE INTERNAL "major version")
set(fastIA_VERSION_MINOR 0 CACHE INTERNAL "minor version")

set(NS_FEATURES OFF CACHE BOOL "Building Features Extraction")
set(NS_SEGMENT OFF CACHE BOOL "Building Segmentation")

#message(STATUS "CMAKE binary dir location: ${CMAKE_BINARY_DIR}")
#message(STATUS "PROJECT binary dir location: ${PROJECT_BINARY_DIR}")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
    set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()


# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# Default: dynamic libraries
# ----------------------------------------------------------------------------
#set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")

# ----------------------------------------------------------------------------
# Include profiling?
# Default: NO
# ----------------------------------------------------------------------------
set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")

# Profiling?
if(ENABLE_PROFILING)
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
else()
    # Remove unreferenced functions: function level linking
    if(NOT APPLE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
    endif()
endif()

# ----------------------------------------------------------------------------
#                       common libaries needed by all
# ----------------------------------------------------------------------------


#### OpenMP
set(USE_OPENMP OFF CACHE BOOL "Build with OpenMP support")
if (USE_OPENMP)
    include(FindOpenMP)

	 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(USE_OPENMP)


#### for compression
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
	include_directories(${ZLIB_INCLUDE_DIRS})
endif(ZLIB_FOUND)


if(NS_FEATURES OR NS_SEGMENT)
    #### OpenCV
    find_path(OpenCV_CMAKE_DIR NAMES FindOpenCV.cmake OpenCVConfig.cmake
    	HINTS ${CMAKE_MODULE_PATH} ENV OPENCV_HOME
    	DOC "path to FindOpenCV.cmake or OpenCVConfig.cmake file") 
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${OpenCV_CMAKE_DIR})
    find_package( OpenCV REQUIRED )
    #message(STATUS "openCV lib = ${OpenCV_LIBS}")
    
    #### CUDA
    set(USE_CUDA OFF CACHE BOOL "Include NVidia Cuda Runtime support")
    if(USE_CUDA)
        find_package(CUDA 4.0)
        
        if(CUDA_FOUND) 		
            message(STATUS "CUDA detected: " ${CUDA_VERSION})
    
            # These vars will be processed in other scripts
         	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math -arch=sm_20 -DPREFIX_SUM)
         #	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -G)  # too slow...
           set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -DTHRUST_DEBUG)
            message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")
    	add_definitions( -DWITH_CUDA )
        endif(CUDA_FOUND)
    else(USE_CUDA)
    	set(CUDA_FOUND 0)
    endif(USE_CUDA)

    set(USE_OPENCL OFF CACHE BOOL "Include OpenCL support")
    if(USE_OPENCL)
	find_package(OpenCL)

        if(OPENCL_FOUND) 		
            message(STATUS "OpenCL detected")

	    message(STATUS "opencl library: " ${OPENCL_LIBRARY})
	    message(STATUS "opencl include dir: " ${OPENCL_INCLUDE_DIR})

            add_definitions(-D__CL_ENABLE_EXCEPTIONS)

            # OpenCL C++ bindings
            include_directories(BEFORE ${PROJECT_SOURCE_DIR}/external)

            # OPENCL_INCLUDE_DIR moved to front of include directories to prevent
            # using OpenCL headers from default system include dir
            include_directories(BEFORE ${OPENCL_INCLUDE_DIR})

	endif(OPENCL_FOUND)

    endif(USE_OPENCL)
    
endif(NS_FEATURES OR NS_SEGMENT)

# ----------------------------------------------------------------------------
#                       PROCESS SUBDIRECTORIES:
# ----------------------------------------------------------------------------
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

set(CMAKE_EXECUTABLE_SUFFIX ".exe")

# Check if the user want to build test applications
option (BUILD_TEST_APPLICATIONS "Inform whether test applications should be built" ON)
# Check if the user want to build sample applications
option (BUILD_EXAMPLE_APPLICATIONS "Inform whether sample applications should be built" ON)

add_subdirectory(src)

set(BUILD_UNIT_TESTS ON CACHE BOOL "Building unit tests")

if(BUILD_UNIT_TESTS)

    SET(TEST_DATA_PATH "${CMAKE_SOURCE_DIR}/../../test_data" CACHE STRING "Inform where the test data are located")

    if(EXISTS ${TEST_DATA_PATH})
	message(STATUS "test data path: ${TEST_DATA_PATH}")
    else()
	message(WARNING "test data path not found at: ${TEST_DATA_PATH}")
    endif()

    add_definitions(-DTEST_DATA_PATH="${TEST_DATA_PATH}")

    enable_testing()
    add_subdirectory(test)
endif(BUILD_UNIT_TESTS)
