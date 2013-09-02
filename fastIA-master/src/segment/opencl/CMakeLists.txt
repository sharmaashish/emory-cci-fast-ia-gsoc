project(segment_opencl)

# loading kernels
file(GLOB cl_kernels "kernels/*.cl")

include_directories(..) # to make opencl/utils/ocl_source_registry.h visible for file with ocl kernels

set(ocl_kernels_file segment_ocl_kernels.cpp)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}"
    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}" -P "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake"
    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake")

set(cllibsrcs
    utils/ocl_utils.cpp
    utils/ocl_program_cache.cpp
    utils/ocl_source_registry.cpp
    utils/ocl_type_resolver.cpp
    pixel-ops.cpp
    watershed.cpp
    parallel_queue.cpp
    morph_recon.cpp
    )

list(APPEND cllibsrcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}")

set(libname segment_opencl)

add_library("${libname}" SHARED ${cllibsrcs})
set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_OPENCL") 
target_link_libraries(${libname} ${OPENCL_LIBRARY})

#add_executable(pixel-ops pixel-ops.cpp)
#target_link_libraries(pixel-ops segment_opencl)
