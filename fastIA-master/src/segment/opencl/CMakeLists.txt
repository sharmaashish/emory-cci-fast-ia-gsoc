project(segment_opencl)

# loading kernels
file(GLOB cl_kernels "kernels/*.cl")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp"
    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp" -P "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake"
    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake")

set(cllibsrcs utils/ocl_utils.cpp utils/ocl_program_cache pixel-ops.cpp)

list(APPEND cllibsrcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp")

set(libname segment_opencl)

add_library("${libname}" SHARED ${cllibsrcs})
set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_OPENCL") 
target_link_libraries(${libname} ${OPENCL_LIBRARY})

#add_executable(pixel-ops pixel-ops.cpp)
#target_link_libraries(pixel-ops segment_opencl)
