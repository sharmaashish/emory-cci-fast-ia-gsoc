project(segment_opencl)

set(cllibsrcs pixel-ops.cpp)

set(libname segment_opencl)

cuda_add_library("${libname}" SHARED ${cllibsrcs})
set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_OPENCL") 
