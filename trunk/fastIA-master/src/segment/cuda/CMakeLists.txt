project(segment_cuda)

# Getting all *.cu files from cuda directory automatically
FILE(GLOB cudalibsrcs "*.cu")
FILE(GLOB cudabanlist global_sync.cu watershed-cilamce2009.cu watershed-dw-korbes-orig.cu)
LIST(REMOVE_ITEM cudalibsrcs ${cudabanlist})

FILE(GLOB cudalibheaders "*.cuh")
#set(cudalibheaders cuda/ccl_uf.cuh cuda/change_kernel.cuh cuda/global_queue_dist.cuh cuda/imrecon_queue_int_kernel.cuh cuda/imreconstruct_binary_kernel.cuh cuda/imreconstruct_float_kernel.cuh cuda/imreconstruct_int_kernel.cuh cuda/neighbor-ops.cuh cuda/pixel-ops.cuh cuda/queue.cuh cuda/textures.cuh cuda/watershed-ca-korbes.cuh cuda/watershed-dw-korbes.cuh)
#include_directories(${Segmentation_SOURCE_DIR}/cuda)

set(libname segment_cuda)

cuda_add_library("${libname}" SHARED ${cudalibsrcs} ${cudalibheaders})
set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_CUDA") 
