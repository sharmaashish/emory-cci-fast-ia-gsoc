enable_testing()

################################################
## GENERATING AND BUILDING SOURCE WITH OPENCL ##
################################################

# loading kernels
file(GLOB cl_kernels "opencl/kernels/*.cl")

set(ocl_kernels_file segment_ocl_test_kernels.cpp)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}"
    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}" -P "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake"
    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake_modules/cl2cpp.cmake")

add_library(segment_ocl_test_kernels SHARED ${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file} ${cl_kernels} opencl/staticInitializer.cpp)

####################
## BUILDING TESTS ##
####################

set(EXECUTABLE_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH}/test_segment)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/out)

set(tests sample_test bwareopenTest deconvTest distTransformTest
    imfillTest imreconChunkTest watershedTest watershedNewTest watershedOpenCV #oclTest
    pixel-opsTest parallelQueueTest morphReconInitTest morphReconPropagationTest morphReconTest morphReconSpeedupTest)

foreach(test ${tests})
        add_executable(${test}_segment ${test}.cpp)

        target_link_libraries(${test}_segment cci_common segment)
        target_link_libraries(${test}_segment boost_unit_test_framework)

        add_test(${test}_segment ${EXECUTABLE_OUTPUT_PATH}/${test}_segment.exe)
endforeach(test)

###############################
## BUILDING AND LINKING CUDA ##
###############################

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(globalQueueTest_segment
  ${segment_cuda_SOURCE_DIR}/global_queue.cuh
  ${segment_cuda_SOURCE_DIR}/global_queue.cu
  cuda/globalQueueTest.cu
  OPTIONS -DPREFIX_SUM -gencode=arch=compute_20,code=sm_20
 )
target_link_libraries(globalQueueTest_segment boost_unit_test_framework)


####################
## LINKING OPENCL ##
####################

target_link_libraries(parallelQueueTest_segment segment_ocl_test_kernels)
