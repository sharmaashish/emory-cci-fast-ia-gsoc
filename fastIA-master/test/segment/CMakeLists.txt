enable_testing()

set(EXECUTABLE_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH}/test_segment)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/out)

set(tests sample_test bwareopenTest deconvTest distTransformTest
    imfillTest imreconChunkTest watershedTest watershedNewTest watershedOpenCV #oclTest
    pixel-opsTest)


foreach(test ${tests})
	add_executable(${test}_segment ${test}.cpp)

	target_link_libraries(${test}_segment cci_common segment)
	target_link_libraries(${test}_segment boost_unit_test_framework)

	add_test(${test}_segment ${EXECUTABLE_OUTPUT_PATH}/${test}_segment.exe)
endforeach(test)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
#set(BUILD_SHARED_LIBS OFF)

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(globalQueueTest_segment
  ${segment_cuda_SOURCE_DIR}/global_queue.cuh
  ${segment_cuda_SOURCE_DIR}/global_queue.cu
  cuda/globalQueueTest.cu
  OPTIONS -gencode=arch=compute_20,code=sm_20
 )
target_link_libraries(globalQueueTest_segment boost_unit_test_framework)
