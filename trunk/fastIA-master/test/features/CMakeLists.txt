enable_testing()

add_executable(sample_test2 sample_test.cpp)
target_link_libraries(sample_test2 cci_common segment boost_unit_test_framework)

add_test(SampleTest2 ${EXECUTABLE_OUTPUT_PATH}/sample_test2.exe)
