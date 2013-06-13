enable_testing()

add_executable(sample_test sample_test.cpp)
target_link_libraries(sample_test cci_common segment boost_unit_test_framework)

add_test(SampleTest ${EXECUTABLE_OUTPUT_PATH}/sample_test.exe)
