enable_testing()

set(EXECUTABLE_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH}/test_features)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/out)

set(tests sample_test)

foreach(test ${tests})
	add_executable(${test}_features ${test}.cpp)

	target_link_libraries(${test}_features cci_common featuresAll)
	target_link_libraries(${test}_features boost_unit_test_framework)

	add_test(${test}_features ${EXECUTABLE_OUTPUT_PATH}/${test}_features.exe)
endforeach(test)
