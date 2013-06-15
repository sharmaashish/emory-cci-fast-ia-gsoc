enable_testing()

set(EXECUTABLE_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH}/test_segment)
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/out)

set(tests sample_test bwareopenTest)

foreach(test ${tests})
	add_executable(${test}_segment ${test}.cpp)

	target_link_libraries(${test}_segment cci_common segment)
	target_link_libraries(${test}_segment boost_unit_test_framework)

	add_test(${test}_segment ${EXECUTABLE_OUTPUT_PATH}/${test}_segment.exe)
endforeach(test)
