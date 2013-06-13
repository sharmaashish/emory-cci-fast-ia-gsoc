enable_testing()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)


if (NS_FEATURES)
    include_directories(../src/features)
    add_subdirectory(features)
endif(NS_FEATURES)


if (NS_SEGMENT)
    include_directories(../src/segment)
    add_subdirectory(segment)
endif(NS_SEGMENT)

