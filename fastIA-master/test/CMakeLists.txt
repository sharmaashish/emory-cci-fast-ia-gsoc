enable_testing()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories(.)

if(USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif(USE_CUDA)

include_directories(${cci_common_SOURCE_DIR})
link_directories(${cci_common_BINARY_DIR})

if(NS_FEATURES)
    include_directories(${Features_SOURCE_DIR})
    include_directories(${Features_SOURCE_DIR}/cuda)
    link_directories(${Features_BINARY_DIR})
    add_subdirectory(features)
endif(NS_FEATURES)

if(NS_SEGMENT)
    include_directories(${Segmentation_SOURCE_DIR})
    include_directories(${Segmentation_SOURCE_DIR}/cuda)
    link_directories(${Segmentation_BINARY_DIR})
    add_subdirectory(segment)
endif(NS_SEGMENT)

