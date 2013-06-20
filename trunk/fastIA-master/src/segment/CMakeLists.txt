project( Segmentation )

# The version number
set(Segmentation_VERSION_MAJOR 1)
set(Segmentation_VERSION_MINOR 0)

# Source files used by Segmentation
set(libsrcs HistologicalEntities.cpp MorphologicOperations.cpp PixelOperations.cpp NeighborOperations.cpp ConnComponents.cpp )
set(gpulibsrcs HistologicalEntitiesGPU.cpp MorphologicOperationsGPU.cpp PixelOperationsGPU.cpp NeighborOperationsGPU.cpp UtilsCVImageIO.cpp)

include_directories(${cci_common_SOURCE_DIR})
link_directories(${cci_common_BINARY_DIR})

set(libname segment)

if(CUDA_FOUND)
	add_subdirectory(cuda)
	include_directories(${segment_cuda_SOURCE_DIR})
	link_directories(${segment_cuda_BINARY_DIR})	
    
	cuda_add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
    
	target_link_libraries(${libname} segment_cuda)
	set_target_properties(${libname} PROPERTIES COMPILE_DEFINITIONS "WITH_CUDA")
else(CUDA_FOUND)
	add_library("${libname}" SHARED ${libsrcs} ${gpulibsrcs})
endif(CUDA_FOUND)

# Link opencv stuff to the execution engine
target_link_libraries("${libname}" ${OpenCV_LIBS})

# if build test applications is set, than build them: currently just one
if(BUILD_TEST_APPLICATIONS)
	add_subdirectory("test/src")
endif(BUILD_TEST_APPLICATIONS)

