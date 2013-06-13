add_subdirectory(common)

if(NS_FEATURES)
    add_subdirectory(features)
endif(NS_FEATURES)
    

if(NS_SEGMENT)
    add_subdirectory(segment)
endif(NS_SEGMENT)


