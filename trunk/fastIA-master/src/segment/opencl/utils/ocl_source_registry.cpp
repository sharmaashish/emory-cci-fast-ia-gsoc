#include "ocl_source_registry.h"

#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#include "iostream"
#endif

SourceRegistry::SourceRegistry()
{
}

SourceRegistry& SourceRegistry::getInstance()
{
    static SourceRegistry instance;
    return instance;
}

int SourceRegistry::registerSource(const std::string &programName,
                                    const char *source)
{
#ifdef DEBUG_PRINT
    std::cout << "registering source: " << programName << std::endl;
#endif

    registry[programName] = source;
    return 0;
}


const char* SourceRegistry::getSource(const std::string &programName) const
{
    std::map<std::string, const char*>::const_iterator it
            = registry.find(programName);

    return it != registry.end() ? it->second : NULL;
}
