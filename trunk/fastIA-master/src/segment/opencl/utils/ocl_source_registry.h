#ifndef OCL_SOURCE_REGISTRY_H
#define OCL_SOURCE_REGISTRY_H

#include <map>
#include <string>

class SourceRegistry
{
public:
    static SourceRegistry& getInstance();
    int registerSource(const std::string& programName, const char* source);
    const char* getSource(const std::string& programName) const;

private:
    SourceRegistry();
    std::map<std::string, const char*> registry;
};

#endif
