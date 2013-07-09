#ifndef OCL_PROGRAM_CACHE_H
#define OCL_PROGRAM_CACHE_H

#include <string>
#include <iostream>
#include <map>

#include <CL/cl.hpp>

class ProgramCache
{
public:
    ProgramCache(cl::Context& context, cl::Device& device);
    cl::Program& getProgram(const std::string& name, const std::string& params = "");

    cl::Context getContext();
    cl::Device getDevice();
    cl::CommandQueue getDefaultCommandQueue();

    static ProgramCache& getGlobalInstance();

private:
    std::map<std::string, cl::Program> programs;
    cl::Context context;
    cl::CommandQueue defaultCommandQueue;

    std::vector<cl::Device> devices;

    static ProgramCache globalCacheInitialization();
};

#endif // OCL_PROGRAM_CACHE_H
