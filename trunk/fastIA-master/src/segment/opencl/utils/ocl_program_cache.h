#ifndef OCL_PROGRAM_CACHE_H
#define OCL_PROGRAM_CACHE_H

#include <string>
#include <iostream>
#include <map>

#include <CL/cl.hpp>

class ProgramCache
{
public:
    ProgramCache(cl::Context& context, cl::Device& devices);
    cl::Program& getProgram(const std::string programName);
    cl::Kernel getKernel(const std::string programName, const std::string kernelName);

private:
    std::map<std::string, cl::Program> programs;
    cl::Context& context;
    std::vector<cl::Device> devices;
};

#endif // OCL_PROGRAM_CACHE_H
