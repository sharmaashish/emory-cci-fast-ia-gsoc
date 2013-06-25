#include "ocl_program_cache.h"

extern const char * HelloOpenCL;
extern const char * Invert;
extern const char * Threshold;
extern const char * Bgr2gray;
extern const char * Mask;
extern const char * Divide;
extern const char * Replace;

const char* getSourceByName(const std::string str_name){

    if(str_name == "HelloOpenCL")
        return HelloOpenCL;
    else if(str_name == "Invert")
        return Invert;
    else if(str_name == "Threshold")
        return Threshold;
    else if(str_name == "Bgr2gray")
        return Bgr2gray;
    else if(str_name == "Mask")
        return Mask;
    else if(str_name == "Divide")
        return Divide;
    else if(str_name == "Replace")
        return Replace;

    std::cout << "source for name: " << str_name << " not found!" << std::endl;

    static const char* empty = "";
    return empty;
}

ProgramCache::ProgramCache(cl::Context& context, cl::Device& device)
    : context(context) {

    devices.push_back(device);
}


cl::Program& ProgramCache::getProgram(const std::string name){
    std::map<std::string, cl::Program>::iterator it = programs.find(name);

    if(it == programs.end()){
        const char* source_str = getSourceByName(name);
        cl::Program::Sources source(1, std::make_pair(source_str, strlen(source_str)));
        programs[name] = cl::Program(context, source);

        std::cout << "building program: " << name << "... ";

        try{
            programs[name].build(devices);
        }catch(cl::Error ex)
        {
            std::cout << std::endl;
            std::cout << programs[name].getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            throw ex;
        }

        std::cout << "OK" << std::endl;
    }

    return programs[name];
}

cl::Kernel ProgramCache::getKernel(const std::string programName, const std::string kernelName)
{
    cl::Program& program = getProgram(programName);
    return cl::Kernel(program, kernelName.c_str());
}
