#include "ocl_program_cache.h"

#include "ocl_utils.h"

extern const char * HelloOpenCL;
extern const char * Invert;
extern const char * Threshold;
extern const char * Bgr2gray;
extern const char * Mask;
extern const char * Divide;
extern const char * Replace;
extern const char * Watershed;

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
    else if(str_name == "Watershed")
        return Watershed;

    std::cout << "source for name: " << str_name << " not found!" << std::endl;

    static const char* empty = "";
    return empty;
}

ProgramCache::ProgramCache(cl::Context& context, cl::Device& device)
#ifdef OPENCL_PROFILE
    : context(context), defaultCommandQueue(cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE))
#else
    : context(context), defaultCommandQueue(cl::CommandQueue(context, device))
#endif
{


    devices.push_back(device);
}


cl::Program& ProgramCache::getProgram(const std::string& name, const std::string& params){

    std::string key = name + params;

    std::map<std::string, cl::Program>::iterator it = programs.find(key);

    if(it == programs.end()){
        const char* source_str = getSourceByName(name);
        cl::Program::Sources source(1, std::make_pair(source_str, strlen(source_str)));
        programs[key] = cl::Program(context, source);

        std::cout << "building program: " << name << "... ";

        try
        {
            programs[key].build(devices, params.c_str());
        }
        catch(cl::Error ex)
        {
            std::cout << std::endl;
            std::cout << programs[key].getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            throw ex;
        }

        std::cout << "OK" << std::endl;
    }

    return programs[key];
}

cl::Context& ProgramCache::getContext()
{
    return context;
}

cl::Device& ProgramCache::getDevice()
{
    return devices.front();
}

cl::CommandQueue& ProgramCache::getDefaultCommandQueue()
{
    return defaultCommandQueue;
}

ProgramCache& ProgramCache::getGlobalInstance()
{
    static ProgramCache globalCache = globalCacheInitialization();
    return globalCache;
}

ProgramCache ProgramCache::globalCacheInitialization()
{
    std::cout << "initializing global opencl program cache" << std::endl;

    cl::Context context;
    std::vector<cl::Device> devices;

    oclSimpleInit(CL_DEVICE_TYPE_ALL, context, devices);

    cl::Device device = devices[0];
    std::cout << "devices count: " << devices.size() << std::endl;

    return ProgramCache(context, device);
}
