#include "ocl_program_cache.h"

#include "ocl_utils.h"
#include "ocl_source_registry.h"

#define DEBUG_PRINT

ProgramCache::ProgramCache(cl::Context& context, cl::Device& device)
#ifdef OPENCL_PROFILE
    : context(context),
      defaultCommandQueue(cl::CommandQueue(context, device,
                                           CL_QUEUE_PROFILING_ENABLE))
#else
    : context(context),
      defaultCommandQueue(cl::CommandQueue(context, device))
#endif
{
    devices.push_back(device);
}

cl::Program& ProgramCache::getProgram(const std::string& name,
                                      const std::string& params)
{
    std::vector<std::string> names(1, name);
    return getProgram(names, params);
}

struct append_to{
    append_to(std::string& str) : str(str){}
    void operator()(const std::string arg) { str.append(arg); }
    std::string& str;
};

cl::Program& ProgramCache::getProgram(const std::vector<std::string>& names,
                                      const std::string& params){

    std::string key;

    std::for_each(names.begin(), names.end(), append_to(key));
    key += params;

    std::map<std::string, cl::Program>::iterator it = programs.find(key);

    if(it == programs.end()){

        cl::Program::Sources sources;
        std::string source_names;

        for(int i = 0; i < names.size(); ++i)
        {
            const char* source_str
                    = SourceRegistry::getInstance().getSource(names[i]);

            if(source_str == NULL)
            {
                std::cerr << "source name not found: "
                          << names[i] << std::endl;
                continue;
            }

            std::pair<const char*, size_t> pair
                    = std::make_pair(source_str, strlen(source_str));

            sources.push_back(pair);
            source_names += names[i] + ((i == names.size() - 1) ? "" : " ");
        }

        programs[key] = cl::Program(context, sources);

#ifdef DEBUG_PRINT
        std::cout << "building program from sources: "
                  << source_names << "... ";
#endif

        try
        {
            programs[key].build(devices, params.c_str());
        }
        catch(cl::Error ex)
        {
            std::cout << std::endl;
            std::cout << programs[key]
                         .getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
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
    std::cout << "initializing global opencl program cache"
              << std::endl;

    cl::Context context;
    std::vector<cl::Device> devices;

    oclSimpleInit(CL_DEVICE_TYPE_ALL, context, devices);

    cl::Device device = devices[0];

    std::cout << "devices count: "
              << devices.size() << std::endl;

    return ProgramCache(context, device);
}
