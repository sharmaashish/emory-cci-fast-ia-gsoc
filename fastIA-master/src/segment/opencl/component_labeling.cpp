#include "component_labeling.h"

#include "utils/ocl_utils.h"
#include <iostream>

void ccl(cl::Buffer src, cl::Buffer labels,
         int width, int height,
         int bgval, int connectivity,
         ProgramCache &cache, cl::CommandQueue &queue)
{

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DBLOCK_SIZE=";
    params_stream << BLOCK_SIZE;

    std::string program_params = params_stream.str();

    cl::Program& program = cache.getProgram("Watershed", program_params);

    cl::Kernel descent_kernel(program, "descent_kernel");
    cl::Kernel increment_kernel(program, "increment_kernel");
    cl::Kernel minima_kernel(program, "minima_kernel");
    cl::Kernel plateau_kernel(program, "plateau_kernel");
    cl::Kernel flood_kernel(program, "flood_kernel");

}
