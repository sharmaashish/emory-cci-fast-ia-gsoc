#include "component_labeling.h"

#include "utils/ocl_utils.h"
#include <iostream>

#define UF_BLOCK_SIZE_X 32
#define UF_BLOCK_SIZE_Y 16

void ccl(cl::Buffer img, cl::Buffer labels,
         int width, int height,
         int bgval, int connectivity,
         ProgramCache &cache, cl::CommandQueue &queue)
{

    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DUF_BLOCK_SIZE_X=";
    params_stream << UF_BLOCK_SIZE_X;
    params_stream << " -DUF_BLOCK_SIZE_Y=";
    params_stream << UF_BLOCK_SIZE_Y;

    std::string program_params = params_stream.str();

    cl::Program& program = cache.getProgram("ComponentLabeling",
                                            program_params);

    cl::Kernel uf_local_kernel(program, "uf_local");
    cl::Kernel uf_global_kernel(program, "uf_global");
    cl::Kernel uf_final_kernel(program, "uf_final");

    size_t global_width = ((width + UF_BLOCK_SIZE_X - 1) / UF_BLOCK_SIZE_X)
                            * UF_BLOCK_SIZE_X;


    size_t global_height = ((height + UF_BLOCK_SIZE_Y - 1)/UF_BLOCK_SIZE_Y)
                            * UF_BLOCK_SIZE_Y;

    std::cout << "global width: " << global_width << std::endl;
    std::cout << "global height: " << global_height << std::endl;

    cl::NDRange global(global_width, global_height);
    cl::NDRange local(UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);

    cl::LocalSpaceArg s_img = cl::__local(UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y
                                          * sizeof(unsigned char));

    cl::LocalSpaceArg s_buffer = cl::__local(UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y
                                          * sizeof(int));

    uf_local_kernel.setArg(0, labels);
    uf_local_kernel.setArg(1, img);
    uf_local_kernel.setArg(2, s_buffer);
    uf_local_kernel.setArg(3, s_img);
    uf_local_kernel.setArg(4, width);
    uf_local_kernel.setArg(5, height);
    uf_local_kernel.setArg(6, connectivity);

    queue.enqueueNDRangeKernel(uf_local_kernel, cl::NullRange,
                               global, local);

    uf_global_kernel.setArg(0, labels);
    uf_global_kernel.setArg(1, img);
    uf_global_kernel.setArg(2, width);
    uf_global_kernel.setArg(3, height);
    uf_global_kernel.setArg(4, connectivity);

    queue.enqueueNDRangeKernel(uf_global_kernel, cl::NullRange,
                               global, local);

    uf_final_kernel.setArg(0, labels);
    uf_final_kernel.setArg(1, img);
    uf_final_kernel.setArg(2, width);
    uf_final_kernel.setArg(3, height);
    uf_final_kernel.setArg(4, bgval);

    queue.enqueueNDRangeKernel(uf_final_kernel, cl::NullRange,
                               global, local);

}
