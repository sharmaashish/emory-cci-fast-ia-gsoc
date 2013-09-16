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



int relabel(cl::Buffer labels, int width, int height, int bgval,
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

    cl::Kernel relabel_first_kernel(program, "relabel_first");
    cl::Kernel relabel_second_kernel(program, "relabel_second");

    size_t global_width = ((width + UF_BLOCK_SIZE_X - 1) / UF_BLOCK_SIZE_X)
                            * UF_BLOCK_SIZE_X;

    size_t global_height = ((height + UF_BLOCK_SIZE_Y - 1)/UF_BLOCK_SIZE_Y)
                            * UF_BLOCK_SIZE_Y;

    int object_counter = 1;
    cl::Buffer counter(context, CL_MEM_READ_WRITE, sizeof(int));
    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);

    cl::Buffer roots(context, CL_MEM_READ_WRITE,
                     sizeof(unsigned char) * width * height);

    cl::NDRange global(global_width, global_height);
    cl::NDRange local(UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);

    //cl::Buffer(context, CL_)

    relabel_first_kernel.setArg(0, labels);
    relabel_first_kernel.setArg(1, roots);
    relabel_first_kernel.setArg(2, counter);
    relabel_first_kernel.setArg(3, width);
    relabel_first_kernel.setArg(4, height);

    queue.enqueueNDRangeKernel(relabel_first_kernel, cl::NullRange,
                               global, local);
    queue.enqueueBarrier();

    relabel_second_kernel.setArg(0, labels);
    relabel_second_kernel.setArg(1, roots);
    relabel_second_kernel.setArg(2, width);
    relabel_second_kernel.setArg(3, height);
    relabel_second_kernel.setArg(4, bgval);

    queue.enqueueNDRangeKernel(relabel_second_kernel, cl::NullRange,
                               global, local);

    queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);

    return object_counter - 1;
}


void area_threshold(cl::Buffer labels, int width, int height,
                    int bgval, int min_size, int max_size,
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

    cl::Kernel area_threshold_reset(program, "area_threshold_reset");
    cl::Kernel area_threshold_count_kernel(program, "area_threshold_count");
    cl::Kernel area_threshold_kernel(program, "area_threshold");

    size_t global_width = ((width + UF_BLOCK_SIZE_X - 1) / UF_BLOCK_SIZE_X)
                            * UF_BLOCK_SIZE_X;

    size_t global_height = ((height + UF_BLOCK_SIZE_Y - 1)/UF_BLOCK_SIZE_Y)
                            * UF_BLOCK_SIZE_Y;

//    int object_counter = 1;
//    cl::Buffer counter(context, CL_MEM_READ_WRITE, sizeof(int));
//    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);

    cl::Buffer area_counters(context, CL_MEM_READ_WRITE,
                             sizeof(int) * width * height);


    cl::NDRange global(global_width, global_height);
    cl::NDRange local(UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);

    //cl::Buffer(context, CL_)

    area_threshold_reset.setArg(0, area_counters);
    area_threshold_reset.setArg(1, width);
    area_threshold_reset.setArg(2, height);

    queue.enqueueNDRangeKernel(area_threshold_reset, cl::NullRange,
                               global, local);

    queue.enqueueBarrier();

    area_threshold_count_kernel.setArg(0, labels);
    area_threshold_count_kernel.setArg(1, area_counters);
    area_threshold_count_kernel.setArg(2, width);
    area_threshold_count_kernel.setArg(3, height);
    area_threshold_count_kernel.setArg(4, bgval);

    cl::NDRange global_t(((width + UF_BLOCK_SIZE_X * 8 - 1) / (UF_BLOCK_SIZE_X*8))
                         * (UF_BLOCK_SIZE_X * 8), 1);
    cl::NDRange local_t(UF_BLOCK_SIZE_X * 8, 1);

    queue.enqueueNDRangeKernel(area_threshold_count_kernel, cl::NullRange,
                               global_t, local_t);
    queue.enqueueBarrier();

    area_threshold_kernel.setArg(0, labels);
    area_threshold_kernel.setArg(1, area_counters);
    area_threshold_kernel.setArg(2, min_size);
    area_threshold_kernel.setArg(3, max_size);
    area_threshold_kernel.setArg(4, width);
    area_threshold_kernel.setArg(5, height);
    area_threshold_kernel.setArg(6, bgval);

    queue.enqueueNDRangeKernel(area_threshold_kernel, cl::NullRange,
                               global, local);

    //queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);
    //return object_counter - 1;
}


void bounding_box(cl::Buffer labels, int width, int height,
                  int bgval, int& count,
                  cl::Buffer out_labels,
                  cl::Buffer x_min, cl::Buffer x_max,
                  cl::Buffer y_min, cl::Buffer y_max,
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

    cl::Kernel b_box_init_kernel(program, "b_box_init");
    cl::Kernel b_box_horizontal_kernel(program, "b_box_horizontal");
    cl::Kernel b_box_vertical_kernel(program, "b_box_vertical");
    cl::Kernel b_box_pack_kernel(program, "b_box_pack");

    size_t global_width = ((width + UF_BLOCK_SIZE_X - 1) / UF_BLOCK_SIZE_X)
                            * UF_BLOCK_SIZE_X;

    size_t global_height = ((height + UF_BLOCK_SIZE_Y - 1)/UF_BLOCK_SIZE_Y)
                            * UF_BLOCK_SIZE_Y;

    int object_counter = 0;
    cl::Buffer counter(context, CL_MEM_READ_WRITE, sizeof(int));
    queue.enqueueWriteBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);


    cl::NDRange global(global_width, global_height);
    cl::NDRange local(UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);

    b_box_init_kernel.setArg(0, x_min);
    b_box_init_kernel.setArg(1, x_max);
    b_box_init_kernel.setArg(2, y_min);
    b_box_init_kernel.setArg(3, y_max);
    b_box_init_kernel.setArg(4, width);
    b_box_init_kernel.setArg(5, height);

    queue.enqueueNDRangeKernel(b_box_init_kernel, cl::NullRange,
                               global, local);

    queue.enqueueBarrier();

    b_box_horizontal_kernel.setArg(0, labels);
    b_box_horizontal_kernel.setArg(1, x_min);
    b_box_horizontal_kernel.setArg(2, x_max);
    b_box_horizontal_kernel.setArg(3, width);
    b_box_horizontal_kernel.setArg(4, height);
    b_box_horizontal_kernel.setArg(5, bgval);

    cl::NDRange global_t(1, global_height);
    cl::NDRange local_t(1, UF_BLOCK_SIZE_Y);

    queue.enqueueNDRangeKernel(b_box_horizontal_kernel, cl::NullRange,
                               global_t, local_t);

    queue.enqueueBarrier();

    b_box_vertical_kernel.setArg(0, labels);
    b_box_vertical_kernel.setArg(1, y_min);
    b_box_vertical_kernel.setArg(2, y_max);
    b_box_vertical_kernel.setArg(3, width);
    b_box_vertical_kernel.setArg(4, height);
    b_box_vertical_kernel.setArg(5, bgval);

    cl::NDRange global_t_1(global_width, 1);
    cl::NDRange local_t_1(UF_BLOCK_SIZE_Y, 1);

    queue.enqueueNDRangeKernel(b_box_vertical_kernel, cl::NullRange,
                               global_t_1, local_t_1);

    b_box_pack_kernel.setArg(0, labels);
    b_box_pack_kernel.setArg(1, counter);
    b_box_pack_kernel.setArg(2, x_min);
    b_box_pack_kernel.setArg(3, x_max);
    b_box_pack_kernel.setArg(4, y_min);
    b_box_pack_kernel.setArg(5, y_max);
    b_box_pack_kernel.setArg(6, width);
    b_box_pack_kernel.setArg(7, height);

    queue.enqueueNDRangeKernel(b_box_pack_kernel, cl::NullRange,
                               global, local);

    //queue.enqueueReadBuffer(counter, CL_TRUE, 0, sizeof(int), &object_counter);
    //return object_counter - 1;

}
