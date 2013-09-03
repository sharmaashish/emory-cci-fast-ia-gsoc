#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

#include "opencl/parallel_queue.h"
#include "opencl/staticInitializer.h"

#include <cstdio>

#define ITER_NUM 1000

//extern cl::Buffer queue_workspace;

static const int oclSourceInit = StaticInitializer::forceOclSourceRegistration();

/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

void dequeueTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
                 cl::Buffer& device_result,
                 ProgramCache& cache = ProgramCache::getGlobalInstance(),
                 cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                    .getDefaultCommandQueue());

//void sumTest(cl::Buffer& device_result, int iterations,
//             ProgramCache& cache = ProgramCache::getGlobalInstance(),
//             cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
//                                                    .getDefaultCommandQueue());

void sumTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
             cl::Buffer& device_result, int iterations,
             ProgramCache& cache = ProgramCache::getGlobalInstance(),
             cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                   .getDefaultCommandQueue());

void bigLocalQueuesTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
            int iterations,
            ProgramCache& cache = ProgramCache::getGlobalInstance(),
            cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                   .getDefaultCommandQueue());

void dequeueTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
                 cl::Buffer& device_result, ProgramCache& cache,
                 cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("ParallelQueueTests");

    cl::Program& program = cache.getProgram(sources, program_params);

    cl::Kernel dequeue_test_kernel(program, "dequeue_test");

    cl::LocalSpaceArg local_mem = cl::__local(sizeof(int));

    dequeue_test_kernel.setArg(0, queue_data);
    dequeue_test_kernel.setArg(1, queue_metadata);
    dequeue_test_kernel.setArg(2, device_result);
    dequeue_test_kernel.setArg(3, local_mem);

    cl::NDRange nullRange;
    cl::NDRange global(512, 1);
    cl::NDRange local(512, 1);

    cl_int status = queue.enqueueNDRangeKernel(dequeue_test_kernel,
                                               nullRange, global, local);
}


void sumTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
             cl::Buffer& device_result, int iterations,
             ProgramCache& cache,
             cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("ParallelQueueTests");

    cl::Program& program = cache.getProgram(sources, program_params);

    cl::Kernel sum_test_kernel(program, "sum_test");

    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

    int warp_size = sum_test_kernel
        .getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

    std::cout << "warp size: " << warp_size << std::endl;

    cl::LocalSpaceArg local_queue
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS * 2);
    cl::LocalSpaceArg reduction_buffer
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg got_work
            = cl::__local(sizeof(int));
    cl::LocalSpaceArg prefix_sum_input
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg prefix_sum_output
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);

    sum_test_kernel.setArg(0, queue_data);
    sum_test_kernel.setArg(1, queue_metadata);
    sum_test_kernel.setArg(2, device_result);
    sum_test_kernel.setArg(3, iterations);
    sum_test_kernel.setArg(4, local_queue);
    sum_test_kernel.setArg(5, reduction_buffer);
    sum_test_kernel.setArg(6, got_work);
    sum_test_kernel.setArg(7, prefix_sum_input);
    sum_test_kernel.setArg(8, prefix_sum_output);

    cl::NDRange nullRange;
    cl::NDRange global(QUEUE_NUM_THREADS, 1);
    cl::NDRange local(QUEUE_NUM_THREADS, 1);

    cl_int status = queue.enqueueNDRangeKernel(sum_test_kernel,
                                               nullRange, global, local);
}

void bigLocalQueuesTest(cl::Buffer queue_data, cl::Buffer queue_metadata,
                        int iterations, ProgramCache& cache,
                        cl::CommandQueue& queue)
{
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    std::stringstream params_stream;
    params_stream << "-DQUEUE_MAX_NUM_BLOCKS=" << QUEUE_MAX_NUM_BLOCKS << " ";
    params_stream << "-DQUEUE_NUM_THREADS=" << QUEUE_NUM_THREADS;

    std::string program_params = params_stream.str();

    std::vector<std::string> sources;
    sources.push_back("ParallelQueue");
    sources.push_back("ParallelQueueTests");

    cl::Program& program = cache.getProgram(sources, program_params);

    cl::Kernel big_local_queues_test(program, "big_local_queues_test");

    cl::LocalSpaceArg local_queue
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS * 5);
    cl::LocalSpaceArg got_work
            = cl::__local(sizeof(int));
    cl::LocalSpaceArg prefix_sum_input
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);
    cl::LocalSpaceArg prefix_sum_output
            = cl::__local(sizeof(int) * QUEUE_NUM_THREADS);

    big_local_queues_test.setArg(0, queue_data);
    big_local_queues_test.setArg(1, queue_metadata);
    big_local_queues_test.setArg(2, iterations);
    big_local_queues_test.setArg(3, local_queue);
    big_local_queues_test.setArg(4, got_work);
    big_local_queues_test.setArg(5, prefix_sum_input);
    big_local_queues_test.setArg(6, prefix_sum_output);

    cl::NDRange nullRange;
    cl::NDRange global(QUEUE_NUM_THREADS, 1);
    cl::NDRange local(QUEUE_NUM_THREADS, 1);

   // std::cout << "enqueuing big_local_queues_test kernel" << std::endl;

    cl_int status = queue.enqueueNDRangeKernel(big_local_queues_test,
                                               nullRange, global, local);
    //fflush(stdout);
}


BOOST_AUTO_TEST_CASE(dequeue_test)
{
    for(int i = 0; i < ITER_NUM; ++i)
    {
        cl_int err = CL_SUCCESS;

        const int input_size = 512;
        const int output_size = 512;
        const int total_size = output_size * 2;
        const int result_size = 512;

        const int input_size_bytes = input_size * sizeof(int);
        const int output_size_bytes = output_size * sizeof(int);
        const int result_size_bytes = result_size * sizeof(int);

        try
        {
            cl::CommandQueue queue = ProgramCache::getGlobalInstance()
                                                    .getDefaultCommandQueue();
            cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

            std::cout << "init queue system" << std::endl;
            //initQueueSystem();

            int host_queue_data[total_size];
            int host_result[result_size];

            std::fill(host_queue_data, host_queue_data + input_size, 1);
            std::fill(host_queue_data + input_size,
                      host_queue_data + total_size, 0);

            cl::Buffer device_queue_data(context, CL_TRUE,
                                         sizeof(host_queue_data));
            queue.enqueueWriteBuffer(device_queue_data, CL_TRUE, 0,
                                     sizeof(host_queue_data), host_queue_data);

            cl::Buffer device_queue_metadata;
            //cl::Buffer execution_code;
            int execution_code_offset;

            // initialize queue metadata buffer
            initQueueMetadata(input_size, total_size, device_queue_metadata, execution_code_offset);

            cl::Buffer device_result(context, CL_TRUE, result_size_bytes);

            std::cout << "dequeue test" << std::endl;
            dequeueTest(device_queue_data,
                        device_queue_metadata,device_result);

            queue.enqueueReadBuffer(device_result, CL_TRUE, 0,
                                    result_size_bytes, host_result);

            for(int i = 0; i < result_size; ++i)
            {
                //std::cout << "result[" << i << "] = " << host_result[i] << std::endl;
                assert(host_result[i] == 1);
            }

        }
        catch (cl::Error err)
        {
            oclPrintError(err);
        }

        std::cout << "dequeue test finished" << std::endl;
    }
}


BOOST_AUTO_TEST_CASE(queue_sum_test)
{
    for(int i = 0; i < ITER_NUM; ++i)
    {
        std::cout << "parallel queue sum test hello" << std::endl;

        cl_int err = CL_SUCCESS;

        const int input_size = 1024;
        const int output_size = 1024;
        //const int result_size = 512;
        const int total_size = output_size * 2;

        const int iterations = 10;
        const int result_size = 1;


        const int input_size_bytes = input_size * sizeof(int);
        const int output_size_bytes = output_size * sizeof(int);
        const int result_size_bytes = result_size * sizeof(int);

        try
        {
            cl::CommandQueue queue = ProgramCache::getGlobalInstance()
                    .getDefaultCommandQueue();
            cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

            int host_queue_data[total_size];
            int host_result[result_size];

            std::fill(host_queue_data, host_queue_data + input_size, 1);
            std::fill(host_queue_data + input_size,
                      host_queue_data + total_size, 0);


            cl::Buffer device_queue_data(context, CL_TRUE,
                                         sizeof(host_queue_data));
            queue.enqueueWriteBuffer(device_queue_data, CL_TRUE, 0, sizeof(host_queue_data), host_queue_data);

            cl::Buffer device_queue_metadata;
            //cl::Buffer executionCode;
            int execution_code_offset;

            // initialize queue metadata buffer
            initQueueMetadata(input_size, total_size, device_queue_metadata, execution_code_offset);



            cl::Buffer device_result(context, CL_TRUE, result_size_bytes);

            std::cout << "parallel sum..." << std::endl;
            //dequeueTest(device_result);
            sumTest(device_queue_data, device_queue_metadata,
                    device_result, iterations);

            queue.enqueueReadBuffer(device_result, CL_TRUE, 0,
                                    result_size_bytes, host_result);

            for(int i = 0; i < result_size; ++i)
            {
                std::cout << "result[" << i << "] = "
                          << host_result[i] << std::endl;
            }
        }
        catch (cl::Error err)
        {
            oclPrintError(err);
        }

        std::cout << "sum queue test finished" << std::endl;
    }
}




BOOST_AUTO_TEST_CASE(big_local_queues_test)
{
    std::cout << "big_local_queues_test hello" << std::endl;

    for(int i = 0; i < ITER_NUM; ++i)
    {
        std::cout << "iter: " << i << std::endl;

        cl_int err = CL_SUCCESS;

        const int input_size = 512;
        const int output_size = 512;
        const int total_size = output_size * 2;
        //const int result_size = 512;

        const int iterations = 1;

        const int input_size_bytes = input_size * sizeof(int);
        const int output_size_bytes = output_size * sizeof(int);

        try
        {
            cl::CommandQueue queue = ProgramCache::getGlobalInstance()
                                                    .getDefaultCommandQueue();
            cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

            int host_queue_data[total_size];
            //int host_result[result_size];

            std::fill(host_queue_data, host_queue_data + input_size, 1);
            std::fill(host_queue_data + input_size,
                      host_queue_data + total_size, 0);

            cl::Buffer device_queue_data(context, CL_TRUE,
                                         sizeof(host_queue_data));
            queue.enqueueWriteBuffer(device_queue_data, CL_TRUE, 0, sizeof(host_queue_data), host_queue_data);

           cl::Buffer device_queue_metadata;
           //cl::Buffer executionCode;
           int execution_code_offset;

            // initialize queue metadata buffer
            initQueueMetadata(input_size, total_size, device_queue_metadata, execution_code_offset);


//            int host_in[input_size];
//            int host_out[output_size];

//            std::fill(host_in, host_in + input_size, 1);
//            std::fill(host_out, host_out + output_size, 0);

//            cl::Buffer device_in(context, CL_TRUE, input_size_bytes);
//            queue.enqueueWriteBuffer(device_in, CL_TRUE,
//                                     0, input_size_bytes, host_in);

//            cl::Buffer device_out(context, CL_TRUE, output_size_bytes);
//            queue.enqueueWriteBuffer(device_out, CL_TRUE,
//                                     0, output_size_bytes, host_out);

            // initialize queue
         //   initQueue(device_in, input_size, device_out, output_size);

            bigLocalQueuesTest(device_queue_data, device_queue_metadata,
                               iterations);

        }
        catch (cl::Error err)
        {
            oclPrintError(err);
        }
    }

    std::cout << "big_local_queues_test finished" << std::endl;
}

