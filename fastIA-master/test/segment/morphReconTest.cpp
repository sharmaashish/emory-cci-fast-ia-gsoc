#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>

#include "TestUtils.h"
#include "Logger.h"
#include "opencl/utils/ocl_utils.h"

#include <iostream>
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"

#include "MorphologicOperations.h"

#include "opencl/morph_recon.h"

#define ITER_NUM 1

#define PROCESS_FRIRST_N 5
#define WRITE_OUTPUT /* first iteration will write output to file */

/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/


static const char* markers[] = {
    "mr_tests/in-imrecon-gray-marker.png",
    "mr_tests/gbm2.1.ndpi-0000004096-0000004096_inv_eroded_4x.png",
    "mr_tests/normal.3.ndpi-0000028672-0000012288_inv_eroded_3x.png",
    "mr_tests/oligoastroIII.1.ndpi-0000053248-0000008192_inv_eroded_3x.png",
    "mr_tests/oligoIII.1.ndpi-0000012288-0000028672_inv_eroded_3x.png"
};

static const char* masks[] = {
    "mr_tests/in-imrecon-gray-mask.png",
    "mr_tests/gbm2.1.ndpi-0000004096-0000004096_inv.png",
    "mr_tests/normal.3.ndpi-0000028672-0000012288_inv.png",
    "mr_tests/oligoastroIII.1.ndpi-0000053248-0000008192_inv.png",
    "mr_tests/oligoIII.1.ndpi-0000012288-0000028672_inv.png"
};

static const char* outputs[] = {
    "mr_out_in-imrecon-gray-marker_out.png",
    "mr_out_gbm2.1.ndpi-0000004096-0000004096_inv_eroded_4x_out.png",
    "mr_out_normal.3.ndpi-0000028672-0000012288_inv_eroded_3x_out.png",
    "mr_out_oligoastroIII.1.ndpi-0000053248-0000008192_inv_eroded_3x_out.png",
    "mr_out_oligoIII.1.ndpi-0000012288-0000028672_inv_eroded_3x_out.png"
};


uint64 morphReconOcl(const std::string& marker_file,
                    const std::string& mask_file,
                    const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
       // std::cout << "reading data..." << std::endl;

        cv::Mat marker = cv::imread(marker_file, -1);
        cv::Mat mask = cv::imread(mask_file, -1);

        cv::Mat markerInt;
        cv::Mat maskUChar;

        marker.convertTo(markerInt, CV_32S);
        mask.convertTo(maskUChar, CV_8UC1);

        assert(markerInt.channels() == 1);
        assert(maskUChar.channels() == 1);

        assert(markerInt.isContinuous());
        assert(maskUChar.isContinuous());

        int marker_width = marker.cols;
        int marker_height = marker.rows;

        int mask_width = mask.cols;
        int mask_height = mask.rows;

        assert(marker_width == mask_width);
        assert(marker_height == marker_height);

        int width = marker_width;
        int height = marker_height;
        int size = width * height;

        cl::CommandQueue queue
                = ProgramCache::getGlobalInstance().getDefaultCommandQueue();

        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

        cl::Buffer device_marker(context, CL_TRUE, sizeof(int) * size);
        cl::Buffer device_mask(context, CL_TRUE, sizeof(unsigned char) * size);

        queue.enqueueWriteBuffer(device_marker, CL_TRUE, 0,
                                 sizeof(int) * size, markerInt.data);

        queue.enqueueWriteBuffer(device_mask, CL_TRUE, 0,
                                 sizeof(unsigned char) * size, maskUChar.data);

        //std::cout << "runnging MR" << std::endl;

        uint64 t1, t2;

        t1 = cci::common::event::timestampInUS();

        morphRecon<int, unsigned char>(device_marker, device_mask,
                                       width, height, 2, 14);

        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        //std::cout << "MR finsihed" << std::endl;
        //std::cout << "MR time: " << exec_time << "us" << std::endl;

        //cv::imwrite(DATA_OUT("reconstruction_out_0.png"), markerInt);

        queue.enqueueReadBuffer(device_marker, CL_TRUE, 0,
                                 sizeof(int) * size, markerInt.data);

        if(!output_file.empty() && i == 0)
            cv::imwrite(output_file, markerInt);
    }

    return total_time;
}

uint64 morphReconCuda(const std::string& marker_file,
                      const std::string& mask_file,
                      const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        Mat marker = imread(marker_file, -1);
        Mat mask = imread(mask_file, -1);

        Stream stream;
        GpuMat g_marker;
        GpuMat g_mask, g_recon;

        stream.enqueueUpload(marker, g_marker);
        stream.enqueueUpload(mask, g_mask);

        Mat markerInt, maskInt;
        marker.convertTo(markerInt, CV_32SC1, 1, 0);
        mask.convertTo(maskInt, CV_32SC1, 1, 0);

        GpuMat g_marker_int, g_mask_int;
        stream.enqueueUpload(markerInt, g_marker_int);
        stream.enqueueUpload(maskInt, g_mask_int);
        stream.waitForCompletion();

        uint64_t t1, t2;
        int numFirstPass = 2;
        int nBlocks = 14;

//        std::cout << "morph recon start" << std::endl;

        t1 = cci::common::event::timestampInUS();
        g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(
                    g_marker, g_mask, 4, numFirstPass, stream, nBlocks);

        stream.waitForCompletion();
        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

//        std::cout << "morph recon finished" << std::endl;
//        std::cout << "morph recon speedup time: " << exec_time << "ms" << std::endl;

        Mat recon;

        g_recon.download(recon);

        if(!output_file.empty() && i == 0)
            imwrite(output_file, recon);
    }

    return total_time;
}

uint64 morphReconCpu(const std::string& marker_file,
                     const std::string& mask_file,
                     const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
     //   std::cout << "reading data..." << std::endl;

        cv::Mat marker = cv::imread(marker_file, -1);
        cv::Mat mask = cv::imread(mask_file, -1);

        cv::Mat markerInt;
        cv::Mat maskUChar;

        marker.convertTo(markerInt, CV_32S);
        mask.convertTo(maskUChar, CV_8UC1);

        assert(markerInt.channels() == 1);
        assert(maskUChar.channels() == 1);

        assert(markerInt.isContinuous());
        assert(maskUChar.isContinuous());

        int marker_width = marker.cols;
        int marker_height = marker.rows;

        int mask_width = mask.cols;
        int mask_height = mask.rows;

        assert(marker_width == mask_width);
        assert(marker_height == marker_height);

        int width = marker_width;
        int height = marker_height;
        int size = width * height;

        uint64_t t1, t2;

        t1 = cci::common::event::timestampInUS();
        cv::Mat recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

//        std::cout << "MR cpu finsihed" << std::endl;
//        std::cout << "MR cpu time: " << exec_time << "us" << std::endl;

        if(!output_file.empty() && i == 0)
            cv::imwrite(output_file, recon);
    }

    return total_time;
}

uint64 morphReconCpuMulticore(const std::string& marker_file,
                              const std::string& mask_file,
                              const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        //std::cout << "reading data..." << std::endl;

        cv::Mat marker = cv::imread(marker_file, -1);
        cv::Mat mask = cv::imread(mask_file, -1);

        cv::Mat markerInt;
        cv::Mat maskUChar;

        marker.convertTo(markerInt, CV_32S);
        mask.convertTo(maskUChar, CV_8UC1);

        assert(markerInt.channels() == 1);
        assert(maskUChar.channels() == 1);

        assert(markerInt.isContinuous());
        assert(maskUChar.isContinuous());

        int marker_width = marker.cols;
        int marker_height = marker.rows;

        int mask_width = mask.cols;
        int mask_height = mask.rows;

        assert(marker_width == mask_width);
        assert(marker_height == marker_height);

        int width = marker_width;
        int height = marker_height;
        int size = width * height;

        int nThreads = 4;

        uint64_t t1, t2;

        cv::Mat marker_border(marker.size() + cv::Size(2,2), marker.type());
        copyMakeBorder(marker, marker_border, 1, 1, 1, 1, BORDER_CONSTANT, 0);
        cv::Mat mask_border(mask.size() + cv::Size(2,2), mask.type());
        copyMakeBorder(mask, mask_border, 1, 1, 1, 1, BORDER_CONSTANT, 0);

        mask.release();marker.release();
        cv::Mat marker_copy(marker_border, cv::Rect(1,1,marker_border.cols-2,marker_border.rows-2));
        cv::Mat mask_copy(mask_border, cv::Rect(1,1,mask_border.cols-2,mask_border.rows-2));
        marker.release(); mask.release();
        t1 = cci::common::event::timestampInUS();
        cv::Mat reconQueue = nscale::imreconstructParallelQueue<unsigned char>(marker_border,
                                                                               mask_border,8,true, nThreads);
        t2 = cci::common::event::timestampInUS();
//        std::cout << "QueueTime = "<< t2-t1 << std::endl;

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        if(!output_file.empty() && i == 0)
            cv::imwrite(output_file, reconQueue);
    }


    return total_time;
}



enum MR_TYPE { OPENCL, CUDA, CPU, CPU_MULTICORE};


void printTime(uint64 time)
{
    std::cout << "time raw: " << time << std::endl;

    uint64 tmp = time / 1000;

    int ms = tmp % 1000;
    int s = tmp / 1000;
    int min = 0;

    if(s > 60)
    {
        min = s / 60;
        s = s % 60;
    }

    if(min)
        std::cout << min << "min " << s << "s " << ms << "ms" << std::endl;
    else
        std::cout << s << "s " << ms << "ms" << std::endl;
}


std::string getOutSuffix(MR_TYPE type)
{
    switch(type)
    {
    case OPENCL:
        return "_opencl.png";
    case CUDA:
        return "_cuda.png";
    case CPU:
        return "_cpu.png";
    case CPU_MULTICORE:
        return "_cpu_multicore.png";
    }
}

void testRunner(MR_TYPE type)
{
    int count = sizeof(markers)/sizeof(markers[0]);

    std::cout << "number of test pairs (mask, marker): " << count << std::endl;

    uint64 time_total = 0;

    for(int i = 0; i < count && i < PROCESS_FRIRST_N; ++i)
    {
        const char* marker = markers[i];
        const char* mask = masks[i];
        const char* output = outputs[i];

        std::cout << "XXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        std::cout << "dataset " << i << ":" << std::endl;
        std::cout << "marker: " << marker << std::endl;
        std::cout << "mask: " << mask << std::endl;

        std::string out;

#ifdef WRITE_OUTPUT
        out = output;
        out = out.substr(0, out.size() - 4);
        out += getOutSuffix(type);
        out = DATA_OUT(out);
#endif
        uint64 exec_time;

        switch(type)
        {
        case OPENCL:
            exec_time = morphReconOcl(DATA_IN(marker),
                                      DATA_IN(mask), out, ITER_NUM);
            break;
        case CUDA:
            exec_time = morphReconCuda(DATA_IN(marker),
                                       DATA_IN(mask), out, ITER_NUM);
            break;
        case CPU:
            exec_time = morphReconCpu(DATA_IN(marker),
                                      DATA_IN(mask), out, ITER_NUM);
            break;
        case CPU_MULTICORE:
            exec_time = morphReconCpuMulticore(DATA_IN(marker),
                                               DATA_IN(mask), out, ITER_NUM);
            break;
        }

        time_total += exec_time;

        std::cout << "IMG AVG TIME: ";
        printTime(exec_time/ITER_NUM);
        // << exec_time/ITER_NUM << std::endl;
    }

    std::cout << "TOTAL AVG TIME: ";
    printTime(time_total/ITER_NUM);
}


BOOST_AUTO_TEST_CASE(morphReconOclTest)
{
    testRunner(OPENCL);
}


BOOST_AUTO_TEST_CASE(morphReconCudaTest)
{
    testRunner(CUDA);
}


BOOST_AUTO_TEST_CASE(morphReconCpuTest)
{
    testRunner(CPU);
}


BOOST_AUTO_TEST_CASE(morphReconCpuMulticoreTest)
{
    testRunner(CPU_MULTICORE);
}



BOOST_AUTO_TEST_CASE(morphReconAllTest)
{
    std::cout << "###################### ";
    std::cout << "RUNNING TESTS USING OPENCL" << std::endl;
    testRunner(OPENCL);
    std::cout << "###################### ";
    std::cout << "RUNNING TESTS USING CUDA" << std::endl;
    testRunner(CUDA);
    std::cout << "###################### ";
    std::cout << "RUNNING TESTS USING CPU" << std::endl;
    testRunner(CPU);
    std::cout << "###################### ";
    std::cout << "RUNNING TESTS USING CPU MULTICORE" << std::endl;
    testRunner(CPU_MULTICORE);

    std::cout << "###################### ";
    std::cout << "TESTS FINISHED" << std::endl;
}


