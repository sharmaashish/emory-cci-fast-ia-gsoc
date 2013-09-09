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

/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/


uint64 morphReconOcl(const std::string& marker_file,
                    const std::string& mask_file,
                    const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        std::cout << "reading data..." << std::endl;

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

        std::cout << "runnging MR" << std::endl;

        uint64 t1, t2;

        t1 = cci::common::event::timestampInUS();

        morphRecon<int, unsigned char>(device_marker, device_mask,
                                       width, height, 2, 14);

        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "MR finsihed" << std::endl;
        std::cout << "MR time: " << exec_time << "us" << std::endl;

        //cv::imwrite(DATA_OUT("reconstruction_out_0.png"), markerInt);

        queue.enqueueReadBuffer(device_marker, CL_TRUE, 0,
                                 sizeof(int) * size, markerInt.data);

        if(!output_file.empty())
            cv::imwrite(output_file, markerInt);
    }

    std::cout << "MR, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;

    return total_time;
}

uint64 morphReconCuda(const std::string& marker_file,
                      const std::string& mask_file,
                      const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        Mat marker = imread(marker_file);
        Mat mask = imread(mask_file);

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

        std::cout << "morph recon start" << std::endl;

        t1 = cci::common::event::timestampInUS();
        g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(
                    g_marker, g_mask, 4, numFirstPass, stream, nBlocks);

        stream.waitForCompletion();
        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "morph recon finished" << std::endl;
        std::cout << "morph recon speedup time: " << exec_time << "ms" << std::endl;

        Mat recon;

        g_recon.download(recon);

        if(!output_file.empty())
            imwrite(output_file, recon);
    }

    std::cout << "morph recon speedup, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;

    return total_time;
}

uint64 morphReconCpu(const std::string& marker_file,
                     const std::string& mask_file,
                     const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        std::cout << "reading data..." << std::endl;

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

        std::cout << "MR cpu finsihed" << std::endl;
        std::cout << "MR cpu time: " << exec_time << "us" << std::endl;

        if(!output_file.empty())
            cv::imwrite(output_file, recon);
    }

    std::cout << "MR, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;

    return total_time;
}

uint64 morphReconCpuMulticore(const std::string& marker_file,
                              const std::string& mask_file,
                              const std::string& output_file, int iter_num)
{
    uint64 total_time = 0;

    for(int i = 0; i < iter_num; ++i)
    {
        std::cout << "reading data..." << std::endl;

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

        int nThreads = 2;

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
        std::cout << "QueueTime = "<< t2-t1 << std::endl;

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "MR cpu finsihed" << std::endl;
        std::cout << "MR cpu time: " << exec_time << "us" << std::endl;

        cv::imwrite(output_file, reconQueue);
    }

    std::cout << "MR, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;

    return total_time;
}


BOOST_AUTO_TEST_CASE(morphReconOclTest)
{
    const std::string marker = DATA_IN("microscopy/in-imrecon-gray-marker.png");
    const std::string mask = DATA_IN("microscopy/in-imrecon-gray-mask.png");
    const std::string out = DATA_OUT("reconstruction_out.png");

    morphReconOcl(marker, mask , out, ITER_NUM);
}


BOOST_AUTO_TEST_CASE(morhReconCudaTest)
{
    const std::string marker = DATA_IN("microscopy/in-imrecon-gray-marker.png");
    const std::string mask = DATA_IN("microscopy/in-imrecon-gray-mask.png");
    const std::string out = DATA_OUT("reconstruction_out.png");

    morphReconCuda(marker, mask , out, ITER_NUM);
}


BOOST_AUTO_TEST_CASE(morphReconCpuTest)
{
    const std::string marker = DATA_IN("microscopy/in-imrecon-gray-marker.png");
    const std::string mask = DATA_IN("microscopy/in-imrecon-gray-mask.png");
    const std::string out = DATA_OUT("reconstruction_out.png");

    morphReconCpu(marker, mask , out, ITER_NUM);
}


BOOST_AUTO_TEST_CASE(morphReconCpuMulticoreTest)
{
    const std::string marker = DATA_IN("microscopy/in-imrecon-gray-marker.png");
    const std::string mask = DATA_IN("microscopy/in-imrecon-gray-mask.png");
    const std::string out = DATA_OUT("reconstruction_out.png");

    morphReconCpuMulticore(marker, mask , out, ITER_NUM);
}
