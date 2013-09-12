#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>

#include <iostream>
#include <stdio.h>

#include "TestUtils.h"
#include "Logger.h"
#include "opencl/utils/ocl_utils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"

#include "MorphologicOperations.h"
#include "UtilsCVImageIO.h"

#define ITER_NUM 1


/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/


BOOST_AUTO_TEST_CASE(cclTest)
{
    const std::string binary_image_file = DATA_IN("sizePhantom.png");

    uint64 total_time = 0;

    for(int i = 0; i < ITER_NUM; ++i)
    {
        std::cout << "reading data..." << std::endl;

        cv::Mat binary_image = cv::imread(binary_image_file, -1);

        assert(binary_image.channels() == 1);
        assert(binary_image.type() == CV_8U);

        std::cout << "running cuda code..." << std::endl;
        std::cout << "upload data..." << std::endl;

        Stream stream;
        GpuMat g_maskb = cv::gpu::createContinuous(binary_image.size(),
                                                   binary_image.type());

        stream.enqueueUpload(binary_image, g_maskb);

        stream.waitForCompletion();

        std::cout << "running kernel..." << std::endl;

        cv::gpu::GpuMat d_output = nscale::gpu::bwlabel(g_maskb, 4, false, stream);

        std::cout << "downloading data..." << std::endl;

        cv::Mat output(binary_image.size(), CV_32SC1);

        stream.enqueueDownload(d_output, output);
        stream.waitForCompletion();

        cciutils::cv::normalizeLabels(output);

        cv::imwrite(DATA_OUT("ccl_output_cuda.png"), output);
    }

 //   return total_time;
}

