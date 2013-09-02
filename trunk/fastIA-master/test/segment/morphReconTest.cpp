#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"

#include <iostream>

#include "opencv2/opencv.hpp"

#include "opencl/morph_recon.h"

#define ITER_NUM 1

/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

BOOST_AUTO_TEST_CASE(morphReconstruction)
{
    for(int i = 0; i < ITER_NUM; ++i)
    {
        std::cout << "reading data..." << std::endl;

        cv::Mat marker = cv::imread(DATA_IN("microscopy/in-imrecon-gray-marker.png"), -1);
        cv::Mat mask = cv::imread(DATA_IN("microscopy/in-imrecon-gray-mask.png"), -1);

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
        morphRecon<int, unsigned char>(device_marker, device_mask,
                                       width, height, 2, 2);

        cv::imwrite(DATA_OUT("reconstruction_out_0.png"), markerInt);

        queue.enqueueReadBuffer(device_marker, CL_TRUE, 0,
                                 sizeof(int) * size, markerInt.data);

        cv::imwrite(DATA_OUT("reconstruction_out.png"), markerInt);
    }
}
