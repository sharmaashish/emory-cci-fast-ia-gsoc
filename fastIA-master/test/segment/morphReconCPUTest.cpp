#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "Logger.h"

#include <iostream>

#include "opencv2/opencv.hpp"

#include "MorphologicOperations.h"


#define ITER_NUM 1

/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

BOOST_AUTO_TEST_CASE(morphReconstructionCPU)
{
    uint64 total_time = 0;

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

        uint64_t t1, t2;

        t1 = cci::common::event::timestampInUS();
        cv::Mat recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "MR cpu finsihed" << std::endl;
        std::cout << "MR cpu time: " << exec_time << "us" << std::endl;

        cv::imwrite(DATA_OUT("reconstruction_out_cpu.png"), recon);
    }

    std::cout << "MR, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;
}
