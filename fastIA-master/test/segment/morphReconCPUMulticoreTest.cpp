#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"
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
        cv::Mat reconQueue = nscale::imreconstructParallelQueue<unsigned char>(marker_border, mask_border,8,true, nThreads);
        t2 = cci::common::event::timestampInUS();
        std::cout << "QueueTime = "<< t2-t1 << std::endl;

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "MR cpu finsihed" << std::endl;
        std::cout << "MR cpu time: " << exec_time << "us" << std::endl;

        cv::imwrite(DATA_OUT("reconstruction_out_cpu_multicore.png"), reconQueue);
    }

    std::cout << "MR, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;
}
