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

#include "opencl/component_labeling.h"

#include "UtilsCVImageIO.h"

#define ITER_NUM 1


/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/


BOOST_AUTO_TEST_CASE(cclTest)
{
    const std::string binary_image_file = DATA_IN("cell_binary_mask.png");

    uint64 total_time = 0;

    for(int i = 0; i < ITER_NUM; ++i)
    {
        std::cout << "reading data..." << std::endl;

        cv::Mat binary_image = cv::imread(binary_image_file, -1);

        assert(binary_image.channels() == 1);
        assert(binary_image.type() == CV_8U);

        cv::Mat output(binary_image.size(), CV_32SC1);

        int width = binary_image.cols;
        int height = binary_image.rows;

        int size = width * height;

        cl::CommandQueue queue = ProgramCache::getDefaultCommandQueue();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

        cl::Buffer d_image(context, CL_TRUE, sizeof(unsigned char) * size);
        cl::Buffer d_output(context, CL_TRUE, sizeof(int) * size);

        queue.enqueueWriteBuffer(d_image, CL_TRUE, 0, sizeof(unsigned char)
                                                    * size, binary_image.data);


        std::cout << "runnging CCL" << std::endl;

        uint64 t1, t2;

        t1 = cci::common::event::timestampInUS();

        ccl(d_image, d_output, width, height, 0, 4);

        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        //std::cout << "MR finsihed" << std::endl;
        //std::cout << "MR time: " << exec_time << "us" << std::endl;

        //cv::imwrite(DATA_OUT("reconstruction_out_0.png"), markerInt);

        queue.enqueueReadBuffer(d_output, CL_TRUE, 0,
                                 sizeof(int) * size, output.data);

        cciutils::cv::normalizeLabels(output);

        cv::imwrite(DATA_OUT("ccl_output.png"), output);

//        if(!output_file.empty() && i == 0)
//            cv::imwrite(output_file, markerInt);
    }

 //   return total_time;
}

