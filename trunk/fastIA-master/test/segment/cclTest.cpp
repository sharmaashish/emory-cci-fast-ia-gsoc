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
    //const std::string binary_image_file = DATA_IN("cell_binary_mask.png");
//    const std::string binary_image_file = DATA_IN("sizePhantom.png");

    const std::string binary_image_file = DATA_IN("braintumor/astroII.1.ndpi-0000008192-0000008192.mask.png");


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

        cl::Buffer d_image(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * size);
        cl::Buffer d_output(context, CL_MEM_READ_WRITE, sizeof(int) * size);

        queue.enqueueWriteBuffer(d_image, CL_TRUE, 0, sizeof(unsigned char)
                                                    * size, binary_image.data);


        std::cout << "runnging CCL" << std::endl;

        uint64 t1, t2, t3, t4;

        t1 = cci::common::event::timestampInUS();

        ccl(d_image, d_output, width, height, -1, 4);

        t3 = cci::common::event::timestampInUS();
        area_threshold(d_output, width, height, -1, 150,
                       std::numeric_limits<int>::max());
        queue.enqueueBarrier();
        t4 = cci::common::event::timestampInUS();

        int object_count = relabel(d_output, width, height, -1);

        cl::Buffer x_min(context, CL_MEM_READ_WRITE, sizeof(int) * size);
        cl::Buffer x_max(context, CL_MEM_READ_WRITE, sizeof(int) * size);
        cl::Buffer y_min(context, CL_MEM_READ_WRITE, sizeof(int) * size);
        cl::Buffer y_max(context, CL_MEM_READ_WRITE, sizeof(int) * size);

        int bb_count = 0;


        bounding_box(d_output, width, height, 0, bb_count,
                     x_min, x_max, y_min, y_max);



        int* x_min_host = new int[size];
        int* x_max_host = new int[size];
        int* y_min_host = new int[size];
        int* y_max_host = new int[size];

        queue.enqueueReadBuffer(x_min, CL_TRUE, 0,
                                size * sizeof(int), x_min_host);
        queue.enqueueReadBuffer(x_max, CL_TRUE, 0,
                                size * sizeof(int), x_max_host);
        queue.enqueueReadBuffer(y_min, CL_TRUE, 0,
                                size * sizeof(int), y_min_host);
        queue.enqueueReadBuffer(y_max, CL_TRUE, 0,
                                size * sizeof(int), y_max_host);

        for(int i = 0; i < 10; ++i)
        {
            std::cout << "x_min: " << x_min_host[i] << ", ";
            std::cout << "x_max: " << x_max_host[i] << ", ";
            std::cout << "y_min: " << y_min_host[i] << ", ";
            std::cout << "y_max: " << y_max_host[i] << std::endl;
        }

        delete[] x_min_host;
        delete[] x_max_host;
        delete[] y_min_host;
        delete[] y_max_host;

        std::cout << "bounding box, count: " << bb_count << std::endl;




        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        uint64 threshold_time = t4-t3;
        std::cout << "threshold time: " << threshold_time << std::endl;

        std::cout << "objects count: " << object_count << std::endl;

        //std::cout << "MR finsihed" << std::endl;
        //std::cout << "MR time: " << exec_time << "us" << std::endl;

        //cv::imwrite(DATA_OUT("reconstruction_out_0.png"), markerInt);

        queue.enqueueReadBuffer(d_output, CL_TRUE, 0,
                                 sizeof(int) * size, output.data);

        //cciutils::cv::normalizeLabels(output);

        cv::imwrite(DATA_OUT("ccl_output.png"), output);




        //bounding_box();

//        if(!output_file.empty() && i == 0)
//            cv::imwrite(output_file, markerInt);
    }

 //   return total_time;
}

