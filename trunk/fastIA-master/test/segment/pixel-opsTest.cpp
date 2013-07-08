#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

#include "opencl/pixel-ops.h"


/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

/*
 * Used kernels:
 *  -> invert
 *  -> threshold
 **/

BOOST_AUTO_TEST_CASE(simple_operations_test)
{

    std::cout << "invert test hello" << std::endl;

    cl_int err = CL_SUCCESS;

    try {

        cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

        cv::Mat img = cv::imread(DATA_IN("coins.png"));
        cv::Mat labeled(img);

        cv::Size img_size = img.size();

        size_t img_byte_size = img_size.height * img.step;
        size_t img_byte_width = img_size.width * img.elemSize();

        cl::Buffer srcBuff;
        cl::Buffer dstBuff(context, CL_TRUE, img_byte_size);

        ocvMatToOclBuffer(img, srcBuff, context, queue);

        invert(img_byte_width, img_size.height, srcBuff, img.step, dstBuff, img.step);

        oclBufferToOcvMat(labeled, dstBuff, img_byte_size, queue);
        cv::imwrite(DATA_OUT("coins_invert.png"), labeled);

        threshold(img_byte_width, img_size.height, srcBuff, img.step, dstBuff, img.step,
                  90, 240, true, true);

        oclBufferToOcvMat(labeled, dstBuff, img_byte_size, queue);
        cv::imwrite(DATA_OUT("coins_threshold.png"), labeled);

    }
    catch (cl::Error err) {
        oclPrintError(err);
    }
}

