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

        cl::Context context;
        std::vector<cl::Device> devices;

        oclSimpleInit(CL_DEVICE_TYPE_ALL, context, devices);

        cl::Device device = devices[0];

        std::cout << "devices count: " << devices.size() << std::endl;

        ProgramCache cache(context, device);

        cl::Kernel invertKernel = cache.getKernel("Invert", "invert");
        cl::Kernel thresholdKernel = cache.getKernel("Threshold", "threshold");
        cl::Kernel bgr2grayKernel = cache.getKernel("Bgr2gray", "bgr2gray");
        cl::Kernel maskKernel = cache.getKernel("Mask", "mask");
        cl::Kernel divideKernel = cache.getKernel("Divide", "divide");
        cl::Kernel replaceKernel = cache.getKernel("Replace", "replace");

        cl::CommandQueue queue(context, device, 0, &err);

        cv::Mat img = cv::imread(DATA_IN("coins.png"));
        cv::Mat img_out(img);

        cv::Size img_size = img.size();

        size_t img_byte_size = img_size.height * img.step;
        size_t img_byte_width = img_size.width * img.elemSize();

        cl::Buffer srcBuff;
        cl::Buffer dstBuff(context, CL_TRUE, img_byte_size);

        ocvMatToOclBuffer(img, srcBuff, context, queue);

        invert(queue, invertKernel, img_byte_width, img_size.height, srcBuff, img.step, dstBuff, img.step);

        oclBufferToOcvMat(img_out, dstBuff, img_byte_size, context, queue);
        cv::imwrite(DATA_OUT("coins_invert.png"), img_out);

        threshold(queue, thresholdKernel, img_byte_width, img_size.height, srcBuff, img.step, dstBuff, img.step,
                  90, 240, true, true);

        oclBufferToOcvMat(img_out, dstBuff, img_byte_size, context, queue);
        cv::imwrite(DATA_OUT("coins_threshold.png"), img_out);

    }
    catch (cl::Error err) {
        oclPrintError(err);
    }
}

