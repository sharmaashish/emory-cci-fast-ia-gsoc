#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include "TestUtils.h"
#include "opencl/utils/ocl_utils.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

#include "opencl/watershed.h"

#include "opencv2/ocl/ocl.hpp"


/*
 * runtime parameter --catch_system_errors=no  is needed if there is no gpu
 * (opencl internally invokes subprocess that return non-zero value)
 **/

/*
 * Used kernels:
 *  -> watershed
 **/

BOOST_AUTO_TEST_CASE(simple_operations_test)
{
    std::cout << "watershed test hello" << std::endl;

    cl_int err = CL_SUCCESS;

    try {

//        cl::Context context;
//        std::vector<cl::Device> devices;
//        oclSimpleInit(CL_DEVICE_TYPE_ALL, context, devices);
//        cl::Device device = devices[0];
//        std::cout << "devices count: " << devices.size() << std::endl;
//        ProgramCache cache(context, device);
//        cl::Kernel watershedKernel = cache.getKernel("Watershed", "descent_kernel");
//        cl::CommandQueue queue(context, device, 0, &err);

        cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue();

        cv::Mat img = cv::imread(DATA_IN("watershed_test.png"), CV_LOAD_IMAGE_GRAYSCALE);

        img.convertTo(img, CV_8UC1);

        //cv::imwrite(DATA_OUT("coins_1b.png"), img);

        cv::Size img_size = img.size();
        int width = img_size.width;
        int height = img_size.height;
        int dst_buff_size = width * height * sizeof(float);

        cv::Mat labeled = cv::imread(DATA_IN("watershed_test.png"), CV_LOAD_IMAGE_GRAYSCALE);
        labeled.convertTo(labeled, CV_32F);

       // cv::Mat labeled(height, width, CV_32F, .0f);

     //   cv::imwrite(DATA_OUT("coins_4f.png"), labeled);

        std::cout << "labeled step: " << labeled.step << std::endl;
        std::cout << "labeled width: " << labeled.size().width << std::endl;
        std::cout << "labeled elem size: " << labeled.elemSize() << std::endl;



        cl::Buffer srcBuff = ocvMatToOclBuffer(img, queue);

        cl::Buffer dstBuff = ocvMatToOclBuffer(labeled, queue);
//        cl::Buffer dstBuff(context, CL_TRUE, dst_buff_size);

        std::cout << "running opencl watershed" << std::endl;
        //watershed(queue, watershedKernel, width, height, srcBuff, dstBuff);

        watershed(width, height, srcBuff, dstBuff);
        watershed(width, height, srcBuff, dstBuff);

        std::cout << "reading output (labels)" << std::endl;
        
        oclBufferToOcvMat(labeled, dstBuff, dst_buff_size, queue);
        
        std::cout << "reading output finished" << std::endl;
        

        float* data = (float*)labeled.data;

        //normalizing output

        float min = 0;
        float max = 0;

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                float val = data[i * width + j];

                if(min > val) min = val;
                if(max < val) max = val;
            }
        }

        cv::Mat grayOutput(height, width, CV_8UC1);

        unsigned char* outputData = (unsigned char *)grayOutput.data;

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                float val = (data[i * width + j] - min) / (max - min) * 255;

                outputData[i * width + j] = val;
            }
        }


        cv::imwrite(DATA_OUT("wateshed_test_out.png"), grayOutput);
    }
    catch (cl::Error err) {
        oclPrintError(err);
    }
}

