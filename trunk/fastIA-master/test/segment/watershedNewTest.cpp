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

//#define INPUT_FILE "watershed_test.png"
#define INPUT_FILE "microscopy/in-imrecon-gray-mask_blurred.png"
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


        cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue();

        cv::Mat img = cv::imread(DATA_IN(INPUT_FILE), CV_LOAD_IMAGE_GRAYSCALE);

        img.convertTo(img, CV_8UC1);

        //cv::imwrite(DATA_OUT("coins_1b.png"), img);

        cv::Size img_size = img.size();
        int width = img_size.width;
        int height = img_size.height;
        int dst_buff_size = width * height * sizeof(float);

        cv::Mat labeled = cv::imread(DATA_IN(INPUT_FILE), CV_LOAD_IMAGE_GRAYSCALE);
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

        float total_time = getLastExecutionTime();

        std::cout << "Execution time in milliseconds = " << std::fixed << std::setprecision(3)
                  << total_time << " ms" << std::endl;

        watershed(width, height, srcBuff, dstBuff);

        total_time = getLastExecutionTime();

        std::cout << "Execution time in milliseconds = " << std::fixed << std::setprecision(3)
                  << total_time << " ms" << std::endl;

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
        cv::Mat edgesOutput(height, width, CV_8UC1, cv::Scalar(0));

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                float val = (data[i * width + j] - min) / (max - min) * 255;

                grayOutput.data[i * width + j] = val;
            }
        }
        
        cv::imwrite(DATA_OUT("wateshed_test_out.png"), grayOutput);
        
        
        cv::Mat combinedOutput(img);
        combinedOutput.convertTo(combinedOutput, CV_8UC1);

        const char neighbourhood_x[] = {-1, 0, 1, 1, 1, 0,-1,-1};
        const char neighbourhood_y[] = {-1,-1,-1, 0, 1, 1, 1, 0};
        
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                
                int central_idx = i * width + j;
                
                unsigned char central_val = grayOutput.data[central_idx];
                
                for(int k = 0; k < 8; ++k){
                    int x_offset = neighbourhood_x[k];
                    int y_offset = neighbourhood_y[k];
                    
                    int x = j + x_offset;
                    int y = i + y_offset;
                    
                    if(x < 0 || x >= width || y < 0 || y >= height)
                        continue;
                    
                    unsigned char neighbour_val = grayOutput.data[y*width + x];
                    
                    if(neighbour_val < central_val)
                    {
                        edgesOutput.data[central_idx] = 255;
                        combinedOutput.data[central_idx] = 255;
                    }
                }               
            }
        }
        
        cv::imwrite(DATA_OUT("wateshed_test_out_edges.png"), edgesOutput);
        cv::imwrite(DATA_OUT("wateshed_test_out_combined.png"), combinedOutput);
        
    }
    catch (cl::Error err) {
        oclPrintError(err);
    }
}

