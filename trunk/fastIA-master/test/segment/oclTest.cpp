#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <opencv2/ocl/private/util.hpp>

//#include <CL/cl.
#include <CL/cl.hpp>

#include "TestUtils.h"

using namespace cv;

int main(int argc, char* argv[]){
    
    //int ocl::getDevice(std::vector<Info>& oclinfo, int devicetype=CVCL_DEVICE_TYPE_GPU )
   
    std::vector<ocl::Info> deviceVector;
    
    ocl::getDevice(deviceVector);
    
    for(int i = 0; i < deviceVector.size(); ++i){
        std::cout << deviceVector[i].DeviceName[0] << std::endl;
    }
    
       
    Mat image = imread(DATA_IN("coins.png"), 1);
    
    std::cout << image.size() << std::endl;
    
    ocl::oclMat ocl_input(image);
    ocl::oclMat ocl_output;
    
    std::cout << ocl_input.type() << std::endl;
    std::cout << ocl_input.ocltype() << std::endl;
    
    //CV_8UC1
    
    ocl::add(ocl_input, ocl_input, ocl_output);
    //ocl::threshold(ocl_input, ocl_output, 0.5, 100);
    
    Mat out_img(ocl_output);
    
    imwrite(DATA_OUT("oclTest.png"), out_img);
     
    return 0;
}
