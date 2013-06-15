#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include <stdio.h>
//#include "PixelOperations.h"
#include "UtilsCVImageIO.h"
#include "ConnComponents.h"

#include "opencv2/opencv.hpp"

#if defined (WITH_CUDA)
#include "opencv2/gpu/stream_accessor.hpp"

#include "opencv2/gpu/gpu.hpp"
#include "ccl_uf.cuh"
#endif

using namespace cv;

BOOST_AUTO_TEST_CASE(test1)
{
    BOOST_CHECK(2 == 2);
    
    std::cout << RESOURCE("blabla.jpg") << std::endl;
    
    Mat img = imread(RESOURCE("text.png"));
    
    std::cout << img.size();
    
    
//    uint64_t t1, t2;
//    std::vector<int> stages;
//    for (int stage = 0; stage <= 200; ++stage) {
//        stages.push_back(stage);
//    }
    

//    std::string prefix;
//    prefix.assign("test/out-bwareaopen-test-");
//    std::string suffix;
//    suffix.assign(".pbm");
//    ::cciutils::cv::IntermediateResultHandler *iwrite = new ::cciutils::cv::IntermediateResultWriter(prefix, suffix, stages);

}

