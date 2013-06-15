#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;

BOOST_AUTO_TEST_CASE(test1)
{   
    std::cout << DATA_IN("blabla.jpg") << std::endl;
    
    Mat img = imread(DATA_IN("text.png"));
    
    BOOST_CHECK(img.data);
    
    imwrite(DATA_OUT("text.png"), img);
}

