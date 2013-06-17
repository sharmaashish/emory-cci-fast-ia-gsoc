#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "TestUtils.h"

#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"

#include <iostream>
#include <iomanip>

using namespace cv;

BOOST_AUTO_TEST_CASE(dist_transform_test_1)
{   
    
    Mat input = imread(DATA_IN("cell_binary_mask.png"), -1);
	if(input.data == NULL){
        BOOST_FAIL("Failed reading");
	}
//	std::cout << "input - " << (int) input.ptr(10)[20] << std::endl;

    int zoomFactor = 1; 
    
    if(zoomFactor > 1){
        Mat tempMask = Mat::zeros((input.cols*zoomFactor) ,(input.rows*zoomFactor), input.type());
        for(int x = 0; x < zoomFactor; x++){
            for(int y = 0; y <zoomFactor; y++){
                Mat roiMask(tempMask, cv::Rect((input.cols*x), input.rows*y, input.cols, input.rows ));
                input.copyTo(roiMask);
            }
        }
        input = tempMask;
    }

	
//	gpu::setDevice(2);
	Mat point(10,10, CV_8UC1);
	point.ones(10,10, CV_8UC1);

	for(int x = 0; x < point.rows; x++){
		uchar* ptr = point.ptr(x);
		for(int y = 0; y < point.cols; y++){
			ptr[y] = 1;
			if(x==1 && y==3){
				ptr[y] = 0;
			}
//			if(x==9 && y==9){
//				ptr[y] = 0;
//			}
//			if(x==6 && y==1){
//				ptr[y] = 0;
//			}
//			std::cout << (int) ptr[y] <<" ";
		}
//		std::cout<<std::endl;
	}
//	uchar *ptr = point.ptr(1);
//	ptr[3] = 0;

	Mat dist(point.size(), CV_32FC1);


	uint64_t t1 = cci::common::event::timestampInUS();
	distanceTransform(input, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "distTransf CPU  took " << t2-t1 <<" ms"<<std::endl;
    
    imwrite(DATA_OUT("cell_binary_mask_dst_trans_cpu_out.png"), dist);
    
	dist.release();

	t1 = cci::common::event::timestampInUS();
	Mat queueBasedDist = nscale::distanceTransform(input);
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTranf CPU queue took "<< t2-t1 << " ms" << std::endl;
	queueBasedDist.release();


#if defined (WITH_CUDA)
	GpuMat g_warm(input);
	g_warm.release();
#endif

	t1 = cci::common::event::timestampInUS();
	Mat queueBasedTiled = nscale::distanceTransformParallelTile(input,4096, 8);
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTranf CPU queue tiled took "<< t2-t1 << " ms" << std::endl;
	queueBasedTiled.release();
//	for(int x = 0; x < queueBasedDist.rows; x++){
//		float* ptr = queueBasedTiled.ptr<float>(x);
//		for(int y = 0; y < queueBasedTiled.cols; y++){
//			std::cout << std::setprecision(2) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}
//
#if defined (WITH_CUDA)
	t1 = cci::common::event::timestampInUS();
	GpuMat g_mask(input);
	t2 = cci::common::event::timestampInUS();
	std::cout << "upload:"<< t2-t1 << std::endl;
	Stream stream;

	t1 = cci::common::event::timestampInUS();
	GpuMat g_distance = nscale::gpu::distanceTransform(g_mask, stream);

	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTransf GPU  took " << t2-t1 <<" ms"<<std::endl;

	t1 = cci::common::event::timestampInUS();
	Mat h_distance(g_distance);
	t2 = cci::common::event::timestampInUS();
    
    imwrite(DATA_OUT("cell_binary_mask_dst_trans_gpu_out.png"), h_distance);

	std::cout << "download:"<< t2-t1 << std::endl;
#endif

//	return 0;
}

