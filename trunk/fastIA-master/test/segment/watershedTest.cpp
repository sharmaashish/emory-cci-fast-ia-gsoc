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
#include "PixelOperations.h"
#include "NeighborOperations.h"

#include "Logger.h"
#include <stdio.h>


#if defined (WITH_CUDA)
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/stream_accessor.hpp"
#endif

using namespace cv;
using namespace cv::gpu;

BOOST_AUTO_TEST_CASE(watershed_test_1)
{   
    std::vector<std::string> segfiles;
    segfiles.push_back(DATA_IN("tissue.png"));
//	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/astroII.1/astroII.1.ndpi-0000008192-0000008192-15.mask.pbm"));
//	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/gbm2.1/gbm2.1.ndpi-0000004096-0000004096-15.mask.pbm"));
//	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/normal.3/normal.3.ndpi-0000028672-0000012288-15.mask.pbm"));
//	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/oligoastroIII.1/oligoastroIII.1.ndpi-0000053248-0000008192-15.mask.pbm"));
//	segfiles.push_back(std::string("/home/tcpan/PhD/path/Data/seg-validate-cpu/oligoIII.1/oligoIII.1.ndpi-0000012288-0000028672-15.mask.pbm"));

	std::vector<std::string> imgfiles;
    
    imgfiles.push_back(DATA_IN("tissue.png"));
//	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1/astroII.1.ndpi-0000008192-0000008192.tif"));
//	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/gbm2.1/gbm2.1.ndpi-0000004096-0000004096.tif"));
//	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/normal.3/normal.3.ndpi-0000028672-0000012288.tif"));
//	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/oligoastroIII.1/oligoastroIII.1.ndpi-0000053248-0000008192.tif"));
//	imgfiles.push_back(std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/oligoIII.1/oligoIII.1.ndpi-0000012288-0000028672.tif"));

	for (int i = 0; i < segfiles.size(); ++i ) {
    
        printf("testing file : %s\n", segfiles[i].c_str());
        Mat seg_big = imread(segfiles[i].c_str(), -1);
        Mat img = imread(imgfiles[i].c_str(), -1);
        // original
        Stream stream;
    
        uint64_t t1, t2;
    
        // distance transform:  matlab code is doing this:
        // invert the image so nuclei candidates are holes
        // compute the distance (distance of nuclei pixels to background)
        // negate the distance.  so now background is still 0, but nuclei pixels have negative distances
        // set background to -inf
    
        // really just want the distance map.  CV computes distance to 0.
        // background is 0 in output.
        // then invert to create basins
        Mat dist(seg_big.size(), CV_32FC1);
    
        // opencv: compute the distance to nearest zero
        // matlab: compute the distance to the nearest non-zero
        distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
        double mmin, mmax;
        minMaxLoc(dist, &mmin, &mmax);
    
        // invert and shift (make sure it's still positive)
        //dist = (mmax + 1.0) - dist;
        dist = - dist;  // appears to work better this way.
    
    //	cciutils::cv::imwriteRaw("test/out-dist", dist);
    
        // then set the background to -inf and do imhmin
        //Mat distance = Mat::zeros(dist.size(), dist.type());
        // appears to work better with -inf as background
        Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
        dist.copyTo(distance, seg_big);
    //	cciutils::cv::imwriteRaw("test/out-distance", distance);
    
    
        // then do imhmin. (prevents small regions inside bigger regions)
        Mat distance2 = nscale::imhmin<float>(distance, 1.0f, 8);
    
    //cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);
    
    
        /*
         *
            seg_big(watershed(distance2)==0) = 0;
            seg_nonoverlap = seg_big;
         *
         */
    
            Mat minima = nscale::localMinima<float>(distance2, 8);
            // watershed is sensitive to label values.  need to relabel.
            std::vector<Vec4i> dummy;
            Mat_<int> labels = nscale::bwlabel(minima, false, 8, false, dummy);
    
        Mat_<int> labels2 = nscale::bwlabel2(minima, 8, true);
    
        
    
    
        Mat nuclei = Mat::zeros(img.size(), img.type());
    //	Mat distance3 = distance2 + (mmax + 1.0);
    //	Mat dist4 = Mat::zeros(distance3.size(), distance3.type());
    //	distance3.copyTo(dist4, seg_big);
    //	Mat dist5(dist4.size(), CV_8U);
    //	dist4.convertTo(dist5, CV_8U, (std::numeric_limits<unsigned char>::max() / mmax));
    //	cvtColor(dist5, nuclei, CV_GRAY2BGR);
        img.copyTo(nuclei, seg_big);
    
        t1 = cci::common::event::timestampInUS();
    
        // watershed in openCV requires labels.  input foreground > 0, 0 is background
        // critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
        Mat watermask = nscale::watershed(nuclei, distance2, 8);
    //	cciutils::cv::imwriteRaw("test/out-watershed", watermask);
    
        t2 = cci::common::event::timestampInUS();
        std::cout << "cpu watershed loop took " << t2-t1 << "ms" << std::endl;
        double mn, mx;
        minMaxLoc(watermask, &mn, &mx);
        watermask = (watermask - mn) * (255.0 / (mx-mn));
    
        imwrite("test/out-cpu-watershed-oligoIII.1-1.png", watermask);
    
        t1 = cci::common::event::timestampInUS();
    
        // watershed in openCV requires labels.  input foreground > 0, 0 is background
        // critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
        watermask = nscale::watershed2(nuclei, distance2, 8);
    //	cciutils::cv::imwriteRaw("test/out-watershed", watermask);
    
        t2 = cci::common::event::timestampInUS();
        std::cout << "cpu watershed2 loop took " << t2-t1 << "ms" << std::endl;
    
        // cpu version of watershed.
        mn, mx;
        minMaxLoc(watermask, &mn, &mx);
        watermask = (watermask - mn) * (255.0 / (mx-mn));
    
        imwrite("test/out-cpu-watershed-oligoIII.1-2.png", watermask);
        dist.release();
        distance.release();
        watermask.release();
    
    
    #if defined (WITH_CUDA)
        // gpu version of watershed
        //Stream stream;
        GpuMat g_distance2, g_watermask, g_seg_big;
        stream.enqueueUpload(distance2, g_distance2);
        stream.enqueueUpload(seg_big, g_seg_big);
        stream.waitForCompletion();
        std::cout << "finished uploading" << std::endl;
    
        t1 = cci::common::event::timestampInUS();
        g_watermask = nscale::gpu::watershedDW(g_seg_big, g_distance2, -1, 8, stream);
        stream.waitForCompletion();
        t2 = cci::common::event::timestampInUS();
        std::cout << "gpu watershed DW loop took " << t2-t1 << "ms" << std::endl;
    
        Mat temp(g_watermask.size(), g_watermask.type());
        stream.enqueueDownload(g_watermask, temp);
        stream.waitForCompletion();
        minMaxLoc(temp, &mn, &mx);
        printf("masked:  min = %f, max = %f\n", mn, mx);
        //temp = nscale::PixelOperations::mod<int>(temp, 256);
        temp = (temp - mn) * (255.0 / (mx-mn));
        imwrite("test/out-gpu-watershed-oligoIII.1.png", temp);
    
    
    
        printf("watermask size: %d %d,  type %d\n", g_watermask.rows, g_watermask.cols, g_watermask.type());
    //	printf("g_border size: %d %d,  type %d\n", g_border.rows, g_border.cols, g_border.type());
    //	Mat watermask2(g_border.size(), g_border.type());
    //	stream.enqueueDownload(g_watermask, watermask2);
    //	stream.waitForCompletion();
    //	printf("here\n");
    
        g_watermask.release();
        g_distance2.release();
        g_seg_big.release();
        
    #endif
    
        seg_big.release();
        img.release();
	}
}

