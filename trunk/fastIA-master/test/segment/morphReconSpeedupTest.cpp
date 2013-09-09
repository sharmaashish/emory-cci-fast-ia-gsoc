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
#include <stdio.h>



using namespace cv;
using namespace cv::gpu;
using namespace std;

#define ITER_NUM 1

BOOST_AUTO_TEST_CASE(morhReconSpeedupTest)
{
    uint64 total_time = 0;

    for(int i = 0; i < ITER_NUM; ++i)
    {
        Mat marker = imread(DATA_IN("microscopy/in-imrecon-gray-marker.png"), -1);
        Mat mask = imread(DATA_IN("microscopy/in-imrecon-gray-mask.png"), -1);

        Stream stream;
        GpuMat g_marker;
        GpuMat g_mask, g_recon;

        stream.enqueueUpload(marker, g_marker);
        stream.enqueueUpload(mask, g_mask);

        Mat markerInt, maskInt;
        marker.convertTo(markerInt, CV_32SC1, 1, 0);
        mask.convertTo(maskInt, CV_32SC1, 1, 0);

        GpuMat g_marker_int, g_mask_int;
        stream.enqueueUpload(markerInt, g_marker_int);
        stream.enqueueUpload(maskInt, g_mask_int);
        stream.waitForCompletion();

        uint64_t t1, t2;
        int numFirstPass = 2;
        int nBlocks = 14;

        std::cout << "morph recon start" << std::endl;

        t1 = cci::common::event::timestampInUS();
        g_recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(
                    g_marker, g_mask, 4, numFirstPass, stream, nBlocks);

        stream.waitForCompletion();
        t2 = cci::common::event::timestampInUS();

        uint64 exec_time = t2 - t1;
        total_time += exec_time;

        std::cout << "morph recon finished" << std::endl;
        std::cout << "morph recon speedup time: " << exec_time << "ms" << std::endl;

        Mat recon;

        g_recon.download(recon);
        imwrite(DATA_OUT("reconstruction_fast_out.png"), recon);
    }

    std::cout << "morph recon speedup, AVG time: "
              << total_time / ITER_NUM << "us" << std::endl;
}
