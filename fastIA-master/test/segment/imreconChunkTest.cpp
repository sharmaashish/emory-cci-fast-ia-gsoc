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

#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::gpu;


void runTest(const char* markerName, const char* maskName, bool binary, int w=-1, int b=0) {
	Mat marker, mask, recon;
	if (binary) marker = imread(markerName, 0) > 0;
	else marker = imread(markerName, 0);
	if (binary) mask = imread(maskName, 0) > 0;
	else mask = imread(maskName, 0);

//	std::cout << "testing with " << markerName << " + " << maskName << " " << (binary ? "binary" : "grayscale") << std::endl;
	std::cout << "hw,algo,binary,conn,chunk,border,time(us)" << std::endl;
	
	Stream stream;
	GpuMat g_marker, g_mask, g_recon;

	uint64_t t1, t2;
	Size s = marker.size();
;
	
/*	t1 = cci::common::event::timestampInUS();
	if (binary) recon = nscale::imreconstructBinary<unsigned char>(marker, mask, 4);
	else recon = nscale::imreconstruct<unsigned char>(marker, mask, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << "\tcpu recon 4-con took " << t2-t1 << "ms" << std::endl;
*/
	if (w == -1) {
		t1 = cci::common::event::timestampInUS();
		if (binary) recon = nscale::imreconstructBinary<unsigned char>(marker, mask, 8);
		else recon = nscale::imreconstruct<unsigned char>(marker, mask, 8);
		t2 = cci::common::event::timestampInUS();
		std::cout << "cpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<<s.width << ",0," << t2-t1 << std::endl;
	} else {
		t1 = cci::common::event::timestampInUS();
		for (int j = 0; j < s.height; j+=w) {
			for (int i = 0; i < s.height; i+=w) {
				uint64_t t3 = cci::common::event::timestampInUS();
		
				Range rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
				Range ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
				
				if (binary) recon = nscale::imreconstructBinary<unsigned char>(marker(rx, ry), mask(rx, ry), 8);
				else recon = nscale::imreconstruct<unsigned char>(marker(rx, ry), mask(rx, ry), 8);
				uint64_t t4 = cci::common::event::timestampInUS();

				//std::cout << "\t\tchunk "<< i << "," << j << " took " << t4-t3 << "ms " <<  std::endl;
                
                
                //std::ostringstream str_stream;
                //str_stream << "recon_i_" << i << "_j_" << j << ".png";
                //str_stream.flush();
                //std::string out_file_name = str_stream.str();
                //std::cout << "filename: " << out_file_name << std::endl;
                //imwrite(out_file_name, recon);
			}
		}
		t2 = cci::common::event::timestampInUS();
		std::cout << "cpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< w << "," << b << "," << t2-t1 << std::endl;


	}
	
/*	
	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstructUChar(marker, mask, 4);
	t2 = cci::common::event::timestampInUS();
	std::cout << "\tcpu reconUChar 4-con took " << t2-t1 << "ms" << std::endl;

	t1 = cci::common::event::timestampInUS();
	recon = nscale::imreconstructUChar(marker, mask, 8);
	t2 = cci::common::event::timestampInUS();
	std::cout << "\tcpu reconUChar 8-con took " << t2-t1 << "ms" << std::endl;
*/
	
		
	stream.enqueueUpload(marker, g_marker);
	stream.enqueueUpload(mask, g_mask);
	stream.waitForCompletion();
//	std::cout << "\tfinished uploading to GPU" << std::endl;

//	t1 = cci::common::event::timestampInUS();
//	g_recon = nscale::gpu::imreconstruct2<unsigned char>(g_marker, g_mask, 4, stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "\tgpu recon2 4-con took " << t2-t1 << "ms" << std::endl;
//	g_recon.release();
//
//	t1 = cci::common::event::timestampInUS();
//	g_recon = nscale::gpu::imreconstruct2<unsigned char>(g_marker, g_mask, 8, stream);
//	stream.waitForCompletion();
//	t2 = cci::common::event::timestampInUS();
//	std::cout << "\tgpu recon2 8-con took " << t2-t1 << "ms" << std::endl;
//	g_recon.release();
	
/*	t1 = cci::common::event::timestampInUS();
	if (binary) g_recon = nscale::gpu::imreconstructBinary<unsigned char>(g_marker, g_mask, 4, stream);
	else g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, 4, stream);
	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS();
	std::cout << "\tgpu recon 4-con took " << t2-t1 << "ms" << std::endl;
	g_recon.release();
*/
	if (w == -1) {
		t1 = cci::common::event::timestampInUS();
		if (binary) g_recon = nscale::gpu::imreconstructBinary<unsigned char>(g_marker, g_mask, 8, stream);
		else g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker, g_mask, 8, stream);
		stream.waitForCompletion();
		t2 = cci::common::event::timestampInUS();
		std::cout << "gpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< s.width <<",0,"<< t2-t1 << std::endl;
		g_recon.release();
	} else {
		s = g_marker.size();
		t1 = cci::common::event::timestampInUS();
		unsigned int iter;
		for (int j = 0; j < s.height; j+=w) {
			for (int i = 0; i < s.height; i+=w) {
				uint64_t t3 = cci::common::event::timestampInUS();
		
				Range rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
				Range ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
				
				if (binary) g_recon = nscale::gpu::imreconstructBinary<unsigned char>(g_marker(rx, ry), g_mask(rx, ry), 8, stream, iter);
				else g_recon = nscale::gpu::imreconstruct<unsigned char>(g_marker(rx, ry), g_mask(rx, ry), 8, stream, iter);
				stream.waitForCompletion();
				uint64_t t4 = cci::common::event::timestampInUS();

				g_recon.release();
				//std::cout << "\t\tchunk "<< i << "," << j << " took " << t4-t3 << "ms with " << iter << " iters" <<  std::endl;
			}
		}
		t2 = cci::common::event::timestampInUS();
		std::cout << "gpu,imrecon," << (binary ? "binary" : "grayscale") << ",8,"<< w << "," << b << "," << t2-t1 << std::endl;
	}
    
	g_marker.release();
	g_mask.release();	
}


BOOST_AUTO_TEST_CASE(imrecon_chunk_test_1)
{
    for (int w = 128; w < 4096; w = w* 2) {
		for (int b2 = 1; b2 <= 512 && b2 <= (w/2); b2 = b2 * 2) {

            const std::string in1 = DATA_IN("microscopy/in-imrecon-gray-marker.png");
            const std::string in2 = DATA_IN("microscopy/in-imrecon-gray-mask.png");

            runTest(in1.c_str(), in2.c_str(), false, w, b2/2);
		}
	}
}

