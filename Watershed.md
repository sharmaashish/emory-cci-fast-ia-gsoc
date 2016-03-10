# Introduction #

Watershed algorithm summary.

OpenCL implementation based on:
https://github.com/louismullie/watershed-cuda


# Implementation #

Watershed algorithm can be found in /segment/opencl.

Opencl kernels are located at /segment/opencl/kernels/watershed.cl


```

void watershed(int width, int height,
               cl::Buffer& src,
               cl::Buffer& labeled,
               ProgramCache& cache = ProgramCache::getGlobalInstance(),
               cl::CommandQueue queue = ProgramCache::getGlobalInstance().getDefaultCommandQueue());

```

# Example #

As an example, we have a picture from the tomography.
Original image was preprocessed with gaussian blur.

Input to the algorithm is as follows:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/watershed_test.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/watershed_test.png)

The raw watershed output is not clearly readable due to small differences in color.

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out.png)

Better method to show result is to draw edges between adjacent basins:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out_edges.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out_edges.png)

Edges combined with original input data:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out_combined.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/other/out/watershed_test_out_combined.png)

# Performance test #

The testing dataset contains images of different sizes. Additionally, for each size there are three variants of image filling (20%, 50% and 100%).
Original microscopy data were preprocess with gaussian blur with radius=10.

Original and preprocessed data are located in test\_data directory.


| | 128 (20%) | 128 (50%) | 128 (100%) | 256 (20%) | 256 (50%) | 256 (100%) | 512 (20%) | 512 (50%) | 512 (100%) | 1024 (20%) | 1024 (50%) | 1024 (100%) | 2048 (20%) | 2048 (50%) | 2048 (90%) | 4096 (50%) |
|:|:----------|:----------|:-----------|:----------|:----------|:-----------|:----------|:----------|:-----------|:-----------|:-----------|:------------|:-----------|:-----------|:-----------|:-----------|
| **Intel Core 2 Duo 2.5GHz** | 98        | 54        | 20         | 505       | 351       | 51         | 3520      | 1912      | 137        | 32379      | 20577      | 608         | 228960     | 157905     | 64873      | 1341238    |
| **GeForce GT 430** | 26        | 18        | 6          | 111       | 77        | 11         | 849       | 483       | 34         | 7997       | 5077       | 129         | 70199      | 49283      | 14049      | 474620     |
| **Tesla C2070** | 18        | 12        | 4          | 41        | 29        | 7          | 231       | 123       | 13         | 1607       | 1037       | 39          | 13035      | 9230       | 2705       | 85839      |

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/chart.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/chart.png)

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/chart_2.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/chart_2.png)

<a href='Hidden comment: 
Sample test data (256 - 20%, 256 - 50%, 256 - 100%):
'></a>

Sample test file (size: 256, fill: 100%)

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_10x10/in-imrecon-gray-mask-size_256-fill_100.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_10x10/in-imrecon-gray-mask-size_256-fill_100.png)

Results (raw, edges, combined):

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out.png)
![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out_edges.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out_edges.png)
![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out_combined.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_256-fill_100_out_combined.png)

<a href='Hidden comment: 
http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_5x5/in-imrecon-gray-mask-size_256-fill_25.png http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_5x5/in-imrecon-gray-mask-size_256-fill_50.png http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_5x5/in-imrecon-gray-mask-size_256-fill_100.png
'></a>

## Why less filled pictures take more time? ##

When image contains big areas of background, algorithm creates big basins. Big basins are inefficient in parallel watershed - they require a lot of iterations to propagate through image.

As an example, lets take a picture:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_10x10/in-imrecon-gray-mask-size_1024-fill_50.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/test_data/watershed/gaussian_blur_10x10/in-imrecon-gray-mask-size_1024-fill_50.png)

Combined output from watershed:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_1024-fill_50_out_combined.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_1024-fill_50_out_combined.png)

Raw watershed output with the biggest basins marked:

![http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_1024-fill_50_out_big_basin_example.png](http://emory-cci-fast-ia-gsoc.googlecode.com/svn/resources/_watershed_gaussian_blur_10x10_in-imrecon-gray-mask-size_1024-fill_50_out_big_basin_example.png)

## Suggested solution ##

The current version of the algorithm works without any mask.
Providing special mask to omit background points should fix the problem.
User would have to provide the mask obtained from some kind of preprocessing (e.g. simple thresholding).

## Another issue: oversegmentation ##
The result of the algorithm is heavily dependent on pre-processing methods. Perhaps other methods (e.g. diffusion instead of gaussian blur) would yield better results.

# Difficulties #
Memory violation within the kernel can cause hard to find bugs.
Some devices works fine and bug stay hidden. After running on another device, some unspecific errors may happen. Example: kernel tested on Nvidia GPU worked fine, but moved to CPU periodically caused segfaults in cv::Mat destructor.
Good practice is to test OpenCL kernel on different platforms and devices.

# Reference #

[1](http://www.fem.unicamp.br/~labaki/Academic/cilamce2009/1820-1136-1-RV.pdf) Vitor B, Körbes A. Fast image segmentation by watershed transform on graphical hardware. In: Proceedings of the 17th International Conference on Systems, Signals and Image Processing, pp. 376-379, Rio de Janeiro, Brazil.