# Introduction #

MR performance comparison for OpenCL, Nvidia Cuda, OpenMP and standard serial CPU implementations.


# Details #

## OpenCL/CUDA on GPU ##

| | OPENCL | NVIDIA CUDA | **ratio** |
|:|:-------|:------------|:----------|
| GeForce GT 460 | 1s 380ms | 1s 228ms    | 0,89      |
| GeForce GT 430 | 4s 912ms | 4s 223ms    | 0,86      |
| Tesla C2070 | 1s 146ms | 0s 897ms    | 0,79      |
| AMD FirePro™ W9000 | 2s 652ms |             |           |

## OpenCL/serial CPU/OpenMP on CPU ##

| | OPENCL | CPU SERIAL | CPU MULTICORE | **ratio** |
|:|:-------|:-----------|:--------------|:----------|
| Intel Core Duo T9300 2.5GHz | 50s 495ms | 7s 113ms   | 4s 134ms      | 0,082     |
| Intel Xeon E5649 2.5GHz | 11s 385ms | 5s 340ms   | 5s 480ms      | 0,47      |

### Possible reasons of poor performance for OpenCL implementation using CPU ###
  * Using local queues per work-item and moving them to higher level queue using prefix sum is probably a substantial reason. On CPU architectures every work-group is executed by single thread, from beginning to the end. It cause that there is no concurrency between work-times in work-group. For CPU version, local per-work-item queues should be replaced by per-work-group queues. Each work-item could add data to local queue by simply increasing shared index.
  * OpenCL version was tested using AMD's Ocl implementations on Intel CPU. I may cause that code is not the best optimized.

### Possible solutions ###
  * Separate MR Ocl implementation for gpu and cpu version.



## Test data ##

Data used are located in test\_data directory on svn repositry.

The last part of the marker file name ('eroded\_Nx') means how many times
erosion was performed on original image (mask).

Marker - mask pairs used for testing:

  * mr\_tests/in-imrecon-gray-marker.png
  * mr\_tests/gbm2.1.ndpi-0000004096-0000004096\_inv\_eroded\_4x.png
  * mr\_tests/normal.3.ndpi-0000028672-0000012288\_inv\_eroded\_3x.png
  * mr\_tests/oligoastroIII.1.ndpi-0000053248-0000008192\_inv\_eroded\_3x.png
  * mr\_tests/oligoIII.1.ndpi-0000012288-0000028672\_inv\_eroded\_3x.png

  * mr\_tests/in-imrecon-gray-mask.png
  * mr\_tests/gbm2.1.ndpi-0000004096-0000004096\_inv.png
  * mr\_tests/normal.3.ndpi-0000028672-0000012288\_inv.png
  * mr\_tests/oligoastroIII.1.ndpi-0000053248-0000008192\_inv.png
  * mr\_tests/oligoIII.1.ndpi-0000012288-0000028672\_inv.png