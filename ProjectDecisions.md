  1. **The way of storing OpenCL’s kernels code.** <br /> OpenCL’s kernels are visible in a project tree as as separete files with .cl extension. However, they are not loaded from files in runtime, when algorithms are invoked. There is cmake script (cmake\_modules/cl2cpp.cmake) that automatically converts all kernels to single file (kernels.cpp). Each kernel is transformed to global c-string. After that, kernels.cpp is compiled like another source files. <br />
  1. **Use C++ binding to OpenCL (cl.hpp) instead of standard C interface (cl.h).** <br /> All the library is written in C++, so using C++ bindings seems to be obvious decision. C++ bindings are provided by OpenCL’s creators (http://www.khronos.org) and it is a part of a standard. I have included the source of this binding to project manually (external/CL). It’s necessary as some vendors provide only C interface without C++ bindings. These bindings internally include cl.h and automatically detect OpenCL version.<br />
  1. **Use CTest and Boost Test for testing purposes.** <br />CTest, part of CMake, provides good mechanism to define and run unit tests. However CTest is totally code-independant and doesn’t provide any utilities for asserting and verifying inside code. It is the reason why Boost Test is used.<br />
  1. **OpenCV and OpenCL integration.** <br /> We decided that it’s not important at this moment to worry about good integration.  Specifically, this is due to OpenCV OpenCL integration being at its early stage.
  1. **OpenCL algorithms interface.**<br /> An example of OpenCL algorithm declaration
```
void threshold(cl::CommandQueue& queue, cl::Kernel& kernel,
      int width, int height, cl::Buffer& src, int src_pitch, cl::Buffer& dst, int dst_pitch,
      unsigned char lower, unsigned char upper,
      bool lower_inclusive, bool upper_inclusive);

```
  1. All types come from OpenCL C++ bindings api. Queue is an equivalent to Stream from Cuda. Buffers represents data already allocated on the device.<br /> At this moment kernels for particular algorithms are given from outside. I allows to be sure that we have already compiled kernel and algorithm caller function don’t have to do that. On the other hand it is an disadvantage to bother user to do that. I suppose that finally algorithms should use internal kernel cache. <br />
  1. **OpenCL version - 1.1** <br /> At this moment not all main vendors implement the newest standard (1.2). Nvidia uses version 1.1. If we want to run algorithms on Nvidia’s GPUs, we have to use 1.1 version.
  1. **Templating OpenCL kernels** <br />There is no support for templates in 1.1 version of OpenCL standard. However some simple mechanism for parameterizing  kernels would be useful. I will do some research to find out possible solutions.
  1. **Time measurement, efficiency comparison between OpenCL / CUDA**