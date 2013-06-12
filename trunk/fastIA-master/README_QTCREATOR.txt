cmake options to compile full-features version of the library:

-DNS_SEGMENT=true -DNS_FEATURES=true -DUSE_CUDA=true -DUSE_OPENMP=true -DOpenCV_CMAKE_DIR=/home/michalc/GSOC/opencv-2.4.5_build -DOpenCV_DIR=/home/michalc/GSOC/opencv-2.4.5_build -DCMAKE_BUILD_TYPE=Debug

tested under Qt Creator 2.4.0 (works very well)
