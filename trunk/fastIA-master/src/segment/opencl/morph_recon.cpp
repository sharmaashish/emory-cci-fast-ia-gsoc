#include "morph_recon.h"


#include <string>
#include <sstream>




//void morphReconInitScan(cl::Buffer marker, cl::Buffer mask,
//                        int width, int height,
//                        cl::CommandQueue &queue, cl::Program &program)
//{

//}


//void morphReconInitScan(cl::Buffer marker, cl::Buffer mask,
//                        int width, int height,
//                        ProgramCache &cache, cl::CommandQueue &queue)
//{

//}



//void morphReconInitQueue(cl::Buffer marker, cl::Buffer mask,
//                    cl::Buffer queueData,
//                    int width, int height, int& queue_size,
//                    cl::CommandQueue &queue, cl::Program &program)
//{

//}


//void morphReconInitQueue(cl::Buffer marker, cl::Buffer mask,
//                         cl::Buffer queueData, int width, int height,
//                         int& queue_size,
//                         ProgramCache &cache, cl::CommandQueue &queue)
//{

//}


//void morphReconQueuePropagation(cl::Buffer inputQueueData, int dataElements,
//                int queueSize,
//                cl::Buffer seeds, cl::Buffer image, int ncols, int nrows,
//                cl::CommandQueue &queue, cl::Program &program)
//{
//}


//void morphReconQueuePropagation(cl::Buffer inputQueueData, int dataElements,
//                int queueSize,
//                cl::Buffer seeds, cl::Buffer image, int ncols, int nrows,
//                ProgramCache &cache, cl::CommandQueue &queue)
//{
// //   std::cout << "parallel queue-based morphological reconstruction "
//   //              "ocl program params: " << program_params << std::endl;


//}


//void morphRecon(cl::Buffer marker, cl::Buffer mask,
//                const std::string& buffers_types, int width, int height,
//                ProgramCache &cache, cl::CommandQueue &queue)
//{
// //   std::cout << "parallel queue-based morphological reconstruction "
//   //              "ocl program params: " << program_params << std::endl;

//    std::vector<std::string> sources;
//    sources.push_back("ParallelQueue");
//    sources.push_back("MorphRecon");

//    cl::Program& program = cache.getProgram(sources, morphReconParams);


//    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

//    morphReconInitScan(marker, mask, width, height, queue, program);

////    int queue_total_size = width * height;
////    int queue_size;

////    cl::Buffer device_queue(context, CL_TRUE, sizeof(int) * queue_total_size);

////    morphReconInitQueue(marker, mask, device_queue, width, height, queue_size,
////                        queue, program);

////    morphReconQueuePropagation(device_queue, queue_size, queue_total_size,
////                               marker, mask, width, height, queue, program);
//}
