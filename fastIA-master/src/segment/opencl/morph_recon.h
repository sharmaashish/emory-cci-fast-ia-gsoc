#ifndef MORPH_RECON_H
#define MORPH_RECON_H

#include <CL/cl.hpp>
#include "utils/ocl_program_cache.h"

void morphReconInit(cl::Buffer seends, cl::Buffer image,
                    int width, int height,
                    ProgramCache& cache = ProgramCache::getGlobalInstance(),
                    cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                   .getDefaultCommandQueue());

void morphRecon(cl::Buffer input_list, int dataElements,
                cl::Buffer seeds,cl::Buffer image, int ncols, int nrows,
                ProgramCache& cache = ProgramCache::getGlobalInstance(),
                cl::CommandQueue& queue = ProgramCache::getGlobalInstance()
                                                    .getDefaultCommandQueue());

#endif // MORPH_RECON_H
