#ifndef CUDA_ERRORS_H
#define CUDA_ERRORS_H

#include <stdio.h>

#define checkError( err )   _checkError( (err) , __FILE__, __LINE__ )
#define lastError()         _lastError( __FILE__, __LINE__ )
#define syncErrorCheck()    _syncErrorCheck( __FILE__, __LINE__ )

inline void _checkError(cudaError_t err, const char *file, int line)
{
    if( err!=cudaSuccess)
    {
        fprintf(stderr, "failed! %s, at %s, line %d\n", cudaGetErrorString( err ), file, line);
        exit(-1);
    }
}

inline void _lastError(const char *file,int line)
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "failed! at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

inline void _syncErrorCheck(const char* file, int line)
{
    cudaError err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif //CUDA_ERRORS_H

