#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
//config.h defines the number of threads per block 
//and the maximum mask size (less than threads per block)
//and the constants: GLOBAL, SHARED, CONSTANT
//that indicate which kernel to launch
#include "config.h" 
#include "d_convolute.h"

//prototypes for kernels in this file
__global__ 
void d_convoluteGlobalKernel(float * d_result, float * d_mask, 
                             float * vector, int maskLen, int vectorLen);

__global__ 
void d_convoluteSharedKernel(float * d_result, float * d_mask, 
                             float * vector, int maskLen, int vectorLen);

__global__ 
void d_convoluteConstantKernel(float * d_result, float * vector, 
                               int maskLen, int vectorLen);

__constant__ float constmask[MAXMASKLEN];

/*  d_convolute
    This function prepares and invokes a kernel to perform
    a vector convolution on the GPU.  
    Inputs:
    result - points to the vector to hold the result
    mask - points to the mask to use in the convolution
    vector - points to the vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector to convolute
    which - indicates which kernel to use (GLOBAL, SHARED, CONSTANT)
*/
float d_convolute(float * result, float * mask, float * vector, 
                  int maskLen, int vectorLen, int which)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

/*
    //your code goes here
    //THREADSPERBLOCK is defined in config.h

    if (which == GLOBAL)

    else if (which == SHARED)

    else if (which == CONSTANT)

*/

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

/*  
    d_convoluteGlobalKernel
    Kernel code for convolution.  This code accesses both
    the mask and the vector out of global memory.
    Inputs:
    d_result - pointer to the array in the global memory to hold the result
    d_mask - pointer to the array in the global memory that holds the mask
    d_vector - pointer to the array in the global memory that holds the 
               vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector
*/
__global__ void d_convoluteGlobalKernel(float * d_result, float * d_mask,
                                        float * d_vector, int maskLen, int vectorLen) 
{
}      

/*  
    d_convoluteSharedKernel
    Kernel code for convolution.  This code loads both
    the mask and the vector into shared memory before performing
    the convolution.
    Inputs:
    d_result - pointer to the array in the global memory to hold the result
    d_mask - pointer to the array in the global memory that holds the mask
    d_vector - pointer to the array in the global memory that holds the 
               vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector
*/
__global__ 
void d_convoluteSharedKernel(float * d_result, float * d_mask,
                             float * d_vector, int maskLen, int vectorLen) 
{
    //MAXMASKLEN and THREADSPERBLOCK are defined in config.h
    //you'll use both of those in this code
}      

/*  
    d_convoluteConstantKernel
    Kernel code for convolution.  This code 
    loads the vector into shared memory before performing the convolution.
    The mask should have been loaded into constant memory before the
    kernel launch.
    Inputs:
    d_result - pointer to the array in the global memory to hold the result
    d_mask - pointer to the array in the global memory that holds the mask
    d_vector - pointer to the array in the global memory that holds the 
               vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector
*/
__global__ 
void d_convoluteConstantKernel(float * d_result, float * d_vector, 
                               int maskLen, int vectorLen) 
{
    //THREADSPERBLOCK is defined in config.h

}      

