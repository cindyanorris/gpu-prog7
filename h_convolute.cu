#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "h_convolute.h"

//prototype for function local to this file
void convoluteOnCPU(float* h_result, float * h_mask, float * h_vector,
                    int maskLen, int vectorLen);

/*  h_convolute
    This function returns the amount of time it takes to perform
    a vector convolution on the CPU.
    Inputs:
    h_result - points to the vector to hold the result
    h_mask - points to the mask to use in the convolution
    h_vector - points to the vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector to convolute

    returns the amount of time it takes to perform the
    convolution
*/
float h_convolute(float* h_result, float * h_mask, float * h_vector,
                  int maskLen, int vectorLen)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use CUDA functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));  
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));   
    
    //call function that does the actual work
    convoluteOnCPU(h_result, h_mask, h_vector, maskLen, vectorLen);
   
    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu)); 

    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*  h_convolute
    This function performs the vector convolution on the CPU.  
    Inputs:
    h_result - points to the vector to hold the result
    h_mask - points to the mask to use in the convolution
    h_vector - points to the vector to convolute
    maskLen - length of the mask
    vectorLen - length of the vector to convolute

    modifies the h_result vector
*/
void convoluteOnCPU(float* h_result, float * h_mask, float * h_vector,
                    int maskLen, int vectorLen)
{
    int i, j, vStartIdx, vIdx;
    float cvalue;

    int midMaskIdx = maskLen/2;
    for (i = 0; i < vectorLen; i++)
    {
        cvalue = 0; 
        //starting index into vector is i - (1/2 maskLen)
        vStartIdx = i - midMaskIdx; 
        for (j = 0; j < maskLen; j++)
        {
            vIdx = vStartIdx + j;
            //don't access h_vector if vIdx is out of bounds
            if (vIdx >= 0 && vIdx < vectorLen)
                cvalue += h_vector[vIdx] * h_mask[j];
        }
        h_result[i] = cvalue;
    }
}
