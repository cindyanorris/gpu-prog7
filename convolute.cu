#include <stdio.h>
#include <stdlib.h>
#include "h_convolute.h"
#include "d_convolute.h"
#include "wrappers.h"
//config.h defines the number of threads per block, the maximum size
//for the mask and the constants: GLOBAL, SHARED, CONSTANT that 
//indicate which kernel to launch
#include "config.h"     

//prototypes for functions in this file
void initVector(float * array, int length, int fraction);
void getLengths(int argc, char * argv[], int *, int *);
void compare(float * result1, float * result2, int n, const char * label);
void printUsage();

/*
   driver for the convolute program.  
*/
int main(int argc, char * argv[])
{
    int maskLen, vectorLen;
    getLengths(argc, argv, &maskLen, &vectorLen);
    float * h_vector = (float *) Malloc(sizeof(float) * vectorLen);
    float * h_mask = (float *) Malloc(sizeof(float) * maskLen);
    float * h_result = (float *) Malloc(sizeof(float) * vectorLen);
    float * d_result = (float *) Malloc(sizeof(float) * vectorLen);
    float h_time, d_globaltime, d_sharedtime, d_consttime, speedup;

    //initialize vector and mask values
    initVector(h_vector, vectorLen, 0);
    initVector(h_mask, maskLen, 1);
   
    //perform the convolution on the CPU
    h_time = h_convolute(h_result, h_mask, h_vector, maskLen, vectorLen);
    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);

    //perform the convolution on the GPU using global memory
    d_globaltime = d_convolute(d_result, h_mask, h_vector, 
                               maskLen, vectorLen, GLOBAL);
    //compare GPU and CPU results 
    compare(h_result, d_result, vectorLen, "global");
    printf("GPU (global memory): \t\t%f msec\n", d_globaltime);
    speedup = h_time/d_globaltime;
    printf("Speedup: \t\t\t%f\n", speedup);


    //perform the convolution on the GPU using shared memory
    d_sharedtime = d_convolute(d_result, h_mask, h_vector, 
                          maskLen, vectorLen, SHARED);
    //compare GPU and CPU results 
    compare(h_result, d_result, vectorLen, "shared");
    printf("GPU (shared memory): \t\t%f msec\n", d_sharedtime);
    speedup = h_time/d_sharedtime;
    printf("Speedup: \t\t\t%f\n", speedup);

    //perform the convolution on the GPU using shared and constant memory
    d_consttime = d_convolute(d_result, h_mask, h_vector, 
                          maskLen, vectorLen, CONSTANT);
    //compare GPU and CPU results 
    compare(h_result, d_result, vectorLen, "shared");
    printf("GPU (constant memory): \t\t%f msec\n", d_consttime);
    speedup = h_time/d_consttime;
    printf("Speedup: \t\t\t%f\n", speedup);

    free(h_result);
    free(d_result);
    free(h_mask);
    free(h_vector);
}    

/* 
    getLengths
    This function parses the command line arguments to get
    the value of the mask length and the value of the
    vector length.  If the command line arguments are invalid, 
    it prints usage information and exits.
    Inputs:
    argc - count of the number of command line arguments
    argv - array of command line arguments
    maskLen - pointer to an int to be set to the mask length
    vectorLen - pointer to an int to be set to the vector length
*/
void getLengths(int argc, char * argv[], int * maskLen, int * vectorLen)
{
    int i, vlen = 0, flen = 0;
    //program can be invoked with: -m <n> -v <n>
    //or: -v <n> -m <n>
    //<n> should be an integer
    if (argc != 5) printUsage();
    for (i = 1; i < argc; i+=2)
    {
        if (strcmp("-m", argv[i]) == 0)
        {
            flen = atoi(argv[i+1]);        
            //mask length must be greater than 0, less than
            //or equal to MAXMASKLEN and odd 
            if (flen <= 0 || flen > MAXMASKLEN || (flen % 2) != 1) printUsage();
        } else if (strcmp("-v", argv[i]) == 0)
        {
            vlen = atoi(argv[i+1]);        
            //vector length must be greater than 0
            if (vlen <= 0) printUsage();
        }
        else
            printUsage();
    }
    //vector length must be greater than or equal to filter length
    if (flen > vlen) printUsage();
    (*maskLen) = flen;
    (*vectorLen) = vlen;
}

/*
    printUsage
    prints usage information and exits
*/
void printUsage()
{
    printf("\nThis program performs a convolution of 1D data. The length of the\n");
    printf("mask and the length of vector to convolute are provided as command\n");
    printf("line arguments. The convolution is performed on the CPU and the GPU.\n");
    printf("The program verifies the GPU results by comparing them to the CPU\n");
    printf("results and outputs the times it takes to perform each convolution.\n");
    printf("usage: convolute -m <mask size> -v <vector size>\n");
    printf("       <mask size> size of the mask\n");
    printf("                   must not be greater than %d\n", MAXMASKLEN);
    printf("                   must be odd\n");
    printf("       <vector size> size of the randomly generated vector to convolute\n");
    printf("                     must not be less than mask size\n");
    exit(EXIT_FAILURE);
}

/* 
    initVector
    Initializes an array of floats of size
    length to random values between 0 and 99 or between 0 and 1
    depending upon the value of fraction.
    Inputs:
    array - pointer to the array to initialize
    length - length of array
    fraction - if set to 1, array is initialized to values between 0 and 1
               otherwise, it is initialized to values between 0 and 99
*/
void initVector(float * array, int length, int fraction)
{
    int i;
    for (i = 0; i < length; i++)
    {
        array[i] = (float)(rand() % 100);
        if (fraction) array[i] = array[i] / 100.0;
    }
}

/*
    compare
    Compares the values in two vectors and outputs an
    error message and exits if the values do not match.
    result1, result2 - float vectors
    n - length of each vector
    label - string to use in the output message if an error occurs
*/
void compare(float * result1, float * result2, int n, const char * label)
{
    int i;
    for (i = 0; i < n; i++)
    {
        float diff = abs(result1[i] - result2[i]);
        if (diff > 0.01) // 
        {
            printf("%s GPU convolution does not match CPU results.\n", label);
            printf("cpu result[%d]: %f, gpu: result[%d]: %f\n", 
                   i, result1[i], i, result2[i]);
            exit(EXIT_FAILURE);
        }
    }
}
