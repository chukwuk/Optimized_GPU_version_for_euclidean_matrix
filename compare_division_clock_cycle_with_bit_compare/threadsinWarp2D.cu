#include <math.h>
#include "threadsinWarp.h"    
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;


__global__  void threadsInWarp(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  blockDim.x +  threadIdx.x;
   
   size_t copyvalue; 
   readtimer[threadIdx.x] = globalData[threadIdx.x];
   copyvalue = readtimer[threadIdx.x];
   size_t t;
   size_t z = 10;
   unsigned long long int startTime = clock64();  
    
   t  = z/copyvalue;

   unsigned long long finishTime = clock64();  


   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = t;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   
}


