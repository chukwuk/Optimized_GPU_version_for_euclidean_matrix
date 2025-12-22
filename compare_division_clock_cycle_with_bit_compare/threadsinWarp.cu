#include <math.h>
#include "threadsinWarp.h"    
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;


__global__  void threadsInWarp(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  blockDim.x +  threadIdx.x;
   
   int copyvalue; 
   readtimer[threadIdx.x] = globalData[threadIdx.x];
   copyvalue = readtimer[threadIdx.x];
   size_t t = 1;
   size_t z = 10;
   unsigned long long int startTime = clock64();  
   
   if (z >= ((t + 1) * copyvalue)) {
      t = t + 1;
   } 
    

   unsigned long long finishTime = clock64();  


   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = t;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   
}


