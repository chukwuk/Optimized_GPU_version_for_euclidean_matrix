#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


#include "threadsinWarp.h"



#define IDX2C(i,j,ld) (((i)*(ld))+(j))

#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)




using namespace std;


inline
cudaError_t checkCudaErrors(cudaError_t result, string functioncall = "")
{
//#if defined(DEBUG) || defined(_DEBUG)
  //fprintf(stderr, "CUDA Runtime Error: %d\n", result);
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error for this function call ( %s ) : %s\n", 
            functioncall.c_str(), cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}


	
int
main( int argc, char* argv[ ] )
{ 
  
  
  int BLOCKSIZE;
  int NUMBLOCKS;
  int MINGRIDSIZE;  
  
  
  cudaOccupancyMaxPotentialBlockSize( &MINGRIDSIZE, &BLOCKSIZE, 
                                      threadsInWarp, 0, 0); 
   
  BLOCKSIZE = 128;
  NUMBLOCKS = (BLOCKSIZE+BLOCKSIZE-1)/BLOCKSIZE;
   
  
   
  //threadProperties* threads = new threadProperties[BLOCKSIZE];
  
  threadProperties* threadProp; 
  
  int* globalData;
  
  cudaError_t status;
  int threadPropertiesDataSize = (sizeof(threadProperties) * BLOCKSIZE);
  int globalDataSize = (sizeof(int) * BLOCKSIZE);
  fprintf (stderr, "Thread Properties Struct Size %i \n", threadPropertiesDataSize);
  fprintf (stderr, "Global Data Size %i \n", globalDataSize);
  
  // pinned data
  
  cudaMallocHost((void**)&threadProp ,threadPropertiesDataSize);
  
  cudaMallocHost((void**)&globalData ,globalDataSize);

  
  
  for (size_t i = 0; i < BLOCKSIZE; i++) {
      globalData[i] = 1;
  } 


  
  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
    
  threadProperties* threadPropDev; 
  int* globalDataDev;

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&threadPropDev), threadPropertiesDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&threadsDev), threadPropertiesDataSize)");
 
  

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&globalDataDev), globalDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&globalDataDev), globalDataSize)");
   
     
     
  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
  
  
  
  
  // Record the start event
  cudaEventRecord(start, 0); 

  // copy data from host memory to the device:
  status = cudaMemcpy(globalDataDev, globalData, globalDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(globalDataDev, globalData, globalDataSize, cudaMemcpyHostToDevice );");  

  // kernel launch 
  threadsInWarp<<< grid, threads >>>(threadPropDev, globalData);
  status = cudaGetLastError(); 
  // check for cuda errors
  checkCudaErrors( status,"threadsInWarp<<< grid, threads >>>( threadsDev, globalData); ");

  // copy data from device memory to host 
  status = cudaMemcpy(threadProp, threadPropDev, threadPropertiesDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(threads, threadsDev, threadsPropertiesDataSize, cudaMemcpyDeviceToHost) "); 
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  float GpuTime = 0;
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
  for (int i = 0; i < BLOCKSIZE; i++) {
  	  
     printf("(threadId.x: %i) execution time for copying data from GMEM to SMEM: %llu clock cycle\n", threadProp[i].thread_x, threadProp[i].time ); 
     //printf ("threadid.x: %i \n", threadProp[i].thread_x);

  }	  


  return 0;
};	
