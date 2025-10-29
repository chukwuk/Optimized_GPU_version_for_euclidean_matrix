#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <fstream>

#include "PrimAlgorithm.h"

#define IDX2C(i,j,ld) (((i)*(ld))+(j))


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
  //srand(time(0));
  unsigned long long int NUMDATA = 10112;
  const unsigned long long int bytes = NUMDATA * (long long int) sizeof(LocationPrim);
  const unsigned long long int bytes4euc = ( NUMDATA *  NUMDATA * (long long int)sizeof(float));
  fprintf (stderr, "Amount of data transfered to the device is %lld bytes\n", bytes4euc);
 // float time = 1.0;
  LocationPrim* locate = new LocationPrim[NUMDATA];
  for (int i = 0; i < NUMDATA; i++) {
      locate[i].x = rand() % 101;
      locate[i].y = rand() % 101;
      if (i == 0) {      
         fprintf (stderr, "%10.4f\n", locate[i].x);
         fprintf (stderr, "%10.4f\n", locate[i].y);     
      }
  }
  fprintf (stderr, "%10.4f\n", locate[NUMDATA-1].x);
  fprintf (stderr, "%10.4f\n", locate[NUMDATA-1].y);

  // Allocate memory on device
  float *distanceBtwAllLocation;
  LocationPrim *cordinateLocation;
  // Allocate memory on host
  float *HdistanceBtwAllLocation;// = new float[NUMDATA * NUMDATA];
  
  cudaMallocHost((void**)&HdistanceBtwAllLocation, bytes4euc); 


  int BLOCKSIZE;
  int MINGRIDSIZE;

  cudaOccupancyMaxPotentialBlockSize( &MINGRIDSIZE, &BLOCKSIZE, 
                                      euclideanMatrix, 0, 0); 
  
  
  
  BLOCKSIZE = 128; 
   
 
  int NUMBLOCKS = (NUMDATA + BLOCKSIZE - 1)/BLOCKSIZE;
    
  
  fprintf (stderr, "BLOCKSIZE is %d\n", BLOCKSIZE);
  fprintf (stderr, "NUMBER OF BLOCKS is %d\n", NUMBLOCKS);
  
  

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  cudaError_t status;
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&distanceBtwAllLocation), bytes4euc);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&distanceBtwAllLocation), bytes4euc)");
  
  // allocate memory on the GPU device
  status = cudaMalloc( (void **)(&cordinateLocation), bytes);
  // checks for cuda errors
  checkCudaErrors( status, "cudaMalloc( (void **)(&cordinateLocation), bytes)");

  // copy data from host memory to the device:

  status = cudaMemcpy(cordinateLocation, locate, bytes, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(cordinateLocation, locate, bytes, cudaMemcpyHostToDevice )" );  
   
  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
   
 /*
  int dynSmemSize = 101376; 
  
  status = cudaFuncSetAttribute(euclideanMatrixDynamicSharedMemory, cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmemSize);
   
  checkCudaErrors( status," cudaFuncSetAttribute(euclideanMatrixDynamicSharedMemory, cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmemSize); ");  
   
  int numDataPerThread = dynSmemSize/(sizeof(LocationPrim) * BLOCKSIZE);

  
  printf("  Dynamic Memory Size: %d bytes\n", dynSmemSize);

  printf("  Number of Data loaded on the shared memory per thread: %d \n", numDataPerThread);
   */
   

  // Record the start event
  cudaEventRecord(start, 0); 
  
  // call the kernel
  euclideanMatrix<<< grid, threads>>>( cordinateLocation, distanceBtwAllLocation,  NUMDATA);
  
  status = cudaDeviceSynchronize( );
  

  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  
  checkCudaErrors( status,"euclideanMatrix<<< grid, threads >>>( cordinateLocation, distanceBtwAllLocation,   NUMDATA)");  
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HdistanceBtwAllLocation, distanceBtwAllLocation, bytes4euc, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

   
  // Calculate elapsed time
  float GpuTime = 0;
  cudaEventElapsedTime(&GpuTime, start, stop); 
  
   
  printf("  GPU time: %f milliseconds\n", GpuTime);
  //printf("  Device to Host bandwidth (GB/s): %f\n", HdistanceBtwAllLocation[NUMDATA*(NUMDATA-1)] / time);
  //printf("  Device to Host bandwidth (GB/s): %f\n", HdistanceBtwAllLocation[NUMDATA-1] / time);

  // free device memory 
  cudaFree( distanceBtwAllLocation );
  cudaFree( cordinateLocation ); 
  
   
  /* Running it on CPU************************************/  
  
  // Allocate memory on host
  float** AllLocationDistance = new float* [NUMDATA];
  
  for (int i = 0; i < NUMDATA; i++) {
     AllLocationDistance[i] = new float[NUMDATA];
  }
  
  // Record the start event
  cudaEventRecord(start, 0); 
    
  for (int i = 0; i < NUMDATA; i++) {
     for (int j = 0; j < NUMDATA; j++) {
         float x_co =  (locate[i].x - locate[j].x);
           float y_co =  (locate[i].y - locate[j].y);
           float pow_xco = powf(x_co, 2.0);
           float pow_yco = powf(y_co, 2.0);
       	   float pow_plus = powf((pow_yco+pow_xco), 0.5);
           AllLocationDistance[i][j] = pow_plus;
     }
  }
    
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
     
  // Calculate elapsed time
  float CpuTime = 0;
  cudaEventElapsedTime(&CpuTime, start, stop); 


  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  printf("  CPU time: %f milliseconds\n", CpuTime);
  //printf("  Device to Host bandwidth (GB/s): %f\n", bytes4euc*1e-9/time);
  //double check = 99999.000*100000.000;
  //printf("  Device to Host bandwidth (GB/s): %f\n", check);
   
  /* 
  float cpuData;
  float gpuData; 
  fprintf(stderr, "Checking if the CPU and GPU data are equal \n");      
  for (int i = 0; i < NUMDATA; i++) {
      for (int j = 0; j < NUMDATA; j++) {
         if ( (HdistanceBtwAllLocation[IDX2C(i,j,NUMDATA)] - AllLocationDistance[i][j]) > 0.01) {         
	      cpuData = AllLocationDistance[i][j];
	      gpuData =  HdistanceBtwAllLocation[IDX2C(i,j,NUMDATA)]; 

              fprintf(stderr, "row: %d , cols: %d, CPU data: %.2f , GPU data: %.2f are not equal, with difference (%.10f) \n", i, j,cpuData, gpuData, (gpuData - cpuData));      
         }   
      }
  
  }
  */
  
  FILE* file1 = fopen("mismatch.txt", "w");
   
  float cpuData, gpuData; 
  fprintf(stderr, "Checking if the CPU and GPU data are equal \n");      
  for (int i = 0; i < NUMDATA; i++) {
      for (int j = 0; j < NUMDATA; j++) {
	 cpuData = AllLocationDistance[i][j];
	 gpuData =  HdistanceBtwAllLocation[IDX2C(i,j,NUMDATA)]; 
         if ( (powf(gpuData - cpuData, 2.0)) > powf(0.01, 2.0)) {         

              fprintf(file1, "row: %d , cols: %d, CPU data: %.2f , GPU data: %.2f are not equal, with difference (%.10f) \n", i, j,cpuData, gpuData, (gpuData - cpuData));      
         }   
      } 
  }
  fclose(file1);
 

    
  // free host memory
  //delete[] HdistanceBtwAllLocation; 
  cudaFreeHost(HdistanceBtwAllLocation);

  // free host memory
  delete[] locate;
   
    
  for (int i = 0; i < NUMDATA; i++) {
     delete[] AllLocationDistance[i]; 
  }
  
  delete[] AllLocationDistance; 

  return 0;

};	
