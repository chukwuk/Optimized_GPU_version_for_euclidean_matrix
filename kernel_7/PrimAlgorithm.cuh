#include <math.h>
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

//using namespace std;


__global__ void __launch_bounds__(256, 1) euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread) {
   //size_t gid_start =  blockIdx.x * blockDim.x;
   size_t gid_start = blockIdx.x * FAKE_BLOCKSIZE; 
   size_t gid =  gid_start + threadIdx.x;
   extern  __shared__ LocationPrim locations [];
   const int blocksize = blockDim.x * blockDim.y * blockDim.z;
   const int ref_blocksize =  (gid_start + FAKE_BLOCKSIZE) >= NUMDATA ? (NUMDATA - gid_start) : FAKE_BLOCKSIZE; 
   size_t numofDataperBatch = (numDataPerThread) * blocksize;
   size_t numRef =  numofDataperBatch + blocksize; // ((numDataPerThread+1) * blocksize);
   auto numBatchToFetch = [&](int batchfetched) -> int {	   
     return ((NUMDATA - batchfetched) >= (numofDataperBatch + blocksize)) ? numofDataperBatch : (NUMDATA - batchfetched);
   };
   size_t index;
   size_t real_gid;
   //size_t ref_index; 
   size_t dataFetchSize;  	  
   size_t threadId = threadIdx.x;
   size_t totalDataCompute;
   //if (threadId < FAKE_BLOCKSIZE) {
       locations[numRef + threadId] = cordinates[gid];    
  // }
 //  __syncthreads(); 
   float ref_x[FAKE_BLOCKSIZE];
   float ref_y[FAKE_BLOCKSIZE];
   /*
   #pragma unroll
   for (int i = 0; i < 64; i++) {
       ref_x[i] = locations[numRef + i].x;
       ref_y[i] = locations[numRef + i].y;
   }
   */
   for (int i = 0; i < NUMDATA; i+=numBatchToFetch(i)) {
       dataFetchSize = numBatchToFetch(i);  	  
       for (size_t n = threadId, m = i + threadId; n < dataFetchSize; n+=blocksize, m+= blocksize) {
           //locations[n] = cordinates[m];
	   __pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
       } 
        __pipeline_commit();
        __pipeline_wait_prior(0);
	
       if (i == 0) {
          #pragma unroll
          for (int r = 0; r < FAKE_BLOCKSIZE; r++) {
             ref_x[r] = locations[numRef + r].x;
             ref_y[r] = locations[numRef + r].y;
          }
       }
        __syncthreads();
       totalDataCompute = dataFetchSize;
       for (size_t z = threadId, c = i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
	  float cal_x = locations[z].x;  
          float cal_y = locations[z].y;
          
          #pragma unroll
	  for (int n = 0; n < FAKE_BLOCKSIZE; n++) {           
              real_gid =  n + gid_start;	       
              index = real_gid*NUMDATA;
	      //ref_index = numRef + n; 
             // float x_co =  (locations[ref_index].x - cal_x);
              //float y_co =  (locations[ref_index].y - cal_y);
	      float x_co =  (ref_x[n] - cal_x);
	      float y_co =  (ref_y[n] - cal_y);
	      float pow_xco = x_co * x_co;
              float pow_yco = y_co * y_co;
              float pow_plus = sqrt(pow_yco+pow_xco);
              euclideanDistance[index+c] = pow_plus;  
	  }
       }  
      __syncthreads();	 
   }
}

