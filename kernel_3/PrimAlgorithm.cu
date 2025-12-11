#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA) {
    
   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   //long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   long long int gid_start = (long long int) blockIdx.x * (long long int) blockDim.x; 
   int blocksize =   blockDim.x*blockDim.y*blockDim.z;
   long long int index;
   float count = 0.0;
   long long int real_gid;
   long long int k;
   long long int j;
   for (long long int i = threadIdx.x; i < NUMDATA*blocksize; i+=blocksize)  {
       j = i / NUMDATA;
       real_gid =  j + gid_start;
       if (real_gid >= NUMDATA) {
           continue;
       }
       k = i - (j * NUMDATA); 
       index = real_gid*NUMDATA;	   
       float x_co =  (cordinates[real_gid].x - cordinates[k].x);
       float y_co =  (cordinates[real_gid].y - cordinates[k].y);
       float pow_xco = powf(x_co, 2.0);
       float pow_yco = powf(y_co, 2.0);
       count+=1.0;
       float pow_plus = powf((pow_yco+pow_xco), 0.5);
       euclideanDistance[index+k] = pow_plus;
  }
    
}



/*
__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA) {
    
   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   //long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   long long int gid_start = (long long int) blockIdx.x * (long long int) blockDim.x; 
   int blocksize =   blockDim.x*blockDim.y*blockDim.z;
   long long int index;
   float count = 0.0;
   long long int real_gid;
   long long int k;
   long long int j;
   
   for (int j = 0; j < blocksize; j++) {
   for (long long int i = threadIdx.x; i < NUMDATA; i++)  {
       real_gid =  j + gid_start;
       index = real_gid*NUMDATA;	   
       float x_co =  (cordinates[real_gid].x - cordinates[i].x);
       float y_co =  (cordinates[real_gid].y - cordinates[i].y);
       float pow_xco = powf(x_co, 2.0);
       float pow_yco = powf(y_co, 2.0);
       count+=1.0;
       float pow_plus = powf((pow_yco+pow_xco), 0.5);
       euclideanDistance[index+i] = pow_plus;
  }
  __syncthreads();
  }
    
}
*/



__global__  void euclideanMatrixStaticSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int blocksize) {
    
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
   __shared__ LocationPrim locations [64];

   
   if (gid < NUMDATA) { 
        	   
    	long long int index = gid * (long long int) NUMDATA;
        float ref_x, ref_y;        
        for (int i = gid, j = 0; i < NUMDATA; i+=blocksize, j+=1) {
	    locations[threadIdx.x] = cordinates[i];
	    __syncthreads();
	    if (i == gid) {
                ref_x = locations[threadIdx.x].x;
		ref_y = locations[threadIdx.x].y;
 
	    }
	    for (int k = 0, z = j*blocksize; k < blocksize && z < NUMDATA; k++, z++) {
                if (z >= gid) {
                   float x_co =  (ref_x - locations[k].x);
		   float y_co =  (ref_y - locations[k].y);
                   float pow_xco = powf(x_co, 2.0);
                   float pow_yco = powf(y_co, 2.0);
       	           float pow_plus = powf((pow_yco+pow_xco), 0.5);
                   euclideanDistance[index+z] = pow_plus;
                   long long int symData = (z * NUMDATA) + gid;
                   if (z != gid && symData < ( NUMDATA* NUMDATA)) { 
                       euclideanDistance[symData] = pow_plus;
                    }
		}
            }
	    __syncthreads();
	}

        
    }
    
}



__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread) {
    
   size_t gid_start = (size_t) blockIdx.x * (size_t) blockDim.x;
    
   size_t gid = (size_t) blockIdx.x * (size_t) blockDim.x + threadIdx.x;

   extern  __shared__ LocationPrim locations [];
   //__shared__ LocationPrim ref[256];

   int blocksize = blockDim.x * blockDim.y * blockDim.z;      
   
    
   size_t numofDataperBatch = (numDataPerThread) * blocksize;
   auto numBatchToFetch = [&](int batchfetched) -> int {	   
     return ((NUMDATA - batchfetched) >= numofDataperBatch) ? numofDataperBatch : (NUMDATA - batchfetched);
   };
   //float ref_x, ref_y;
      
   size_t index;
   size_t real_gid;
   size_t t = 0;
   size_t k;
   size_t dataSub;
   size_t ref_index;
   size_t d; 
   size_t dataFetchSize;  	  
   size_t threadId = threadIdx.x;
   size_t totalDataCompute; 
   if (gid < NUMDATA) {
      //ref[threadIdx.x] = cordinates[gid];
       locations[numofDataperBatch + threadId] = cordinates[gid];    
 
   } 
    
   for (int i = 0; i < NUMDATA; i+=numBatchToFetch(i)) {

       dataFetchSize = numBatchToFetch(i);  	  
       for (size_t n = threadId, m = i + threadId; n < dataFetchSize; n+=blocksize, m+= blocksize) {
           //locations[n] = cordinates[m];
	   __pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
       } 
       __pipeline_commit();
       __pipeline_wait_prior(0);
       //__syncthreads();
       
       t = 0;
       totalDataCompute = dataFetchSize*blocksize;       
       //count = threadIdx.x;
       for (size_t z = threadId, c = i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
           
	  t  = z/dataFetchSize;
          
          real_gid =  t + gid_start;
	   
          if (real_gid >= NUMDATA) {
            continue;
          }
	  dataSub = t * dataFetchSize;
          k = c - dataSub; 
          index = real_gid*NUMDATA;
          d = z - dataSub;
	  ref_index = numofDataperBatch + t;  
          //float x_co =  (cordinates[real_gid].x - locations[d].x);
          //float y_co =  (cordinates[real_gid].y - locations[d].y);
          //float x_co =  (ref[t].x - locations[d].x);
          //float y_co =  (ref[t].y - locations[d].y);
          float x_co =  (locations[ref_index].x - locations[d].x);
          float y_co =  (locations[ref_index].y - locations[d].y);
	  float pow_xco = x_co * x_co;
          float pow_yco = y_co * y_co;
          float pow_plus = sqrt(pow_yco+pow_xco);
          euclideanDistance[index+k] = pow_plus;
       }  
      __syncthreads();
	 
      }
 
    
}

