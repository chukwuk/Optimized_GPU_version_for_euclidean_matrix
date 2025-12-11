#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA) {
    
   size_t gid =  blockIdx.x * blockDim.x + threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < NUMDATA) {
    	size_t index = gid * NUMDATA;
    	for (int i = 0; i < NUMDATA; i++)  {
           float x_co =  (cordinates[gid].x - cordinates[i].x);
           float y_co =  (cordinates[gid].y - cordinates[i].y);
           float pow_xco = x_co * x_co;
           float pow_yco = y_co * y_co;
       	   float pow_plus = sqrt(pow_yco+pow_xco);
           euclideanDistance[index+i] = pow_plus;
         }
    }
    
}



__global__  void euclideanMatrixStaticSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA, int blocksize) {
    
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



__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA, int numDataPerThread) {
    
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
   extern  __shared__ LocationPrim locations [];

   
   if (gid < NUMDATA) {
       
      long long int index = gid * (long long int) NUMDATA;
      float ref_x, ref_y; 
      int blocksize = blockDim.x * blockDim.y * blockDim.z;      
      for (int i = 0, j = 0; i < NUMDATA; i+=(numDataPerThread*blocksize), j+=1) {

	  
	  for (int n = 0, m = i; n < (numDataPerThread*blocksize) && m < NUMDATA; n+=blocksize, m+= blocksize) {
             //locations[threadIdx.x + n*blocksize] = cordinates[m + threadIdx.x];
	     __pipeline_memcpy_async(&locations[threadIdx.x + n], &cordinates[m], sizeof(LocationPrim));
          }
	  __pipeline_commit();
          __pipeline_wait_prior(0);
	  __syncthreads();
	  if (i == gid) {
             ref_x = locations[threadIdx.x].x;
	     ref_y = locations[threadIdx.x].y;
 
	  }
	  
	  for (int k = 0, z = (j*numDataPerThread*blocksize); k < (numDataPerThread*blocksize) && z < NUMDATA; k++, z++) {
               float x_co =  (ref_x - locations[k].x);
	       float y_co =  (ref_y - locations[k].y);
               float pow_xco = powf(x_co, 2.0);
               float pow_yco = powf(y_co, 2.0);
       	       float pow_plus = powf((pow_yco+pow_xco), 0.5);
               euclideanDistance[index+z] = pow_plus;
               //long long int symData = (z * NUMDATA) + gid;
	     
          }
	 __syncthreads();
	 
      }
   
   }
    
}
