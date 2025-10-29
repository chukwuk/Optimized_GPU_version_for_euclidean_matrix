#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#define SHARED_MEM_SIZE 128

using namespace std;


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA) {
    
   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < NUMDATA) {
    	long long int index = gid * (long long int) NUMDATA;
    	float  count = 0.0;
    	for (long long int i = gid; i < NUMDATA; i++)  {
           float x_co =  (cordinates[gid].x - cordinates[i].x);
           float y_co =  (cordinates[gid].y - cordinates[i].y);
           float pow_xco = powf(x_co, 2.0);
           float pow_yco = powf(y_co, 2.0);
	   count+=1.0;
       	   float pow_plus = powf((pow_yco+pow_xco), 0.5);
           euclideanDistance[index+i] = pow_plus;
           long long int symData = (i* NUMDATA) + gid;
           if (i != gid && symData < ( NUMDATA* NUMDATA)) { 
              euclideanDistance[symData] = pow_plus;
           }
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
   
   //__shared__ LocationPrim locations [6100];

   
   __shared__ float ref_x [SHARED_MEM_SIZE]; 
   __shared__ float ref_y [SHARED_MEM_SIZE];
   if (gid < NUMDATA) {
       
      long long int index = gid * (long long int) NUMDATA;
      int k_start;
      int z_start; 
      //int i_start = gid - threadIdx.x; 
      int i_start = blockIdx.x * blockDim.x;     
      int blocksize = blockDim.x * blockDim.y * blockDim.z;
      
      for (int i = i_start, j = 0; i < NUMDATA; i+=(numDataPerThread*blocksize), j+=1) {

	  
	  for (int n = threadIdx.x, m = i + threadIdx.x; n < (numDataPerThread*blocksize) && m < NUMDATA; n+=blocksize, m+= blocksize) {
             locations[n] = cordinates[m];
	     //__pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
          }
	  //__pipeline_commit();
          //__pipeline_wait_prior(0);
	  __syncthreads();
	   
	  for (int b = 0; b < blocksize; b++) {
	     index = (i_start + b) * (long long int) NUMDATA;
	     if (i == i_start) {
                ref_x[b] = locations[b].x;
	        ref_y[b] = locations[b].y;
	        k_start = b;
	        z_start = i_start + b;

 	    } else {
              k_start = 0;
	      z_start = i;

            }
          
	  
	     
	  for (int k = k_start + threadIdx.x, z = z_start + threadIdx.x; k < (numDataPerThread*blocksize) && z < NUMDATA; k+=blocksize, z+=blocksize) {
              if (z >= (i_start + b)) {
                 float x_co =  (ref_x[b] - locations[k].x);
		 float y_co =  (ref_y[b] - locations[k].y);
                 float pow_xco = powf(x_co, 2.0);
                 float pow_yco = powf(y_co, 2.0);
       	         float pow_plus = powf((pow_yco+pow_xco), 0.5);
                 euclideanDistance[index+z] = pow_plus;
                 long long int symData = (z * NUMDATA) + (i_start + b);
                 if (z != (i_start + b) && symData < ( NUMDATA* NUMDATA)) { 
                    euclideanDistance[symData] = pow_plus;
                 }
	      }
            }
	    __syncthreads();
	  
	 }

	//__syncthreads();
	 
      }
   
   }
    
}
