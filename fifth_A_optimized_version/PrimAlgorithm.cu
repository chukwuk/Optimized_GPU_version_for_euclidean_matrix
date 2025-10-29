#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

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



__global__  void euclideanMatrixStaticSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA) {
    
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
   __shared__ float locationsX [6144];
   __shared__ float locationsY [6144];


    
   if (gid < NUMDATA) {
       
      long long int index = gid * (long long int) NUMDATA;
      float ref_x, ref_y;
      int k_start;
      int z_start; 
      int blocksize = blockDim.x * blockDim.y * blockDim.z;
      int numDataPerThread = 6144/blockDim.x;
      int i_start = gid - threadIdx.x;       
      for (int i = i_start, j = 0; i < NUMDATA; i+=(numDataPerThread*blocksize), j+=1) {

	  
	  for (int n = threadIdx.x, m = i + threadIdx.x; n < (numDataPerThread*blocksize) && m < NUMDATA; n+=blocksize, m+= blocksize) {
             locationsX[n] = cordinates[m].x;
	     locationsY[n] = cordinates[m].y;
	     //__pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
          }
	  //__pipeline_commit();
          //__pipeline_wait_prior(0);
	  __syncthreads();
	  if (i == i_start) {
             ref_x = locationsX[threadIdx.x];
	     ref_y = locationsY[threadIdx.x];
	     k_start = threadIdx.x;
	     z_start = gid;

 	  } else {
            k_start = 0;
	    z_start = i;

          }
	  
	  for (int k = k_start, z = z_start; k < (numDataPerThread*blocksize) && z < NUMDATA; k++, z++) {
              if (z >= gid) {
                 float x_co =  (ref_x - locationsX[k]);
		 float y_co =  (ref_y - locationsY[k]);
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

   
   if (gid < NUMDATA) {
       
      long long int index = gid * (long long int) NUMDATA;
      float ref_x, ref_y;
      int k_start;
      int z_start; 
      int blocksize = blockDim.x * blockDim.y * blockDim.z;
      int i_start = gid - threadIdx.x;       
      for (int i = i_start, j = 0; i < NUMDATA; i+=(numDataPerThread*blocksize), j+=1) {

	  
	  for (int n = threadIdx.x, m = i + threadIdx.x; n < (numDataPerThread*blocksize) && m < NUMDATA; n+=blocksize, m+= blocksize) {
             locations[n] = cordinates[m];
	     //__pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
          }
	  //__pipeline_commit();
          //__pipeline_wait_prior(0);
	  __syncthreads();
	  if (i == i_start) {
             ref_x = locations[threadIdx.x].x;
	     ref_y = locations[threadIdx.x].y;
	     k_start = threadIdx.x;
	     z_start = gid;

 	  } else {
            k_start = 0;
	    z_start = i;

          }
	  
	  for (int k = k_start, z = z_start; k < (numDataPerThread*blocksize) && z < NUMDATA; k++, z++) {
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
