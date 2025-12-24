#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>


#pragma nv_diag_suppress static_var_with_dynamic_init

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

__device__ size_t divFast (size_t t, size_t current_compute_stage, size_t dataFetchSize, size_t z)
  {
  //size_t count = t;  
   while (z >= (current_compute_stage + (t + 1) * dataFetchSize)) {
        t = t + 1;
   }
   return t;
 }

__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread) {
    
   size_t gid_start = (size_t) blockIdx.x * (size_t) blockDim.x;
    
   size_t gid = (size_t) blockIdx.x * (size_t) blockDim.x + threadIdx.x;

   extern  __shared__ LocationPrim locations [];

   int blocksize = blockDim.x * blockDim.y * blockDim.z;      
   
   auto grid = cooperative_groups::this_grid();
   auto block = cooperative_groups::this_thread_block();
   constexpr size_t stages_count = 2; // Pipeline with two stages

   size_t numofDataperBatch = (numDataPerThread) * blocksize;
   size_t numofDataperHalfBatch = (numDataPerThread/2) * blocksize;
   size_t numRef =  numofDataperBatch + blocksize; 

   auto numBatchToFetch = [&](int batchfetched) -> int {	   
     return ((NUMDATA - batchfetched) >= (numofDataperHalfBatch + blocksize)) ? numofDataperHalfBatch : (NUMDATA - batchfetched);
   };

   size_t shared_offset[stages_count] = {0, numofDataperHalfBatch}; // Offsets to each batch

   // Allocate shared storage for a two-stage cuda::pipeline:
   __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count > shared_state;
   auto pipeline = cuda::make_pipeline(block, &shared_state);
   
   size_t firstBatchNum =  numBatchToFetch(0);
   pipeline.producer_acquire(); 
   cuda::memcpy_async(block, locations + shared_offset[0], cordinates + 0, sizeof(LocationPrim)*firstBatchNum, pipeline);
   pipeline.producer_commit();
  
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
       locations[numRef + threadId] = cordinates[gid];    

   } 
    
   
   dataFetchSize = firstBatchNum;  	  
   size_t compute_stage_idx = 0;
   size_t copy_stage_idx = 1;
   size_t global_i = 0;
   size_t current_compute_stage;
   size_t nextBatchNum;
   for (int i = firstBatchNum; i < NUMDATA; i+=numBatchToFetch(i)) {
       nextBatchNum = numBatchToFetch(i);
      //Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();
        cuda::memcpy_async(block, locations + shared_offset[copy_stage_idx], cordinates + i, sizeof(LocationPrim) * nextBatchNum,  pipeline);
        pipeline.producer_commit();

       // Collectively wait for the operations commited to the
       // previous `compute` stage to complete:
       pipeline.consumer_wait();
       __syncthreads();
              
       t = 0;
       current_compute_stage = shared_offset[compute_stage_idx];  
       totalDataCompute = current_compute_stage + dataFetchSize*blocksize; 
       for (size_t z = current_compute_stage + threadId, c = global_i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
               
	  if (z >= (current_compute_stage + (t + 1) * dataFetchSize)) {
             t = t + 1;
          }
	  
	  //t = divFast (t, current_compute_stage, dataFetchSize, z);
          real_gid =  t + gid_start;
	   
          if (real_gid >= NUMDATA) {
            continue;
          }
	  dataSub = t * dataFetchSize;
          k = c - dataSub; 
          index = real_gid*NUMDATA;
          d = z - dataSub;
	  ref_index = numRef + t;  
          float x_co = (locations[ref_index].x - locations[d].x);
          float y_co = (locations[ref_index].y - locations[d].y); 
	  float pow_xco = x_co * x_co;
          float pow_yco = y_co * y_co;
          float pow_plus = sqrt(pow_yco+pow_xco);
          euclideanDistance[index+k] = pow_plus;
       }

      dataFetchSize = nextBatchNum;
      compute_stage_idx = (compute_stage_idx != 0) ? 0 : 1;
      copy_stage_idx = (copy_stage_idx != 0) ? 0 : 1;
      global_i = i;
      __syncthreads();

      // Collectively release the stage resources
      pipeline.consumer_release();
	 
    }

      
      // Compute the data fetch by the last iteration
       pipeline.consumer_wait(); 
       t = 0;
       current_compute_stage = shared_offset[compute_stage_idx];  
       totalDataCompute = current_compute_stage + dataFetchSize*blocksize; 
       for (size_t z = current_compute_stage + threadId, c = global_i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
              
	  if (z >= (current_compute_stage + (t + 1) * dataFetchSize)) {
             t = t + 1;
          }
	  
	  //t = divFast (t, current_compute_stage, dataFetchSize, z);
          real_gid =  t + gid_start;
	   
          if (real_gid >= NUMDATA) {
            continue;
          }
	  dataSub = t * dataFetchSize;
          k = c - dataSub; 
          index = real_gid*NUMDATA;
          d = z - dataSub;
	  ref_index = numRef + t;  
          //float x_co = 0;// (locations[ref_index].x - locations[d].x);
          //float y_co = 0;// (locations[ref_index].y - locations[d].y);
          float x_co = (locations[ref_index].x - locations[d].x);
          float y_co = (locations[ref_index].y - locations[d].y); 
	  float pow_xco = x_co * x_co;
          float pow_yco = y_co * y_co;
          float pow_plus = sqrt(pow_yco+pow_xco);
          euclideanDistance[index+k] = pow_plus;
       }
      __syncthreads();

      //compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
      pipeline.consumer_release();
  
}

