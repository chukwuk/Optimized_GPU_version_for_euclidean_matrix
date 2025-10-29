#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>

#pragma nv_diag_suppress static_var_with_dynamic_init

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
    
   
   auto grid = cooperative_groups::this_grid();
   auto block = cooperative_groups::this_thread_block();
   
   extern  __shared__ LocationPrim locations [];
   



   if (gid < NUMDATA) {
      

      
      float ref_x, ref_y; 
      ref_x = cordinates[gid].x;
      ref_y = cordinates[gid].y; 
      int jj; 
      long long int index = gid * (long long int) NUMDATA;
      constexpr size_t stages_count = 2; // Pipeline with two stages

      size_t numofDataperBatch = (numDataPerThread/2) * block.size();

      size_t shared_offset[stages_count] = {0, numofDataperBatch}; // Offsets to each batch
   
      // Allocate shared storage for a two-stage cuda::pipeline:
      __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count > shared_state;

      auto pipeline = cuda::make_pipeline(block, &shared_state);
         

      pipeline.producer_acquire(); 
      //int i_start = gid - threadIdx.x;
       //int i_start = 0;

      cuda::memcpy_async(block, locations + shared_offset[0], cordinates + 0, sizeof(LocationPrim)*numofDataperBatch, pipeline);
      

      pipeline.producer_commit();
             
     /* 
            
      pipeline.producer_acquire(); 
      
      cuda::memcpy_async(block, locations + shared_offset[1], cordinates + numofDataperBatch, sizeof(LocationPrim)*numofDataperBatch, pipeline);

      
      pipeline.producer_commit();
      
      */
      //ref_x = locations[threadIdx.x].x;
      //ref_y = locations[threadIdx.x].y;
      
      //int k_prev, z_prev;
      // k_prev = shared_offset[0] + threadIdx.x;
      //k_prev = shared_offset[0];  
      //z_prev = i_start;
      //int k_start, z_start;
     
      
     // Compute the data fetch by the last iteration
     pipeline.consumer_wait();
      

      __syncthreads();
      
            
      for (int m = ((blockIdx.x*NUMDATA)+threadIdx.x), n = threadIdx.x; n < numofDataperBatch; n+=block.size(), m+=block.size()) {
          //locations[threadIdx.x + n*blocksize] = cordinates[m + threadIdx.x];
  	  //__pipeline_memcpy_async(&locations[threadIdx.x + n], &cordinates[m], sizeof(LocationPrim));
           euclideanDistance[m] = locations[n].x;
          
      } 
      
      
     pipeline.consumer_release();
     
     
     /*
     
     // Compute the data fetch by the last iteration
     pipeline.consumer_wait();
      

      __syncthreads();
      
            
      for (int m = ((blockIdx.x*NUMDATA)+threadIdx.x+numofDataperBatch), n = shared_offset[1] + threadIdx.x; n < shared_offset[1]+numofDataperBatch; n+=block.size(), m+=block.size()) {
          //locations[threadIdx.x + n*blocksize] = cordinates[m + threadIdx.x];
  	  //__pipeline_memcpy_async(&locations[threadIdx.x + n], &cordinates[m], sizeof(LocationPrim));
           euclideanDistance[m] = locations[n].x;
          
      } 
      
      
     pipeline.consumer_release();
     */

      /* 
      size_t compute_stage_idx = 0;
      //int blocksize = blockDim.x * blockDim.y * blockDim.z;      
      for (int i = numofDataperBatch, j = 1; i < NUMDATA; i+=numofDataperBatch, j+=1) {

	  jj = j-1;
	  // Stage indices for the compute and copy stages:
          compute_stage_idx = (j - 1) % 2;
          size_t copy_stage_idx = j % 2;
	  
          // Collectively acquire the pipeline head stage from all producer threads:
          pipeline.producer_acquire();


	  //for (int n = shared_offset[copy_stage_idx], m = i; n < (shared_offset[compute_stage_idx] + (numofDataperBatch)) && m < NUMDATA; n+=block.size(), m+= block.size()) {
             //locations[threadIdx.x + n*blocksize] = cordinates[m + threadIdx.x];
	     //__pipeline_memcpy_async(&locations[threadIdx.x + n], &cordinates[m], sizeof(LocationPrim));
             cuda::memcpy_async(block, locations + shared_offset[copy_stage_idx], cordinates + i, sizeof(LocationPrim) * numofDataperBatch, pipeline);


          //}
	  pipeline.producer_commit();

          // Collectively wait for the operations commited to the
          // previous `compute` stage to complete:
          pipeline.consumer_wait();
	  
	  k_start = k_prev;
	  z_start = z_prev;
	  k_prev =  shared_offset[copy_stage_idx];
	  z_prev = i; 

	  for (int k = k_start, z = z_start; k < (shared_offset[compute_stage_idx] + (numofDataperBatch)) && z < NUMDATA; k++, z++) {
              //if (z >= gid) {
               float x_co =  (ref_x - locations[k].x);
	       float y_co =  (ref_y - locations[k].y);
               float pow_xco = powf(x_co, 2.0);
               float pow_yco = powf(y_co, 2.0);
       	       float pow_plus = powf((pow_yco+pow_xco), 0.5);
               euclideanDistance[index+z] = pow_plus;
                 //long long int symData = (z * NUMDATA) + gid;
                // if (z != gid && symData < ( NUMDATA* NUMDATA)) { 
                  //   euclideanDistance[symData] = pow_plus;
                // }
	      //}
          }  
         // Collectively release the stage resources
	 pipeline.consumer_release();
	 
      }
     
     // Compute the data fetch by the last iteration
     pipeline.consumer_wait();

     for (int k = k_start, z = z_start; k < (shared_offset[compute_stage_idx] + (numofDataperBatch)) && z < NUMDATA; k++, z++) {
          // if (z >= gid) {
           float x_co =  (ref_x - locations[k].x);
	   float y_co =  (ref_y - locations[k].y);
           float pow_xco = powf(x_co, 2.0);
           float pow_yco = powf(y_co, 2.0);
       	   float pow_plus = powf((pow_yco+pow_xco), 0.5);
           euclideanDistance[index+z] = pow_plus;
             // long long int symData = (z * NUMDATA) + gid;
           // if (z != gid && symData < ( NUMDATA* NUMDATA)) { 
              //  euclideanDistance[symData] = pow_plus;
           // }
	// }
     }
     //compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
     pipeline.consumer_release();
   
   */
   }
   

    
}
