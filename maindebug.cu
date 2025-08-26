#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>


#include "PrimAlgorithm.h"

using namespace std;

inline
cudaError_t checkCuda(cudaError_t result)
{
//#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}

int
main( int argc, char* argv[ ] )
{ 
  //srand(time(0));
  unsigned int NUMDATA = 10000;
  const unsigned int bytes = NUMDATA * sizeof(LocationPrim);
  const unsigned int bytes4euc = (NUMDATA * NUMDATA * sizeof(float));
  float time = 1.0;
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
  
  float *HdistanceBtwAllLocation = new float[NUMDATA * NUMDATA];
   
  int BLOCKSIZE = 128;
  int NUMBLOCKS = NUMDATA/BLOCKSIZE;
  if (NUMDATA % BLOCKSIZE != 0) {
     NUMBLOCKS+=1;
  }
    
  fprintf (stderr, "NUMBER OF BLOCKS is %d\n", NUMBLOCKS);

  cudaError_t status;
  //status = cudaMalloc( (void **)(&distanceBtwAllLocation), bytes4euc);
  status = cudaMalloc( (void **)(&distanceBtwAllLocation), bytes4euc);
 // checkCudaErrors( status );
  checkCuda (status);

  cudaMalloc( (void **)(&cordinateLocation), bytes);
  //checkCudaErrors( status );

  // copy host memory to the device:

  //status = cudaMemcpy(distanceBtwAllLocation, this->visit_vert, this->num_of_vert*sizeof(bool), cudaMemcpyHostToDevice );
 // cudaMemcpy(distanceBtwAllLocation, this->visit_vert, bytes4euc, cudaMemcpyHostToDevice );
  cudaMemcpy(cordinateLocation, locate, bytes, cudaMemcpyHostToDevice );
  // checkCudaErrors( status );  
   
  
  dim3 threads(BLOCKSIZE, 1, 1 );
  dim3 grid(NUMBLOCKS, 1, 1 );

  
  euclideanMatrix<<< grid, threads >>>( cordinateLocation, distanceBtwAllLocation,   NUMDATA);
   
  cudaMemcpy(HdistanceBtwAllLocation, distanceBtwAllLocation, bytes4euc, cudaMemcpyDeviceToHost);  
  
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes4euc*1e-9/time);
  double check = 99999.000*100000.000;
  printf("  Device to Host bandwidth (GB/s): %f\n", check);
  printf("  Device to Host bandwidth (GB/s): %f\n", HdistanceBtwAllLocation[NUMDATA*(NUMDATA-1)] / time);
  printf("  Device to Host bandwidth (GB/s): %f\n", HdistanceBtwAllLocation[NUMDATA-1] / time);

  cudaFree( distanceBtwAllLocation );
  cudaFree( cordinateLocation ); 
  
  delete[] locate;
  delete[] HdistanceBtwAllLocation;

  return 0;

};	
