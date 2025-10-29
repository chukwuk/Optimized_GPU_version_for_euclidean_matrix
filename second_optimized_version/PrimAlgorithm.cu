#include <math.h>
#include "PrimAlgorithm.h"    
#include <stdio.h>

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


__global__  void euclideanMatrixSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA) {
    
   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
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
