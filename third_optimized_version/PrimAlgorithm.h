//


#ifndef __PRIMALGORITHM_H
#define __PRIMALGORITHM_H

// the graph class
struct LocationPrim {
      float x;
      float y;

};


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA);

__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA, int numDataPerThread, int blocksize);

__global__  void euclideanMatrixStaticSharedMemory(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA, int blocksize);
#endif
