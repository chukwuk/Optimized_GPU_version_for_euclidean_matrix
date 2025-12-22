//


#ifndef __PRIMALGORITHM_H
#define __PRIMALGORITHM_H

// the graph class
struct LocationPrim {
      float x;
      float y;

};


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA);

__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread);

__global__  void euclideanMatrixStaticSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int blocksize);
#endif
