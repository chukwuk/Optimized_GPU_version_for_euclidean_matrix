//


#ifndef __PRIMALGORITHM_H
#define __PRIMALGORITHM_H

// the graph class
struct LocationPrim {
      float x;
      float y;

};

__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread);

#endif
