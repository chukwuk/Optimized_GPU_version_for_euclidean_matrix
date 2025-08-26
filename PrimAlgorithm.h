//


#ifndef __PRIMALGORITHM_H
#define __PRIMALGORITHM_H

// the graph class
struct LocationPrim {
      float x;
      float y;

};


__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, long long int NUMDATA);


#endif
