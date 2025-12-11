//
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


// the graph class
class Graph_n_n {
   private:
      float* x_co;
      float* y_co;
      int* vertex;
      float total_distance;
      int num_of_vert;
      float** euc_mat;
      bool* visit_vert;
      int*  parent;
   public:
      Graph_n_n(float* x_co, float* y_co, int num_of_vert);
     ~Graph_n_n();
     int* return_vertex();
     float return_total_d();
     void bool_fill();
     void euclidean_matrix();
     float distance(int x, int y);
     void cpu_prim_MST();
     void prim_MST();

};


__global__  void Nearest_neigbhor( float *Xcs, float *Ycs,  float *N_N, int NUMDATAS);

__global__ void Fill_bool(bool *Y);
