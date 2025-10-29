//
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <limits>
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"
#include "prim_algorithm.h"


//GPU  Kernel to calculation the euclidean matrix
__global__  void Nearest_neigbhor( float *Xcs, float *Ycs,  float *N_N, int NUMDATAS)
{
    int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
    int blockThreads = blockNum*blockDim.x*blockDim.y;
    int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
//    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int x_index = gid/NUMDATAS;
    int y_index = gid % NUMDATAS;

    float x_co =  (Xcs[x_index]-Xcs[y_index]);
    float y_co =  (Ycs[x_index]-Ycs[y_index]);
    float pow_xco = powf(x_co, 2.0);
    float pow_yco = powf(y_co, 2.0);
    float pow_plus = powf((pow_yco+pow_xco), 0.5);
    N_N[gid] = pow_plus;
}


// GPU Kernel to fill an array with zeros

__global__ void Fill_bool(bool *Y)  
{
 int gid = blockIdx.x*blockDim.x + threadIdx.x;
 Y[gid] = 0;

}


// instantiation of the object
Graph_n_n::Graph_n_n (float* x_co, float* y_co, int num_of_vert) {
    this->x_co = x_co;
    this->y_co = y_co;
    this->num_of_vert = num_of_vert;
   
}


//returns array of vertices with the shortest route
int* Graph_n_n::return_vertex() {
    return this->vertex;
}

//returns the total distance 
float Graph_n_n::return_total_d() {
     return this->total_distance;
}


// function that fills the array with zeros
void Graph_n_n::bool_fill() 
{
        // allocating device memory
        bool *dXcs;

	this->visit_vert = new bool [this->num_of_vert];
       

	int BLOCKSIZE = this->num_of_vert;
	int NUMBLOCKS = 1;
//        int BLOCKSIZE = 32;
//	int NUMBLOCKS = 472;

	cudaError_t status;
	status = cudaMalloc( (void **)(&dXcs), this->num_of_vert*sizeof(bool) );
	checkCudaErrors( status );

	// copy host memory to the device:

	status = cudaMemcpy(dXcs, this->visit_vert, this->num_of_vert*sizeof(bool), cudaMemcpyHostToDevice );
	checkCudaErrors( status );  


	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid(NUMBLOCKS, 1, 1 );

	// create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
	checkCudaErrors( status );
	status = cudaEventCreate( &stop );
	checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
	checkCudaErrors( status );

        // execute the kernel:

        Fill_bool<<< grid, threads >>>( dXcs);

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
	checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
	checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
	checkCudaErrors( status ); 
       
	// copy result from the device to the host:
        
        status = cudaMemcpy( this->visit_vert, dXcs, this->num_of_vert*sizeof(bool), cudaMemcpyDeviceToHost );
        checkCudaErrors( status );
 
        status = cudaFree( dXcs );
	checkCudaErrors( status );



}


//the function to execute the calculation of the euclidean matrix
void Graph_n_n::euclidean_matrix()  {      
        // allocating device memory
        float *dXcs, *dYcs;
	float *dN_N;

	this->euc_mat = new float* [this->num_of_vert];
        for (int i = 0; i < this->num_of_vert; i++) {
          euc_mat[i] = new float [this->num_of_vert];
	}

	//int BLOCKSIZE = this->num_of_vert*this->num_of_vert;
	int BLOCKSIZE = this->num_of_vert;
	int NUMBLOCKS = this->num_of_vert;
//        int BLOCKSIZE = 32;
//	int NUMBLOCKS = 472;

	cudaError_t status;
	status = cudaMalloc( (void **)(&dXcs), this->num_of_vert*sizeof(float) );
	checkCudaErrors( status );

	status = cudaMalloc( (void **)(&dYcs), this->num_of_vert*sizeof(float) );
	checkCudaErrors( status );

	status = cudaMalloc( (void **)(&dN_N), this->num_of_vert*this->num_of_vert*sizeof(float) );
        checkCudaErrors( status );



	checkCudaErrors( status );


	// copy host memory to the device:

	status = cudaMemcpy(dXcs, this->x_co, this->num_of_vert*sizeof(float), cudaMemcpyHostToDevice );
	checkCudaErrors( status );

	status = cudaMemcpy( dYcs, this->y_co, this->num_of_vert*sizeof(float), cudaMemcpyHostToDevice );
        checkCudaErrors( status );
          


	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid(NUMBLOCKS, 1, 1 );

	// create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
	checkCudaErrors( status );
	status = cudaEventCreate( &stop );
	checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
	checkCudaErrors( status );

        // execute the kernel:

        Nearest_neigbhor<<< grid, threads >>>( dXcs, dYcs, dN_N, this->num_of_vert);

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
	checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
	checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
	checkCudaErrors( status ); 
        int num = this->num_of_vert;
	// copy result from the device to the host:
        for (int i = 0; i < this->num_of_vert; i++) {
        status = cudaMemcpy( (&(this->euc_mat[i][0])), &(dN_N[i*num]), this->num_of_vert*sizeof(float), cudaMemcpyDeviceToHost );
        checkCudaErrors( status );
	}
 
        status = cudaFree( dXcs );
	status = cudaFree( dYcs ); 
	status = cudaFree ( dN_N );


	checkCudaErrors( status );

      //	return euc_mat;

}

float Graph_n_n::distance(int x, int y) {
        float dist = (this->x_co[x] - this->x_co[y]);
	float dist_1 = (this->y_co[x] - this->y_co[y]);
        float dist_sqr = pow(dist, 2.0);
	float dist_sqr_1 = pow(dist_1, 2.0);
	float res = pow((dist_sqr + dist_sqr_1), 0.5);
	return res;
     
}



//impementing the prim algorithm on only CPU
void Graph_n_n::cpu_prim_MST()  {
        this->visit_vert = new bool[this->num_of_vert];
     	for (int i = 0; i<this->num_of_vert; i++) {
             this->visit_vert[i] = 0; 
	}
	
	//this->bool_fill();
	this->total_distance = 0.0;
        this->parent = new int [this->num_of_vert];
	this->vertex = new int [this->num_of_vert];
        parent[0] = -1;
	int v = 0;
	/*
	for (int i = 0; i < this->num_of_vert; i++) {
	    for (int j = 0; j < this->num_of_vert; j++) {
                fprintf( stderr, "%10.4f\n",this->distance(i,j));                               
	    }
	}
	*/
	for (int i = 0; i < this->num_of_vert-1; i++) {
      	    this->visit_vert[v] = 1;
            this->vertex[i] = v;
	   // fprintf(stderr,"%d\n", v);
	    float min_dist =  3.3*pow(10, 37);
            int min_index = 0;
            for (int j = 0; j < this->num_of_vert; j++) {       
		if ((this->distance(v,j) > 0.0)  && (min_dist > this->distance(v,j)) && this->visit_vert[j]==0 ) {	
                      min_dist = this->distance(v,j);
		      min_index = j;
		     
                     }
            }
	    parent[min_index] = v;
	    v = min_index;
            this->total_distance+=min_dist;
	    if (i == this->num_of_vert-1) {
                this->vertex[i+1] = v;
	    }

	}
	

}


//implement the prim algorithm using some GPU function
void Graph_n_n::prim_MST() {
           
	/*this->visit_vert = new bool[this->num_of_vert];
     	for (int i = 0; i<this->num_of_vert; i++) {
             this->visit_vert[i] = 0; 
	}
	*/
	this->bool_fill();
	this->total_distance = 0.0;
        this->parent = new int [this->num_of_vert];
	this->vertex = new int [this->num_of_vert];
	this->euclidean_matrix();
	FILE *fp;
	fp = fopen("./texting.txt","w+" );
        for (int i = 0; i < this->num_of_vert; i++) {
	       for (int j = 0; j < this->num_of_vert; j++) {
	     //fprintf( stderr, "%10.4f\n",this->euc_mat[i][j]);
	       fprintf(fp, "%5.4f  ", this->euc_mat[i][j]);
               }
               fprintf(fp, "\n");
        } 
	fclose(fp);
        parent[0] = -1;
	int v = 0;
	for (int i = 0; i < this->num_of_vert-1; i++) {
      	    this->visit_vert[v] = 1;
            this->vertex[i] = v;
	   // fprintf(stderr,"%d\n", v);
	    float min_dist =  3.3*pow(10, 37);
            int min_index = 0;
            for (int j = 0; j < this->num_of_vert; j++) {       
		if ((this->euc_mat[v][j] > 0.0)  && (min_dist > this->euc_mat[v][j]) && this->visit_vert[j]==0 ) {	
                      min_dist = this->euc_mat[v][j];
		      min_index = j;
		     
                     }
            }
	    parent[min_index] = v;
	    v = min_index;
            this->total_distance+=min_dist;
	    if (i == this->num_of_vert-1) {
                this->vertex[i+1] = v;
	    }

	}

}

Graph_n_n::~Graph_n_n() {
     for (int i = 0; i<num_of_vert; i++) {
         delete[] this->euc_mat[i];

     }
     delete[] this->euc_mat;
     delete[] this->x_co;
     delete[] this->y_co;
     delete[] this->vertex;
     delete[] this->visit_vert;
     delete[] this->parent;

}
