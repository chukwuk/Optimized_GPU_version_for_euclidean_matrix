// System includes
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
#include "fun_glo.h"
#include "prim_algorithm.cu"

using namespace std;

// function prototypes:
void		TimeOfDaySeed( );

int
main( int argc, char* argv[ ] )
{        
        // read in the text files
        std::ifstream infile;
        infile.open ("test-input-1.txt");
	//infile.open("tsp_example_3.txt");
        //infile.open("data_2.txt");
        int NUMDATAS;
	infile >> NUMDATAS;
        TimeOfDaySeed( );

	int dev = findCudaDevice(argc, (const char **)argv);
        
	// allocate host memory:

	float *hXcs  = new float[NUMDATAS];
	float *hYcs  = new float[NUMDATAS];
        
        float *kc  = new float[NUMDATAS];
	for (int i = 0; i < NUMDATAS; i++) {
	    infile >> kc[i] >> hXcs[i] >> hYcs[i];
         //    fprintf (stderr, "%10.4f\n", hYcs[i]);
        }

	//instantiation of the object of the graph class
        Graph_n_n graph(hXcs, hYcs, NUMDATAS);
        //float** hN_N = graph.euclidean_matrix();
        //graph.cpu_prim_MST();
        
       	cudaError_t status;
	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
	checkCudaErrors( status );
	status = cudaEventCreate( &stop );
	checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
	checkCudaErrors( status );

	// execute the prim algorithm function 

        graph.prim_MST();	
  
	// record the stop event:

	status = cudaEventRecord( stop, NULL );
	checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
	checkCudaErrors( status );

        // wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
	checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
	checkCudaErrors( status );

       // fprintf(stderr, "%10.3f\n", msecTotal*1000000);

	int* vet = graph.return_vertex();
	float total_d =  graph.return_total_d();
        FILE *fp1;
	fp1 = fopen("./distance.txt","w+" );
	fprintf(fp1, "The total distance of the shortest route %10.3f\n", total_d);
	for (int i = 0; i < NUMDATAS; i++) {
            fprintf(fp1, "%d\n", vet[i]);
	}
        fclose(fp1);


	 	return 0;

}
