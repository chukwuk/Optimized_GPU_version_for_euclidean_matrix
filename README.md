RUNTIME COMPARISON BETWEEN GPU(CUDA) AND CPU IMPLEMENTATION OF EUCLIDEAN MATRIX CALCULATION. 

Euclidean matrix has many application in Prim algorithm and Dijkstra algorithm. The runtime comparison shows that the GPU calculation of the eculidean matrix is faster the CPU calculation. The GPU calculation is faster when you using CudaMallocHost to pin the data before transferring the data to the CPU. The Makefile compiles two codes, in which one of the code(euclideanmatrix) uses CudaMallocHost and the other codes does not use CudaMallocHost (euclideanmatrixNoPin). 
