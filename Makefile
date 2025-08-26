CC=nvcc

all: euclideanMatrix euclideanMatrixNoPin 

euclideanMatrix: main.cu PrimAlgorithm.o
	$(CC) PrimAlgorithm.o main.cu -o euclideanMatrix

euclideanMatrixNoPin: mainNoPin.cu PrimAlgorithm.o
	$(CC) PrimAlgorithm.o mainNoPin.cu -o euclideanMatrixNoPin

PrimAlgorithm.o: PrimAlgorithm.cu PrimAlgorithm.h
	$(CC) -c PrimAlgorithm.cu -o PrimAlgorithm.o


clean:
	rm -f euclideanMatrix euclideanMatrixNoPin PrimAlgorithm.o

