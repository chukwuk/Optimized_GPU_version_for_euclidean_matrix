CC=nvcc

all: euclideanMatrix

euclideanMatrix: main.cu PrimAlgorithm.o
	$(CC) PrimAlgorithm.o main.cu -o euclideanMatrix

PrimAlgorithm.o: PrimAlgorithm.cu PrimAlgorithm.h
	$(CC) -c PrimAlgorithm.cu -o PrimAlgorithm.o


clean:
	rm -f euclideanMatrix PrimAlgorithm.o
