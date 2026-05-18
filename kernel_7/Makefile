CC=nvcc --maxrregcount=255 

all: main

main: main.cu 
	$(CC) main.cu -o euclideanMatrix

clean:
	rm -f euclideanMatrix
