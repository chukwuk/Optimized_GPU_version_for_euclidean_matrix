CC=nvcc

all: main main1

main: main.cu threadsinWarp.o 
	$(CC) threadsinWarp.o main.cu -o main

threadsinWarp.o: threadsinWarp.cu threadsinWarp.h
	$(CC) -c threadsinWarp.cu -o threadsinWarp.o

main1: main1.cu threadsinWarp2D.o 
	$(CC) threadsinWarp2D.o main1.cu -o main1

threadsinWarp2D.o: threadsinWarp2D.cu threadsinWarp2D.h
	$(CC) -c threadsinWarp2D.cu -o threadsinWarp2D.o


clean:
	rm -f main threadsinWarp.o main1 threadsinWarp2D.o
