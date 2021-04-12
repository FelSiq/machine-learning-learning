all: ht.o kmeans.o font.o clean
font.o:
	gcc ht.o kmeans.o -o kmeans -g -lm -Wall
ht.o:
	gcc -c -g ./src/ht.c -Wall
kmeans.o:
	gcc -c -g ./src/kmeans.c -lm -Wall
clean:
	rm *.o
