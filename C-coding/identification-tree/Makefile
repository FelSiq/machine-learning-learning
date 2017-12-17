CC = gcc
CFLAGS = -Wall
LDLIBS = -lm

all: it.o interface.o
	$(CC) -std=c99 it.o interface.o -o it $(LDLIBS) $(CFLAGS)
	rm *.o
it.o:
	$(CC) -std=c99 -c ./src/it.c -Wall $(LDLIBS) $(CFLAGS)
interface.o:
	$(CC) -std=c99 -c ./src/interface.c $(LDLIBS) $(CFLAGS)