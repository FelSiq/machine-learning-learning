all: resources.o commands.o player.o structure.o core.o
	gcc -g resources.o commands.o player.o core.o structure.o -o v0_terminal -Wall -DDEBUG
resources.o:
	gcc -c -g resources.c -Wall -DDEBUG
commands.o:
	gcc -c -g commands.c -Wall -DDEBUG
player.o:
	gcc -c -g player.c -Wall -DDEBUG
structure.o:
	gcc -c -g structure.c -Wall -DDEBUG
core.o:
	gcc -c -g core.c -Wall -DDEBUG
clean:
	rm *.o v0_terminal
