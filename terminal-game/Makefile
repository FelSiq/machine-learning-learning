all: resources.o commands.o player.o core.o
	gcc resources.o commands.o player.o core.o -o v0_terminal -Wall
resources.o:
	gcc -c resources.c -Wall
commands.o:
	gcc -c commands.c -Wall
player.o:
	gcc -c player.c -Wall
core.o:
	gcc -c core.c -Wall
clean:
	rm *.o v0_terminal
