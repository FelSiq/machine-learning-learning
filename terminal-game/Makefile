all: resources.o commands.o player.o structure.o core.o
	gcc -g resources.o commands.o player.o core.o structure.o -o tridle -lpthread -Wall -DDEBUG
	rm *.o
resources.o:
	gcc -c -g ./source/resources.c -Wall -DDEBUG
commands.o:
	gcc -c -g ./source/commands.c -Wall -DDEBUG
player.o:
	gcc -c -g ./source/player.c -Wall -DDEBUG
structure.o:
	gcc -c -g ./source/structure.c -Wall -DDEBUG
core.o:
	gcc -c -g core.c -lpthread -Wall -DDEBUG
clean:
	rm *.o
cleanall:
	rm *.o tridle
