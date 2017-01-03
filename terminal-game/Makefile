CPPFLAGS = -Wall

all: resources.o commands.o player.o structure.o core.o
	gcc -g resources.o commands.o player.o core.o structure.o -o tridle -lpthread $(CPPFLAGS)
	rm *.o
resources.o:
	gcc -c -g ./source/resources.c $(CPPFLAGS)
commands.o:
	gcc -c -g ./source/commands.c $(CPPFLAGS)
player.o:
	gcc -c -g ./source/player.c $(CPPFLAGS)
structure.o:
	gcc -c -g ./source/structure.c $(CPPFLAGS)
core.o:
	gcc -c -g core.c -lpthread $(CPPFLAGS)
clean:
	rm *.o
cleanall:
	rm *.o tridle
