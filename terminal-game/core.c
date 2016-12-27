#include "core.h"
#include "resources.h"
#include "player.h"
#include "commands.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int main(int argc, char const *argv[]){
	PLAYER *p = pinit();
	COMMAND *command = cinit();
	byte i = 6;
	if (command != NULL){
		while(0 < i--){
			(*command).get_command(command);
			(*command).mem_dump(command);
		};
		cdestroy(&command);
	};
	pdestroy(&p);
	return 0;
};