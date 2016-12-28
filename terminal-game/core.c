#include "core.h"
#include "resources.h"
#include "player.h"
#include "commands.h"
#include "structure.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int main(int argc, char const *argv[]){
	GAME *game = ginit();
	byte i = 6;

	if (game != NULL){
		game->world->chsetup(game->world);
		while(0 < i--){
			game->command->get_command(game->command);
			game->command->mem_dump(game->command);
		};
	};
	gdestroy(&game);
	return 0;
};