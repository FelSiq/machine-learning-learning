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
		game->world->allchambers[0]->adjch_setup(game->world->allchambers[0], 1, game->world->allchambers[6]);
		game->world->allchambers[6]->adjch_setup(game->world->allchambers[6], 1, game->world->allchambers[7]);
		printf("%d/%d/%d\n", game->world->allchambers[0]->adjnum, game->world->allchambers[6]->adjnum, game->world->allchambers[7]->adjnum);
		/*while(0 < i--){
			game->command->get_command(game->command);
			game->command->mem_dump(game->command);
		};*/
	};
	gdestroy(&game);
	return 0;
};