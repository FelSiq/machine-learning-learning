#include "core.h"
#include "resources.h"
#include "player.h"
#include "commands.h"
#include "structure.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void decodify(byte *b){

};

int main(int argc, char const *argv[]){
	GAME *game = ginit();
	if (game != NULL){
		game->world->chsetup(game->world);
		game->world->wgetlabels(game->world);
		game->world->wload(game->world);
	};
	gdestroy(&game);
	return 0;
};