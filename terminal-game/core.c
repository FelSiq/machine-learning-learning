#include "./source/core.h"
#include "./source/resources.h"
#include "./source/player.h"
#include "./source/commands.h"
#include "./source/structure.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define THREAD_NUM 3

void decodify(byte *b){

};

int main(int argc, char const *argv[]){
	GAME *game = ginit();

	//system("aplay -q ./snd/s0");
	if (game != NULL){
		#ifdef DEBUG
			printf("D: successfully created GAME structure.\n");
		#endif
		game->world->chsetup(game->world);
		//PARALLELS
		pthread_t *process = malloc(sizeof(pthread_t) * THREAD_NUM);
		if (process != NULL){
			bool **returnvals = malloc(sizeof(bool *) * THREAD_NUM);
			if (returnvals != NULL){
				byte sum = 0;
				#ifdef DEBUG
					printf("D: will start multithreading now...\n");
				#endif
				//Multithread section
				sum += pthread_create((process + 0), NULL, game->world->wgetlabels, (void *) game->world);
				sum += pthread_create((process + 1), NULL, game->world->wload, (void *) game->world);
				sum += pthread_create((process + 2), NULL, game->world->isetup, (void *) game->world);
				if (sum != 0){
					printf("E: can't create threads. abort.\n");
					for(byte i = THREAD_NUM; i > 0; --i)
						pthread_cancel(*(process + i - 1));
				} else {
					#ifdef DEBUG
						printf("D: successfully created threads. Will now join then.\n");
					#endif
					for(byte i = THREAD_NUM; i > 0; --i)
						pthread_join(*(process + i - 1), (void **) (returnvals + i - 1));
				};
		
				sum = 0;
				for(byte i = THREAD_NUM; i > 0; sum += *(*(returnvals + i - 1)), free(*(returnvals + i - 1)), --i);
				free(returnvals);
				if (sum == THREAD_NUM){
					#ifdef DEBUG
						printf("D: successfully constructed WORLD structure.\n");
					#endif
				} else printf("E: something went wrong in WORLD setup on \"%s\". abort.\n", __FUNCTION__);
			} else printf("E: failed to init \"returnvals\" on \"%s\".\n", __FUNCTION__);
			free(process);
		} else printf("E: failed to init \"process\" on \"%s\".\n", __FUNCTION__);
	} else printf("E: failed to init GAME STRUCTURE on \"%s\".\n", __FUNCTION__);
	
	if (gdestroy(&game))
		return 0;

	printf("E: something went wrong on GAME structure destruction.\n");
	return 1;
};