#include "./source/core.h"
#include "./source/resources.h"
#include "./source/player.h"
#include "./source/commands.h"
#include "./source/structure.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#define THREAD_NUM 3

void decodify(byte *b){

};

void codify(byte *b){

};

static bool integrity_test(){
	//ls -F |grep -v / | wc -l
	return TRUE;
	err_exit;
};

int main(int argc, char const *argv[]){
	if (integrity_test()){
		GAME *game = ginit();
		//system("aplay -q ./snd/s0");
		if (game != NULL){
			#ifdef DEBUG
				printf("D: successfully created GAME structure.\n");
				if (argc > 1 && strcmp(*(argv + 1), "--ddebug-level") == 0 
					&& **(argv + 2) >= 49 && **(argv + 2) <= 51){
						game->ddebug_lvl = (**(argv + 2) - 48);
						printf("D: defined ddebug-level as %hu.\n", game->ddebug_lvl);
				};
			#endif
			//Load chamber data
			if(game->world->chsetup(game->world)){
				//Load global command
				if(game->command->loadglobal(game->command, "./source/globalc")){
					if(!game->command->loadfails(game->command, "./source/failstrings"))
						printf("E: can't load fail strings.\n");
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

							//Test if thread creation works fine.
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
							
							//Test return values of functions.
							sum = 0;
							for(byte i = THREAD_NUM; i > 0; sum += *(*(returnvals + i - 1)), free(*(returnvals + i - 1)), --i);
							free(returnvals);
							if (sum == THREAD_NUM){
								#ifdef DEBUG
									printf("D: successfully constructed WORLD structure.\n");
								#endif
								
								CHAMBER *traveller = *(game->world->allchambers + GLOBALV_PLAYER_STDSTART);
								FILE *fp = stdin;

								#ifndef DEBUG
									system("clear");
								#endif
								//Check if it's the player's first time.
								if (access("./data/pname", R_OK) == -1){
									printf("\rSegmentation fault (core dumped)\n");
									for (uint delay = (1 << (sizeof(uint)*BITS_IN_BYTE - 2)); delay > 0; --delay);
									printf("Brincadeira!\nMe parece que este é o seu primeiro acesso. Qual seu nome?\n");
									system("aplay -q ./snd/s1");
								} else fp = fopen("./data/pname", "r");

								if(game->player->pgetname(game->player, fp)){
									#ifdef DEBUG
										printf("D: got a player new name: \"%s\".\n", game->player->name);
									#endif
								} else printf("E: can't get player's name.\n");

								#ifndef DEBUG
									system("clear");
								#endif
								FILE *interface_pointer = NULL;
								//First things first
								game->ginterfacePre(game, traveller, &interface_pointer);
								if(fp != stdin){
									printf("Bem-vindo de volta, %s!\n", game->player->name);
									//system("aplay -q ./snd/s2");
									fclose(fp);
								} else {
									FILE *wfile = fopen("./source/wtext", "r");
									if (wfile != NULL){
										while(!feof(wfile)){
											char *wtext = get_string(wfile);
											if (wtext != NULL){
												for(uint k = 0; *(wtext + k) != '\0'; decodify((byte *) (wtext + k++)));
												printf("%s\n\n", wtext);
												free(wtext);
											};
										};
										fclose(wfile);
									};
								};
								game->ginterfacePos(game, traveller, &interface_pointer);

								//Everyting is set up. Game can now start.
								while(!game->END_FLAG){
									while(!game->command->get_command(game->command));
									system("clear");
									game->ginterfacePre(game, traveller, &interface_pointer);
									game->command->cprocess(game, &traveller, game->command->memory);
									game->ginterfacePos(game, traveller, &interface_pointer);
								};
								//Finishing the game
								printf("\n");
								system("clear");
								printf("Até logo, %s!\n", game->player->name);
								// Saves player progress before games end.
								if(game->grefresh(game)){
									#ifdef DEBUG
										printf("D: saved player's progress.\n");
									#endif
								} else printf("E: can't save new game data. Progress maybe is lost.\n");
							} else printf("E: something went wrong in WORLD setup on \"%s\". abort.\n", __FUNCTION__);
						} else printf("E: failed to init \"returnvals\" on \"%s\".\n", __FUNCTION__);
						free(process);
					} else printf("E: failed to init \"process\" on \"%s\".\n", __FUNCTION__);
				} else printf("E: failed to load global commands on \"%s\".\n", __FUNCTION__);
			} else printf("E: failed to init chamber setup on \"%s\".\n", __FUNCTION__);
		} else printf("E: failed to init GAME STRUCTURE on \"%s\".\n", __FUNCTION__);
		
		//Destroy GAME main structure.
		if (gdestroy(&game))
			return 0;

		printf("E: something went wrong on GAME structure destruction.\n");
	} else printf("E: seens like you don't have everything necessary to run the game. Try download it again.");
	return 1;
};