#include "core.h"
#include "resources.h"
#include "structure.h"
#include "commands.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

typedef struct arg_struct {
	bool FLAG;
	char *string;
	CHAMBER **tvl;
	GAME *g;
} arg_struct;

void string_uppercase(char *s){
	if (s != NULL)
		for(uint i = 0; *(s + i) != '\0'; ++i)
			if (*(s + i) >= ASCIIa && *(s + i) <= ASCIIz)
				*(s + i) -= MODULUS(ASCIIa - ASCIIA);
};

void string_lowercase(char *s){
	if (s != NULL)
		for(uint i = 0; *(s + i) != '\0'; ++i)
			if (*(s + i) >= ASCIIA && *(s + i) <= ASCIIZ)
				*(s + i) += MODULUS(ASCIIa - ASCIIA);
};

static void *cprocess_global(void *vas){
	arg_struct *as = (arg_struct *) vas;
	bool *ret = malloc(sizeof(bool));
	if (ret != NULL && as != NULL){
		*ret = FALSE;
		GAME *g = as->g;
		CHAMBER *tvl = *(as->tvl);
		char *string = as->string;
		//Test global commands
		for(byte i = g->command->gcnum; as->FLAG && !(*ret) && (i > 0); --i){
			if (strcmp(string, *(g->command->gcommands + i -1)) == 0){
				(*ret) = TRUE;
				if (strcmp(string, "sair") == 0){
					g->END_FLAG = TRUE;
				} else if (strcmp(string, "observar") == 0){
					printf("(Você observa ao seu redor, em \"%s\", e consegue notar...)\n", tvl->string);
					if (tvl->adjchambers != NULL){
						CHAMBER *aux;
						for(byte j = tvl->adjnum; j > 0; --j){
							aux = (*(tvl->adjchambers + j - 1))->a;
							if (aux == tvl)
								aux = (*(tvl->adjchambers + j - 1))->b;
							
							printf("Caminho para \"%s\"", aux->string);

							if (!(*(tvl->adjchambers + j - 1))->open){
								if ((*(tvl->adjchambers + j - 1))->string != NULL)
									printf(" (%s)", (*(tvl->adjchambers + j - 1))->string);
								else
									printf(" (interditado)");
							};
							printf("\n");
						};
					};
					if (tvl->iactives != NULL){
						for(byte j = tvl->actnum; j > 0; --j){
							printf("\"%s\"", (*(tvl->iactives + j - 1))->label);
							if ((*(tvl->iactives + j - 1))->progress > 0)
								printf(" [tarefa em progresso...]");
							printf("\n");
						};
					} else printf("e nenhum objeto interessante.\n");
				} else if (strcmp(string, "notas") == 0){
					if (!list_empty(g->player->notes)){
						list_print(g->player->notes);
					} else printf("Você não possui nenhuma tarefa ativa no momento.\n");
				} else if (strcmp(string, "mochila") == 0){
					FILE *fp = fopen("./data/tmeasures", "r");
					uint aux = 0, val;
					if (fp != NULL){
						fscanf(fp, "%u%*c", &aux);
						printf("(Você olha dentro de sua mochila e...)\n");
						for(byte i = 0; i < GLOBALV_PINV_STDSIZE; ++i){
							printf("[");
							val = strlen(*(g->player->colnames + *(g->player->colectibles + i)));
							if (val < aux/GLOBALV_BACKPACK_LINES)
								for(byte j = 0; j < (aux/GLOBALV_BACKPACK_LINES - val + 1)/2 - 1; printf(" "), ++j);
							if (g->player->colnamnum > *(g->player->colectibles + i))
								printf("%s", *(g->player->colnames + *(g->player->colectibles + i)));
							else
								printf("???");
							if (val < aux/GLOBALV_BACKPACK_LINES)
								for(byte j = 0; j < (aux/GLOBALV_BACKPACK_LINES - val)/2 - 1; printf(" "), ++j);
							printf("]");
							if (((i + 1) % GLOBALV_BACKPACK_LINES) == 0)
								printf("\n");
						};
						fclose(fp);
					} else printf("E: can't access \"./data/tmeasures\" on %s.", __FUNCTION__);
				} else if (strcmp(string, "salvar") == 0){
					if (g->grefresh(g))
						printf("Progresso salvo com sucesso.\n");
					else
						printf("Algo deu errado, seu progresso não pôde ser salvo.\n");
				} else if (strcmp(string, "mapa") == 0){
					byte counter = 0, counter2 = 0;
					FILE *wfile = fopen("./source/gmap", "r");
					CHAMBER *chaux = NULL;
					if (wfile != NULL){
						while(!feof(wfile)){
							char *wtext = get_string(wfile);
							if (wtext != NULL){
								for(uint k = 0; *(wtext + k) != '\0'; decodify((byte *) (wtext + k++)));
								
								while(chaux == NULL && counter < g->world->nused)
									chaux = (*(g->world->allchambers + counter2++));

								if (chaux != NULL){
									printf("%s\t%hu. %s", wtext, (counter + 1), chaux->string);
									if (strcmp(chaux->string, tvl->string) == 0)
										printf(" (e aqui estamos!)");
								};

								printf("\n");
								chaux = NULL;
								++counter;
								free(wtext);
							};
						};
					fclose(wfile);
					};
				};
			};
		};
		//End of global testing
	};
	return ret;
};

char *string_copy(char *s){
	uint size = strlen(s);
	char *ns = malloc(sizeof(char) * (size + 1));
	if (ns != NULL){
		*(ns + size) = '\0';
		while(0 < size--)
			*(ns + size) = *(s + size);
	};
	#ifdef DEBUG
		printf("D: copied string (strcmp() = %d):\n\"%s\"\n\"%s\"\n", strcmp(s, ns), s, ns);
	#endif
	return ns;
};

static bool recursive_stackcompare(STACK *a, STACK *b, bool *FLAG){
	if (FLAG){
		if (stack_empty(a)/* && stack_empty(b)*/){
			*FLAG = FALSE;
			return TRUE;
		};
		if (/*stack_empty(a) ^ */stack_empty(b))
			return FALSE;
		#ifdef DEBUG
			if(stack_top(a) != NULL)
				printf("D: stack_top(a): \"%s\".\n", stack_top(a));
			if(stack_top(b) != NULL)
				printf("D: stack_top(b): \"%s\".\n", stack_top(b));
		#endif
		if (stack_top(a) != NULL && stack_top(b) != NULL && strcmp(stack_top(a), stack_top(b)) == 0){
			char *sa = stack_pop(a);
			char *sb = stack_pop(b);
			bool res = recursive_stackcompare(a, b, FLAG);

			if (!res){
				stack_push(a, sa);
				stack_push(b, sb);
			} else {
				free(sa);
				free(sb);
			};
			
			return res;
		};
	};
	return FALSE;
};

static COMMAND *command_aux_init(char *s){
	if (s != NULL){
		COMMAND *newc = cinit();
		if (newc != NULL){
			newc->string = string_copy(s);
			size_t val = strlen(newc->string);
			newc->string = realloc(newc->string, sizeof(char) * (val + 2));
			*(newc->string + val) = 32;
			*(newc->string + val + 1) = '\0';
			newc->str_tokenizer(newc);
		};
		return newc;
	};
	return NULL;
};

static void *cprocess_local_iactv(void *vas){
	arg_struct *as = (arg_struct *) vas;
	bool *ret = malloc(sizeof(bool));
	if (ret != NULL && as != NULL){
		CHAMBER *tvl = *(as->tvl);
		GAME *g = as->g;
		(*ret) = FALSE;
		bool auxret, junk;
		short int index;
		//This is, by far, the slowest function in command recognition.
		for (byte i = tvl->actnum; as->FLAG && !(*ret) && i > 0; --i){
			COMMAND *newc = command_aux_init((*(tvl->iactives + i - 1))->label);
			if (newc != NULL){
				if (strcmp(stack_top((newc)->memory), as->string) == 0){
					free(stack_pop((newc)->memory));
					auxret = recursive_stackcompare((newc)->memory, g->command->memory, &as->FLAG);
					if (auxret){
						//Interactive name recognized, proceed.
						#ifdef DEBUG
							printf("D: recognized \"%s\" interactive in %s function.\n", 
								(*(tvl->iactives + i - 1))->label, __FUNCTION__);
						#endif

						if (!stack_empty(g->command->memory)){
							//There's additional command, verify if it exists.
							for (byte j = (*(tvl->iactives + i - 1))->actnum; j > 0; --j){
								COMMAND *auxc = command_aux_init(*((*(tvl->iactives + i - 1))->actions + j - 1));
								if (auxc != NULL){
									auxret = recursive_stackcompare((auxc)->memory, g->command->memory, &junk);
									if (auxret){
										#ifdef DEBUG
											printf("D: recognized command.\n");
										#endif
										//Verify if recognized command has some requeriments
										index = 0;
										if (*((*(tvl->iactives + i - 1))->colreq + j - 1) != -1){
											//Have requeriments
											index = binsearch(g->player->colectibles, GLOBALV_PINV_STDSIZE,
												*((*(tvl->iactives + i - 1))->colreq + j - 1));
											if (index != -1){
												printf("(\"%s\" foi removido da sua mochila.)\n", 
													*(g->player->colnames + *((*(tvl->iactives + i - 1))->colreq + j - 1)));
												*(g->player->colectibles + index) = 0;
											};
										};

										if (index != -1){
											//Requeriments meet, now check for "ask more command"
											if (*((*(tvl->iactives + i - 1))->extracom + j - 1) == NULL){
												//No "ask more command", proceed quest

												printf("%s\n", *((*(tvl->iactives + i - 1))->script + (*(tvl->iactives + i - 1))->progress + 1));
												if ((*(tvl->iactives + i - 1))->progress < ((*(tvl->iactives + i - 1))->actnum - 1)){
													//Check if this command has a reward
													if (*((*(tvl->iactives + i - 1))->rewards + (*(tvl->iactives + i - 1))->progress) != -1){
														g->player->pgetitem(g->player, 
															*((*(tvl->iactives + i - 1))->rewards + (*(tvl->iactives + i - 1))->progress));
														printf("(\"%s\" adicionado á mochila.)\n", 
															*(g->player->colnames + (*(tvl->iactives + i - 1))->progress));
													};

													//Quest still in progress
													++(*(tvl->iactives + i - 1))->progress;
													if ((*(tvl->iactives + i - 1))->progress == ((*(tvl->iactives + i - 1))->actnum - 1)){
														system("aplay -q ./snd/s0");
														++g->player->tasksdone;
														#ifdef DEBUG
															printf("D: a quest is completed!\n");
														#endif
													};
												};
											} else {
												//"ask more command" confirmed, get it.
												printf("Interagindo com %s, o que fazer agora?\n", (*(tvl->iactives + i - 1))->label);
												char *extracommand = get_string(stdin);
												if (extracommand != NULL){
													if (strcmp(extracommand, *((*(tvl->iactives + i - 1))->extracom + j - 1)) == 0){
														printf("%s\n", *((*(tvl->iactives + i - 1))->script + (*(tvl->iactives + i - 1))->progress + 1));
														if ((*(tvl->iactives + i - 1))->progress < ((*(tvl->iactives + i - 1))->actnum - 1)){
															//Check if this command has a reward
															if (*((*(tvl->iactives + i - 1))->rewards + (*(tvl->iactives + i - 1))->progress) != -1){
																g->player->pgetitem(g->player, 
																	*((*(tvl->iactives + i - 1))->rewards + (*(tvl->iactives + i - 1))->progress));
																printf("(\"%s\" adicionado á mochila.)\n", 
																	*(g->player->colnames + (*(tvl->iactives + i - 1))->progress));
															};

															//Quest still in progress
															++(*(tvl->iactives + i - 1))->progress;
															if ((*(tvl->iactives + i - 1))->progress == ((*(tvl->iactives + i - 1))->actnum - 1)){
																system("aplay -q ./snd/s0");
																++g->player->tasksdone;
																#ifdef DEBUG
																	printf("D: a quest is completed!\n");
																#endif
															};
														};
													} else {
														printf("Não funcionou.\n");
													};
													free(extracommand);
												};
											};
										} else {
											//Requeriments not meet.
											printf("Requer \"%s\" para isto.\n", 
												*(g->player->colnames + *((*(tvl->iactives + i - 1))->colreq + j - 1)));
										};
									};
								};
								cdestroy(&auxc);
							};
						} else {
							//There's no additional command, just examine then.
							printf("(Paramos por um momento e examinamos %s...)\n%s\n", 
								(*(tvl->iactives + i - 1))->label,
								*((*(tvl->iactives + i - 1))->script));
						};
						
						//Final stuff
						stack_clear(g->command->memory);
						(*ret) = TRUE;
					};
				};
				cdestroy(&newc);
			};
		};
	};
	return ret;
};

static void *cprocess_local_path(void *vas){
	arg_struct *as = (arg_struct *) vas;
	bool *ret = malloc(sizeof(bool));
	if (ret != NULL && as != NULL){
		CHAMBER *tvl = *(as->tvl), *chaux;
		GAME *g = as->g;
		(*ret) = FALSE;
		//Testing local commands
		for (byte i = tvl->adjnum; as->FLAG && !(*ret) && i > 0; --i){
			//Getting the correct "other" chamber
			chaux = (*(tvl->adjchambers + i - 1))->a;
			if (chaux == tvl)
				chaux = (*(tvl->adjchambers + i - 1))->b;
			
			COMMAND *newc = command_aux_init(chaux->string);
			if (newc != NULL){
				if (strcmp(stack_top(newc->memory), as->string) == 0){
					free(stack_pop(newc->memory));
					(*ret) = recursive_stackcompare(newc->memory, g->command->memory, &as->FLAG);
					if (*ret){
						stack_clear(g->command->memory);
						if ((*(tvl->adjchambers + i - 1))->open){
							if ((*(tvl->adjchambers + i - 1))->string != NULL)
								printf("%s\n", (*(tvl->adjchambers + i - 1))->string);
							printf("Chegamos em um novo local: \"%s\".\n", chaux->string);
							*(as->tvl) = chaux;
						} else printf("Não é possível prosseguir por ali no momento. Talvez devêssemos tentar mais tarde?!\n");
					};
				};
				cdestroy(&newc);
			};
		};
	};
	return ret;
};

static bool cprocess(GAME *g, CHAMBER **tvl, STACK *s){
	if (g != NULL && g->command != NULL && s != NULL && tvl != NULL){
		bool FLAG = FALSE;
		while(!stack_empty(s)){
			char *string = stack_pop(s);
			if (string != NULL){
				#ifdef DEBUG
					printf("D: will process \"%s\"...\n", string);
				#endif
				//PARALLELS
				pthread_t *process = malloc(sizeof(pthread_t) * GLOBALV_THREADNUM_COMMANDS);
				if (process != NULL){
					arg_struct *as = malloc(sizeof(arg_struct));
					if (as != NULL){
						as->g = g;
						as->tvl = tvl;
						as->string = string;
						as->FLAG = TRUE;
						bool **returnvals = malloc(sizeof(bool *) * GLOBALV_THREADNUM_COMMANDS);
						if (returnvals != NULL){
							for(byte i = GLOBALV_THREADNUM_COMMANDS; i > 0; *(returnvals + --i) = NULL);
							#ifdef DEBUG
								printf("D: will start multithreading (in %s)...\n", __FUNCTION__);
							#endif
							byte sum = 0;
							sum += pthread_create((process + 0), NULL, cprocess_global, (void *) as);
							sum += pthread_create((process + 1), NULL, cprocess_local_path, (void *) as);
							sum += pthread_create((process + 2), NULL, cprocess_local_iactv, (void *) as);

							if (sum == 0){
								#ifdef DEBUG
									printf("D: will now join threads...\n");
								#endif

								for(byte i = GLOBALV_THREADNUM_COMMANDS; i > 0; 
									pthread_join(*(process + i - 1), (void **) (returnvals + i - 1)), --i);

								#ifdef DEBUG
									printf("D: will now check results...\n");
								#endif

								for(byte i = GLOBALV_THREADNUM_COMMANDS; i > 0; --i)
									if (*(returnvals + i - 1) != NULL){
										if (!FLAG && **(returnvals + i - 1))
											FLAG = TRUE;
										free(*(returnvals + i - 1));
									};
							} else {
								printf("E: error in thread init in %s. abort.\n", __FUNCTION__);
								for(byte i = GLOBALV_THREADNUM_COMMANDS; i > 0; pthread_cancel(*(process + i - 1)), --i);
							};

							free(returnvals);

							#ifdef DEBUG
								printf("D: end of parallel processing in %s.\n", __FUNCTION__);
							#endif

						} else printf("E: can't malloc \"returnvals\" on %s.", __FUNCTION__);
						free(as);
					} else printf("E: can't create \"arg_struct\" on %s.\n", __FUNCTION__);
					free(process);
				} else printf("E: can't init threads on %s.\n", __FUNCTION__);

				free(string);
			};
		};
		if (!FLAG){
			rprintf(g->command->fail_strings, g->command->failnum);
			printf("\n(Comando inválido)\n");
		};

		return TRUE;
	};
	err_exit;
};

static bool load_failstrings(COMMAND *c, char const path[]){
	if (c != NULL){
		FILE *fp = fopen(path, "r");
		if (fp != NULL){
			char *aux;
			while (!feof(fp)){
				aux = get_string(fp);
				if (aux != NULL){
					c->fail_strings = realloc(c->fail_strings, sizeof(char *) * (c->failnum + 1));
					*(c->fail_strings + c->failnum++) = aux;
					aux = NULL;
				};
			};
			#ifdef DEBUG
				printf("D: loaded fail strings:\n");
				for(byte i = c->failnum; i > 0; printf("%hu. \"%s\"\n", i, *(c->fail_strings + i - 1)), --i);
			#endif
			fclose(fp);
			return TRUE;
			fclose(fp);
		} else printf("E: can't open \"%s\" on %s.\n", path, __FUNCTION__);
	};
	err_exit;
};

static bool load_globalCommands(COMMAND *c, char const path[]){
	if (c != NULL){
		FILE *fp = fopen(path, "r");
		if (fp != NULL){
			char *aux;
			c->gcommands = malloc(sizeof(char *) * GLOBALV_COMMAND_MAXNUM);
			if (c->gcommands != NULL){
				while (!feof(fp) && c->gcnum < GLOBALV_COMMAND_MAXNUM){
					aux = get_string(fp);
					if (aux != NULL)
						*(c->gcommands + c->gcnum++) = aux;
				};
				c->gcommands = realloc(c->gcommands, sizeof(char *) * (c->gcnum));
				#ifdef DEBUG
					printf("D: loaded global commands:\n");
					for(byte i = c->gcnum; i > 0; printf("%hu. \"%s\"\n", i, *(c->gcommands + i - 1)), --i);
				#endif
				fclose(fp);
				return TRUE;
			} else printf("E: failed to create global commands.\n");
			fclose(fp);
		} else printf("E: can't open \"%s\" on %s.\n", path, __FUNCTION__);
	};
	err_exit;
};

static bool get_command(COMMAND *c){
	if (c != NULL){
		c->string = (*c).get_string(stdin);
		if (c->string != NULL){
			(*c).str_tokenizer(c);
			return TRUE;
		};
	};
	return FALSE;
};

static bool memory_dump(COMMAND *c){
	//Just a debug function
	if (c != NULL){
		byte i = 0;
		while(!stack_empty(c->memory)){
			char *aux = stack_pop(c->memory);
			printf("%d. \"%s\"\n", i++, aux);
			free(aux);
		};
		return TRUE;
	};
	err_exit;
};

static void token_treatment(char **s){
	//Makes command case insensitive
	if (s != NULL && *s != NULL){
		for(byte i = 0; i < GLOBALV_COMMAND_MAXLEN && *(*s + i) != '\0'; ++i)
			if (*(*s + i) >= ASCIIA && *(*s + i) <= ASCIIZ)
				*(*s + i) += MODULUS(ASCIIa - ASCIIA);
	};
};

static void string_tokenizer(COMMAND *c){
	if (c != NULL && c->memory != NULL && c->string != NULL){
		char *token = NULL;
		bool valid = FALSE;

		if (!stack_empty(c->memory))
			stack_clear(c->memory);

		for(int j = 0, i = 0; i < GLOBALV_COMMAND_MAXLEN && *(c->string + i) != '\0'; ++i)
			if (*(c->string + i) != SPACEBAR){
				if (token == NULL){
					//I judge doing a single malloc which covers all the 
					//possibilities much better than a bunch of reallocs 
					//for every single new letter on the new token.
					token = malloc(sizeof(char) * GLOBALV_COMMAND_MAXLEN);
					*(token + GLOBALV_COMMAND_MAXLEN - 1) = '\0';
				}
				*(token + j) = *(c->string + i);
				//This flags shows if there's, in fact, something to 
				//stack up in command's memory, and not a bunch of spacebars.
				valid = TRUE;
				++j;
			} else {
				if (valid){
					token = realloc(token, sizeof(char) * (j + 1));
					*(token + j) = '\0';
					(*c).tkn_treatment(&token);
					stack_push(c->memory, token);
					token = NULL;
					j = 0;
					valid = FALSE;
				};
			};

		if (token != NULL)
			free(token);
		free(c->string);
		c->string = NULL;
	};
};

char *get_string(FILE *fp){
	if (fp != NULL){
		char *s = malloc(sizeof(char) * (GLOBALV_COMMAND_MAXLEN + 1)), c = 0;
		int i = 0;
		
		while(i < GLOBALV_COMMAND_MAXLEN && c != ENTER && c != EOF && c != '\r'){
			c = fgetc(fp);
			*(s + i++) = c;
		};

		if (i > 1){
			s = realloc(s, sizeof(char) * (i + 1));
			if (fp == stdin){
				*(s + i - 1) = SPACEBAR;
				*(s + i) = '\0';
			} else *(s +i - 1) = '\0';
			return s;
		} else {
			free(s);
			#ifdef DEBUG
				printf("D: \"%s\" failed.\n", __FUNCTION__);
			#endif
		};
	};
	return NULL;
};

COMMAND *cinit(){
	COMMAND *c = malloc(sizeof(COMMAND));
	if (c != NULL){
		c->memory = stack_init();
		if (c->memory != NULL){
			c->get_command = &get_command;
			c->str_tokenizer = &string_tokenizer;
			c->tkn_treatment = &token_treatment;
			c->mem_dump = &memory_dump;
			c->get_string = &get_string;
			c->cprocess = &cprocess;
			c->loadglobal = &load_globalCommands;
			c->loadfails = &load_failstrings;
			c->fail_strings = NULL;
			c->gcommands = NULL;
			c->string = NULL;
			c->failnum = 0;
			c->gcnum = 0;
			return c;
		};
		//Something went wrong, abort.
		free(c);
	};
	err_exit;
};

bool cdestroy(COMMAND **c){
	if (c != NULL && *c != NULL){
		if ((*c)->memory != NULL)
			stack_destroy(&(*c)->memory);
		if ((*c)->string != NULL)
			free((*c)->string);
		if ((*c)->gcommands != NULL){
			for(byte i = (*c)->gcnum; i > 0; --i)
				if (*((*c)->gcommands + i - 1) != NULL)
					free (*((*c)->gcommands + i - 1));
			free((*c)->gcommands);
		};
		if ((*c)->fail_strings != NULL){
			for(byte i = (*c)->failnum; i > 0; --i)
				if (*((*c)->fail_strings + i - 1) != NULL)
					free (*((*c)->fail_strings + i - 1));
			free((*c)->fail_strings);
		};
		free(*c);
		(*c) = NULL;
		return TRUE;
	};
	err_exit;
};

void rprintf(char **s, byte num){
	if (s != NULL){
		srand(time(NULL));
		printf("%s\n", *(s + (rand() % num)));
	};
};