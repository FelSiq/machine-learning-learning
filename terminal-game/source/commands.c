#include "core.h"
#include "resources.h"
#include "structure.h"
#include "commands.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

static bool cprocess(GAME *g, CHAMBER *tvl, STACK *s){
	if (g != NULL && g->command != NULL && s != NULL && tvl != NULL){
		bool FLAG = TRUE;
		while(!stack_empty(s)){
			char *string = stack_pop(s);
			if (string != NULL){
				#ifdef DEBUG
					printf("D: will process \"%s\"...\n", string);
				#endif
				//Test global commands first
				for(byte i = g->command->gcnum; FLAG && (i > 0); --i){
					if (strcmp(string, *(g->command->gcommands + i -1)) == 0){
						if (strcmp(string, "sair") == 0){
							g->END_FLAG = TRUE;
							FLAG = FALSE;
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

							FLAG = FALSE;
						} else if (strcmp(string, "notas") == 0){
							if (!list_empty(g->player->notes)){
								list_print(g->player->notes);
							} else printf("Você não possui nenhuma tarefa ativa no momento.\n");
							FLAG = FALSE;
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
							FLAG = FALSE;
						} else if (strcmp(string, "salvar") == 0){
							if (g->grefresh(g))
								printf("Progresso salvo com sucesso.\n");
							else
								printf("Algo deu errado, seu progresso não pôde ser salvo.\n");
							FLAG = FALSE;
						};
					} else {
						//Now test with local commands
					};
				};		
				free(string);
			};
		};
		if (FLAG)
			rprintf(g->command->fail_strings, g->command->failnum);

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