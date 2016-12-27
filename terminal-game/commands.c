#include "core.h"
#include "resources.h"
#include "commands.h"
#include <stdlib.h>
#include <stdio.h>

static void get_command(COMMAND *c){
	if (c != NULL){
		c->string = (*c).get_string();
		(*c).str_tokenizer(c);
	};
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
	return FALSE;
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
				} else --j;
			}

		if (token != NULL)
			free(token);
		free(c->string);
		c->string = NULL;
	};
};

static char *get_string(){
	char *s = malloc(sizeof(char) * (GLOBALV_COMMAND_MAXLEN + 1)), c = 0;
	int i = 0;
	while(i < GLOBALV_COMMAND_MAXLEN && c != ENTER && c != EOF && c != '\r'){
		c = fgetc(stdin);
		*(s + i++) = c;
	}
	if (i > 1){
		*(s + i - 1) = SPACEBAR;
		*(s + i) = '\0';
		return s;
	} else {
		free(s);
		return NULL;
	};
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
			c->string = NULL;
			return c;
		};
		//Something went wrong, abort.
		free(c);
	};
	return NULL;
};

bool cdestroy(COMMAND **c){
	if (c != NULL && *c != NULL){
		if ((*c)->memory != NULL)
			stack_destroy(&(*c)->memory);
		if ((*c)->string != NULL)
			free((*c)->string);
		free(*c);
		(*c) = NULL;
		return TRUE;
	};
	return FALSE;
};

