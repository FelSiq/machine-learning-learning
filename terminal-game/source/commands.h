#ifndef __TERMINAL_COMMANDS_H_
#define __TERMINAL_COMMANDS_H_
#define GLOBALV_COMMAND_MAXLEN 200
#define GLOBALV_COMMAND_MAXNUM 50
#define ASCIIA 65
#define ASCIIZ 90
#define ASCIIa 97
#define ASCIIz 122
#define SPACEBAR 32
#define ENTER 10

#define GLOBALV_BACKPACK_LINES 3

#include "resources.h"
#include "structure.h"
typedef struct commands COMMAND;

struct commands {
	char *string, **gcommands, **fail_strings;
	STACK *memory;
	byte gcnum, failnum;
	
	bool (*loadglobal)(COMMAND *, char const *);
	bool (*loadfails)(COMMAND *, char const *);
	void (*str_tokenizer)(COMMAND *);
	bool (*get_command)(COMMAND *);
	bool (*mem_dump)(COMMAND *);
	
	bool (*cprocess)(GAME *, CHAMBER *, STACK *);
	void (*tkn_treatment)(char **);
	char *(*get_string)();
};

COMMAND *cinit();
bool cdestroy(COMMAND **);
char *get_string(FILE *);
void rprintf(char **, byte);
void string_uppercase(char *);
void string_lowercase(char *);

#endif