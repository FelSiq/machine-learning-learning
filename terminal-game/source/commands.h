#ifndef __TERMINAL_COMMANDS_H_
#define __TERMINAL_COMMANDS_H_
#define GLOBALV_COMMAND_MAXLEN 200
#define ASCIIA 65
#define ASCIIZ 90
#define ASCIIa 97
#define SPACEBAR 32
#define ENTER 10

#include "resources.h"

typedef struct commands COMMAND;

struct commands {
	void (*get_command)(COMMAND *);
	void (*str_tokenizer)(COMMAND *);
	void (*tkn_treatment)(char **);
	char *(*get_string)();
	bool (*mem_dump)(COMMAND *);
	bool (*cprocess)(STACK *);

	STACK *memory;
	char *string;
};

COMMAND *cinit();
bool cdestroy(COMMAND **);
char *get_string(FILE *);

#endif