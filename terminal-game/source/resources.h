#ifndef __TERMINAL_RESOURCE_H_
#define __TERMINAL_RESOURCE_H_

#define MODULUS(A) (A < 0) ? -A : A
#define BITS_IN_BYTE 8
#define GLOBALV_IACTVNUM 30

typedef enum {
	FALSE,
	TRUE
} bool;

typedef unsigned int uint;
typedef unsigned char byte; 

//STACK STUFF
typedef struct stack STACK;

STACK *stack_init();
bool stack_push(STACK *, char *);
bool stack_empty(STACK *);
char *stack_pop(STACK *);
char *stack_top(STACK *);
bool stack_destroy(STACK **);
bool stack_clear(STACK *);

//LIST STUFF
typedef struct list LIST;

LIST *list_init();
bool list_append(LIST *, byte, char *);
bool list_modify(LIST *, byte, char *);
bool list_remove(LIST *, byte);
bool list_clear(LIST *);
bool list_empty(LIST *);
bool list_destroy(LIST **);
void list_print(LIST *);

#endif
