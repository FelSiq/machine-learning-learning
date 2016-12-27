#include "core.h"
#include "resources.h"
#include <stdlib.h>
#include <stdio.h>

//STACK STUFF
typedef struct stack_node {
	char *content;
	struct stack_node *prev;
} SNODE;

struct stack {
	SNODE *top;
};

STACK *stack_init(){
	STACK *s = malloc(sizeof(STACK));
	if (s != NULL)
		s->top = NULL;
	return s;
};

bool stack_push(STACK *s, char *c){
	if (s != NULL && c != NULL){
		SNODE *sn = malloc(sizeof(SNODE));
		if (sn != NULL){
			sn->content = c;
			sn->prev = s->top;
			s->top = sn;
			return TRUE;
		};
	};
	return FALSE;
};

bool stack_empty(STACK *s){
	return (s != NULL) ? ((s->top != NULL) ? FALSE : TRUE) : TRUE;
};

char *stack_pop(STACK *s){
	if (!stack_empty(s)){
		SNODE *sn = s->top;
		char *aux = s->top->content;
		s->top = s->top->prev;
		free(sn);
		return aux;
	};
	return NULL;
};

char *stack_top(STACK *s){
	return (s == NULL) ? NULL : ((s->top == NULL) ? NULL : s->top->content);
};

bool stack_destroy(STACK **s){
	if (s != NULL && *s != NULL){
		SNODE *sn = (*s)->top, *aux;
		while (sn != NULL){
			aux = sn;
			sn = sn->prev;
			free(aux);
		};
		free(*s);
		(*s) = NULL;
		return TRUE;
	};
	return FALSE;
};

bool stack_clear(STACK *s){
	if (s != NULL){
		SNODE *sn = s->top, *aux;
		while (sn != NULL){
			aux = sn;
			sn = sn->prev;
			free(aux);
		};
		return TRUE;
	};
	return FALSE;
};