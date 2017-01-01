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
	err_exit;
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
	err_exit;
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
	err_exit;
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
	err_exit;
};

//LIST STUFF

typedef struct list_node {
	byte key;
	struct list_node *next, *prev;
	char *string;
} LNODE;

struct list {
	LNODE *hnode;
};

LIST *list_init(){
	LIST *l = malloc(sizeof(LIST));
	if (l != NULL){
		l->hnode = malloc(sizeof(LNODE));
		if (l->hnode != NULL){
			l->hnode->next = l->hnode;
			l->hnode->prev = l->hnode;
			l->hnode->string = NULL;
			l->hnode->key = 0;
			return l;
		};
		free(l);
	};
	return NULL;
};

bool list_empty(LIST *l){
	if (l != NULL && l->hnode != NULL)
		return (l->hnode->next == l->hnode);
	return TRUE;
};

bool list_append(LIST *l, byte key, char *string){
	if (l != NULL && string != NULL){
		LNODE *ln = malloc(sizeof(LNODE));
		if (ln != NULL){
			ln->prev = l->hnode->prev;
			ln->next = l->hnode;
			l->hnode->prev->next = ln;
			l->hnode->prev = ln;
			ln->string = string;
			ln->key = key;
			return TRUE;
		};
	};
	err_exit;
};

bool list_modify(LIST *l, byte key, char *string){
	if (l != NULL && string != NULL){
		LNODE *ln = l->hnode->next;
		while(ln != l->hnode && ln->key != key)
			ln = ln->next;
		if (ln != l->hnode){
			if (ln->string != NULL)
				free(ln->string);
			ln->string = string;
			return TRUE;
		};
		return FALSE;
	};
	err_exit;
};

bool list_remove(LIST *l, byte key){
	if (l != NULL){
		LNODE *ln = l->hnode->next;
		while(ln != l->hnode && ln->key != key)
			ln = ln->next;
		if (ln != l->hnode){
			if (ln->string != NULL)
				free(ln->string);
			ln->prev->next = ln->next;
			ln->next->prev = ln->prev;
			free(ln);
			return TRUE;
		};
		return FALSE;
	};
	err_exit;
};

bool list_clear(LIST *l){
	if (l != NULL){
		if (l->hnode != NULL){
			LNODE *aux = (l->hnode->next), *rem;
			while(aux != l->hnode){
				rem = aux;
				aux = aux->next;
				free(rem);
			};
		};
		l->hnode->next = l->hnode;
		l->hnode->prev = l->hnode;
		return TRUE;
	};
	err_exit;
};

bool list_destroy(LIST **l){
	if (l != NULL && *l != NULL){
		if ((*l)->hnode != NULL){
			LNODE *aux = ((*l)->hnode->next), *rem;
			while(aux != (*l)->hnode){
				rem = aux;
				aux = aux->next;
				free(rem);
			};
			free((*l)->hnode);
		};
		free(*l);
		(*l) = NULL;
		return TRUE;
	};
	err_exit;
};

void list_print(LIST *l){
	if (l != NULL && !list_empty(l)){
		LNODE *aux = l->hnode->next;
		byte i = 0;
		while(aux != l->hnode){
			printf("%hu. \"%s\"\n", ++i, aux->string);
			aux = aux->next;
		};
	};
};