#include "core.h"
#include "resources.h"
#include <stdlib.h>
#include <stdio.h>

//GENERAL STUFF
static void quicksort_rec(byte *v, short int const start, short int const end){
	short int i = start, j = end;
	byte pivot = *(v + (start + end)/2);
	while(i <= j){
		while(i <= end && *(v + i) < pivot) 
			++i;
		while(j >= start && *(v + j) > pivot) 
			--j;
		if (i <= j){
			SWAP(*(v + i), *(v + j));
			--j;
			++i;
		};
	};
	if (i < end) 
		quicksort_rec(v, i, end);
	if (j > start)
		quicksort_rec(v, start, j);
};

void quicksort(byte *v, byte size){
	if (v != NULL)
		quicksort_rec(v, 0, size - 1);
};

short int binsearch(byte *v, byte size, byte key){
	if (v != NULL){
		short int start = 0, end = (size - 1), middle;
		while(start <= end){
			middle = (start + end)/2;
			if (*(v + middle) == key)
				return middle;
			else {
				if (*(v + middle) < key)
					end = middle - 1;
				else
					start = middle + 1;
			};
		};
	};
	return -1;
};

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
			if (sn->content != NULL)
				free(sn->content);
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
			if (sn->content != NULL)
				free(sn->content);
			aux = sn;
			sn = sn->prev;
			free(aux);
		};
		s->top = NULL;
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
				if (aux->string != NULL)
					free(aux->string);
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
				if (aux->string != NULL)
					free(aux->string);
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

byte list_count(LIST *l){
	byte counter = 0;
	if (l != NULL)
		for (LNODE *aux = l->hnode->next; aux != l->hnode; aux = aux->next, ++counter);
	return counter;
};


//DYNAMIC HEAP STUFF
typedef struct tnode {
	unsigned int key; 
	struct tnode *sonl, *sonr; 
	char *string;
} TNODE;

struct heapd {
	TNODE *root;
	unsigned int nodenum;
};

static TNODE *tnode_init(char *const s, unsigned int k, 
	TNODE *sl, TNODE *sr){
	TNODE *tn = malloc(sizeof(TNODE));
	if (tn != NULL){
		tn->string = s;
		tn->key = k;
		tn->sonr = sr;
		tn->sonl = sl;
	};
	return tn;
};

static void heapd_travel_rec(TNODE *r, void (*funcp)(TNODE *)){
	if(r != NULL){
		heapd_travel_rec(r->sonl, funcp);
		heapd_travel_rec(r->sonr, funcp);
		funcp(r);
	};
};

static void heapd_print_rec(TNODE *r){
	printf("[key: %u]\t\"%s\"\n", r->key, r->string);
};

static bool heapd_insert_rec(TNODE *tn, TNODE *newnode, unsigned int bitmap){
	if (tn != NULL){
		if (bitmap & 1){
			if(heapd_insert_rec(tn->sonr, newnode, bitmap >> 1))
				tn->sonr = newnode;
			if(tn->sonr->key > tn->key){
				SWAP_VEC(tn->sonr->string, tn->string);
				SWAP_INT(tn->sonr->key, tn->key);
			};
		} else {
			if(heapd_insert_rec(tn->sonl, newnode, bitmap >> 1))
				tn->sonl = newnode;
			if(tn->sonl->key > tn->key){
				SWAP_VEC(tn->sonl->string, tn->string);
				SWAP_INT(tn->sonl->key, tn->key);
			};
		};
	} else return TRUE;
	return FALSE;
};

static unsigned int reverse_bits(unsigned int i, unsigned int *num){
	unsigned int res = 0, aux = i, maska = 1, maskb = 1;
	while(0 < (aux/= 2)){
		maska <<= 1;
		if (num != NULL)
			++(*num);
	};
	while (0 < maska){
		if (i & maska)
			res |= maskb;
		maskb <<= 1;
		maska >>= 1;
	};
	return res;
};

static void tnode_destroy(TNODE *tn){
	if (tn != NULL){
		if (tn->string != NULL)
			free(tn->string);
		free(tn);
	};
};

static void heapd_printlist_rec(TNODE *r, unsigned int l){
	if (r != NULL){
		for(unsigned int i = l; i > 0; printf(" "), --i);
		printf("%u\n", r->key);
		heapd_printlist_rec(r->sonl, 1 + l);
		heapd_printlist_rec(r->sonr, 1 + l);
	};
};

HEAPD *heapd_init(){
	HEAPD *hd = malloc(sizeof(HEAPD));
	if (hd != NULL){
		hd->root = NULL;
		hd->nodenum = 0;
	};
	return hd;
};

bool heapd_destroy(HEAPD **hd){
	if (hd != NULL && *hd != NULL){
		heapd_travel_rec((*hd)->root, &tnode_destroy);
		free(*hd);
		(*hd) = NULL;
		return TRUE;
	};
	return FALSE;
};

bool heapd_clear(HEAPD *hd){
	if(hd != NULL){
		heapd_travel_rec(hd->root, &tnode_destroy);
		hd->root = NULL;
		return TRUE;
	};
	return FALSE;
};

bool heapd_insert(HEAPD *hd, char *s, unsigned int k){
	if (hd != NULL && s != NULL){
		TNODE *newnode;
		if ((newnode = tnode_init(s, k, NULL, NULL)) != NULL){
			if (hd->root != NULL)
				heapd_insert_rec(hd->root, newnode, (reverse_bits(1 + hd->nodenum, NULL) >> 1));
			else 
				hd->root = newnode;
			++hd->nodenum;
			return TRUE;
		};
	};
	return FALSE;
};

char *heapd_remove(HEAPD *hd, unsigned int pos){
	if (hd != NULL && pos > 0 && pos <= hd->nodenum){
		char *s = NULL;
		if (hd->nodenum > 1){
			//get the last position and its father.
			TNODE *flastpos = hd->root, *lastpos = hd->root;
			unsigned int size[2] = {0, 0}, fnbitmap = ((reverse_bits(FATHER(hd->nodenum), size)) >> 1); 

			//Get last pos father (follow bitmap)
			while(0 < (*size)--){
				if (fnbitmap & 1)
					flastpos = flastpos->sonr;
				else
					flastpos = flastpos->sonl;
				fnbitmap >>= 1;
			};

			//Get last pos itself, and take it off the heap
			lastpos = flastpos->sonr;
			if (lastpos == NULL){
				lastpos = flastpos->sonl;
				flastpos->sonl = NULL;
			} else flastpos->sonr = NULL;

			//Check if we don't want to remove the last pos itself
			if (pos != hd->nodenum){
				//Now get the position (follow bitmap)
				TNODE *thepos = hd->root;
				unsigned int pbitmap = ((reverse_bits(pos, (size + 1))) >> 1);

				while(0 < (*(size + 1))--){
					if (pbitmap & 1)
						thepos = thepos->sonr;
					else
						thepos = thepos->sonl;
					pbitmap >>= 1;
				};
				//Now the trade between nodes
				SWAP_INT(lastpos->key, thepos->key);
				SWAP_VEC(lastpos->string, thepos->string);

				//Now fixdown
				TNODE *auxnode;
				do {
					auxnode = NULL;
					if(thepos->sonl != NULL && thepos->key < thepos->sonl->key)
						auxnode = thepos->sonl;
					if(thepos->sonr != NULL && thepos->key < thepos->sonr->key)
						if (auxnode == NULL || (auxnode->key < thepos->sonr->key))
							auxnode = thepos->sonr;
					if (auxnode != NULL){
						SWAP_INT(auxnode->key, thepos->key);
						SWAP_VEC(auxnode->string, thepos->string);
					};
					thepos = auxnode;
				} while (thepos != NULL);
			};
			s = lastpos->string;
			free(lastpos);
			--hd->nodenum;
		} else {
			s = hd->root->string;
			free(hd->root);
			hd->root = NULL;
			--hd->nodenum;
		};
		return s;
	};
	return NULL;
};

bool heapd_print(HEAPD *hd){
	if (hd != NULL){
		heapd_travel_rec(hd->root, &heapd_print_rec);
		return TRUE;
	};
	return FALSE;
};

bool heapd_printlist(HEAPD *hd){
	if (hd != NULL){
		heapd_printlist_rec(hd->root, 0);
		return TRUE;
	};
	return FALSE;
};