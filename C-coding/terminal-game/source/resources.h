#ifndef __TERMINAL_RESOURCE_H_
#define __TERMINAL_RESOURCE_H_

#define MODULUS(A) (A < 0) ? -A : A
#define BITS_IN_BYTE 8
#define GLOBALV_IACTVNUM 30
#define SWAP(A,B) {short int C = A; A = B; B = C;}
#define SWAP_INT(A,B) {int C = A; A = B; B = C;}
#define SWAP_VEC(A,B) {void *C = A; A = B; B = C;}
#define FATHER(A) (A/2)

typedef enum {
	FALSE,
	TRUE
} bool;

typedef unsigned int uint;
typedef unsigned char byte; 

//GENERAL STUFF
void quicksort(byte *, byte);
short int binsearch(byte *, byte, byte);

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
byte list_count(LIST *);

//DYNAMIC HEAP STUFF
typedef struct heapd HEAPD;

HEAPD *heapd_init();
char *heapd_remove(HEAPD *, unsigned int);
bool heapd_insert(HEAPD *, char *, unsigned int);
bool heapd_destroy(HEAPD **);
bool heapd_clear(HEAPD *);
bool heapd_print(HEAPD *);
bool heapd_printlist(HEAPD *);

#endif
