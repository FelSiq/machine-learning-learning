#ifndef __BIN_T_
#define __BIN_T_

typedef struct {
	//O "item" pode ser o que você quiser.
	//De fato, esta struct pode nem precisar existir.
	float value;
} ITEM;

typedef struct binT binT;

binT *btInit();
void btInsert(binT *, unsigned int , ITEM *);
void btPrint(binT *);
ITEM *btSearch(binT *, unsigned int);
ITEM *btRemove(binT *, unsigned int);
void btPurge(binT *);

#endif