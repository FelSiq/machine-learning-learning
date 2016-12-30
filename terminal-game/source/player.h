#ifndef __TERMINAL_PLAYER_H_
#define __TERMINAL_PLAYER_H_

#include "core.h"
#include "resources.h"
#define GLOBALV_PINV_STDSIZE 10
#define GLOBALV_PLAYER_STDSTART (15*4 + 7)

typedef struct the_player PLAYER;

struct the_player {
	//P. Methods
	bool (*psetup)(PLAYER *);
	bool (*pgetname)(PLAYER *);

	//P. characteristics
	bool enable;
	byte *colectibles, pos;
	char *name;
};

PLAYER *pinit();
bool pdestroy(PLAYER **);

#endif