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
	bool (*pdestroy)(PLAYER **);

	//P. characteristics
	bool enable;
	uint *colectibles;
	byte pos;
};

PLAYER *pinit();
bool pdestroy(PLAYER **);

#endif