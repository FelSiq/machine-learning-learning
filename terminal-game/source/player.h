#ifndef __TERMINAL_PLAYER_H_
#define __TERMINAL_PLAYER_H_

#include "core.h"
#include "resources.h"
#define GLOBALV_PINV_STDSIZE 10
#define GLOBALV_PLAYER_STDSTART 8

typedef struct the_player PLAYER;

struct the_player {
	//P. Methods
	bool (*psetup)(PLAYER *);
	bool (*pgetname)(PLAYER *, FILE *);
	bool (*pgetitem)(PLAYER *, byte);

	//P. characteristics
	bool enable;
	byte *colectibles, pos;
	char *name;
};

PLAYER *pinit();
bool pdestroy(PLAYER **);

#endif