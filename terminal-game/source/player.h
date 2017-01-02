#ifndef __TERMINAL_PLAYER_H_
#define __TERMINAL_PLAYER_H_

#include "core.h"
#include "resources.h"
#define GLOBALV_PINV_STDSIZE 12
#define GLOBALV_PLAYER_STDSTART 8
#define GLOBALV_MAXCOLNUM 255

typedef struct the_player PLAYER;
typedef struct list LIST;

struct the_player {
	//P. characteristics
	byte *colectibles, tasksdone;
	char *name, **colnames;
	LIST *notes;
	bool enable;

	byte colnamnum;
	
	//P. Methods
	bool (*psetup)(PLAYER *, char const *, char const *);
	bool (*pgetname)(PLAYER *, FILE *);
	bool (*pgetitem)(PLAYER *, byte);
};

PLAYER *pinit();
bool pdestroy(PLAYER **);

#endif