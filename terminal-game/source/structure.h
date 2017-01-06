#ifndef __TERMINAL_STRUCTURE_H_
#define __TERMINAL_STRUCTURE_H_
#include "resources.h"

#define GLOBALV_NUMPATHS 8
#define GLOBALV_NUMTASK 20
#define GLOBALV_MAPW 6
#define GLOBALV_MAPH 4

#define MAX(A,B) ((A > B) ? A : B)

typedef struct path PATH;
typedef struct the_game GAME;
typedef struct the_world WORLD;
typedef struct the_player PLAYER;
typedef struct interactives IACTV;
typedef struct map_chamber CHAMBER;

typedef struct commands COMMAND;

struct the_game {
	void (*ginterfacePre)(GAME *, CHAMBER *, FILE **);
	void (*ginterfacePos)(GAME *, CHAMBER *, FILE **);
	bool (*gsetup)(GAME *);
	bool (*grefresh)(GAME *);

	WORLD *world;
	PLAYER *player;
	COMMAND *command;

	byte ddebug_lvl, END_FLAG;
};

struct the_world {
	bool (*chsetup)(WORLD *);
	void *(*wgetlabels)(void *);
	void *(*wload)(void *);
	void *(*isetup)(void *);

	CHAMBER **allchambers;
	byte nused;
};

struct interactives {
	IACTV *(*iload)(IACTV *, char const *); 

	char **script, *label, **actions, **extracom;
	byte progress, scpnum, actnum;
	short int *colreq, *rewards;
};

struct map_chamber {
	//Methods
	bool (*adjch_setup)(CHAMBER *, byte, ...);
	bool (*iactv_setup)(CHAMBER *, byte, ...);
	bool (*chpath_setup)(CHAMBER *, ...);

	//Resources
	IACTV **iactives;
	PATH **adjchambers;
	char *string;
	byte actnum, adjnum;
};

struct path {
	CHAMBER *a, *b;
	char *string;
	bool open;
};

GAME *ginit();
WORLD *winit();
IACTV *iinit();
CHAMBER *chinit();
bool wdestroy(WORLD **);
bool gdestroy(GAME **);
bool idestroy(IACTV **);
bool chdestroy(CHAMBER **);

#endif
