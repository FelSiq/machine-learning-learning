#ifndef __TERMINAL_STRUCTURE_H_
#define __TERMINAL_STRUCTURE_H_
#include "commands.h"

#define GLOBAV_NUMPATHS 8
#define GLOBALV_MAPW 6
#define GLOBALV_MAPH 4

typedef struct path PATH;
typedef struct the_game GAME;
typedef struct the_world WORLD;
typedef struct the_player PLAYER;
typedef struct interactives IACTV;
typedef struct map_chamber CHAMBER;

struct the_game {
	bool (*gsetup)(GAME *);

	WORLD *world;
	PLAYER *player;
	COMMAND *command;
};

struct the_world {
	bool (*wload)(WORLD *);
	bool (*wgetlabels)(WORLD *);
	bool (*chsetup)(WORLD *);

	CHAMBER **allchambers;
	byte nused;
};

struct interactives {
	bool (*iload)(IACTV *, char const *); 

	char **script, *label, **actions;
	byte progress;
	byte scpnum, actnum;
};

struct map_chamber {
	//Methods
	bool (*adjch_setup)(CHAMBER *, byte, ...);
	bool (*iactv_setup)(CHAMBER *, byte, ...);

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
