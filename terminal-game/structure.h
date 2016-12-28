#ifndef __TERMINAL_STRUCTURE_H_
#define __TERMINAL_STRUCTURE_H_
#include "commands.h"

#define GLOBALV_MAPW 6
#define GLOBALV_MAPH 4

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
	bool (*chsetup)(WORLD *);

	CHAMBER **allchambers;
	byte nused;
};

struct interactives {
	bool (*iload)(IACTV *); 

	char *label, **actions;
	byte progress;
	byte actnum;
};

struct map_chamber {
	//Methods
	bool (*adjch_setup)(CHAMBER *, byte, ...);
	bool (*iactv_setup)(CHAMBER *, byte, ...);

	//Resources
	IACTV **iactives;
	CHAMBER **adjchambers;
};

GAME *ginit();
WORLD *winit();
CHAMBER *chinit();
bool wdestroy(WORLD **);
bool gdestroy(GAME **);

#endif
