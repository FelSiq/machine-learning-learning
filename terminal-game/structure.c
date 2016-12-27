#include "core.h"
#include "resources.h"
#include "structure.h"
#include "commands.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>

struct the_game {
	WORLD *world;
	PLAYER *player;
	CONTROL *control;
};

struct the_world {
	CHAMBER **warppoints;
};

struct map_chamber {
	//Methods
	bool (*chamber_setup)(CHAMBER *, CHAMBER *, CHAMBER *, CHAMBER *);

	//Resources
	INTERACTIVES **interactives;
	CHAMBER **chambers;
};