#include "core.h"
#include "resources.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>
#include "controls.h"

bool pdestroy(PLAYER **p){
	if (p != NULL && *p != NULL){
		if ((*p)->colectibles != NULL)
			free((*p)->colectibles);
		free(*p);
		(*p) = NULL;
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

static bool func_playerSetup(PLAYER *p){
	if (p != NULL){
		//Setting up main player stuff
		p->enable = TRUE;
		p->pos = GLOBALV_PLAYER_STDSTART;
		p->colectibles = malloc(sizeof(uint) * GLOBALV_PINV_STDSIZE);
		//Cleaning player's invectory up
		if (p->colectibles != NULL){
			for (byte i = GLOBALV_PINV_STDSIZE; i > 0; *(p->colectibles + i - 1) = 0, --i);
			return TRUE;
		};
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

PLAYER *pinit(){
	PLAYER *p = malloc(sizeof(PLAYER));
	if (p != NULL){
		//Setting player's methods
		(*p).psetup = &func_playerSetup;
		//Setting player's invectory up
		if ((*p).psetup(p))
			return p;
		//At this point something went wrong
		free(p);
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return NULL;
};