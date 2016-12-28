#include "core.h"
#include "resources.h"
#include "structure.h"
#include "commands.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define DEBUG

typedef union {
	uint pack;
	byte line[sizeof(uint)];
} mapsketch;

/* World map sketch 
	10111000 
	11101100 
	00111000 
	11101000
*/

static bool chamber_setup(WORLD *w){
	if (NULL != w && NULL != w->allchambers){
		CHAMBER *aux;
		mapsketch ms;
		uint mask = (1 << (sizeof(uint) * BITS_IN_BYTE - 1));
		//Here's some hard codded number, which express exacly the world map sketch above, when in binary notation.
		ms.pack = 3102488808;

		#ifdef DEBUG
			printf("D: will create chamber list in this model:\n");
			for(uint Q = (1 << (sizeof(uint)*BITS_IN_BYTE - 1)), counter = 0; Q > 0; Q >>= 1, ++counter){
				printf("%d", (ms.pack & Q) >= 1);
				if ((counter + 1) % GLOBALV_MAPW == 0){
					printf("\n");
					Q >>= 2;
				};
			};
		#endif
		//Work here.
		for (byte i = 0; i < (GLOBALV_MAPH * GLOBALV_MAPW); ++i, mask >>= 1){
			if (ms.pack & mask){
				aux = chinit();
				if (aux != NULL){
					*(w->allchambers + i - 1) = aux;
				} else {
					//Something went wrong, abort.
					for(byte k = i; k > 0; --k)
						if (*(w->allchambers + k) != NULL)
							free(*(w->allchambers + k));
					printf("E: \"%s\" failed.\n", __FUNCTION__);
					return FALSE;
				};
			};
			if ((i + 1) % GLOBALV_MAPW == 0)
				mask >>= 2;
		};

		#ifdef DEBUG
			printf("D: successfully created chamber list:\n");
			for(byte Q = 0; Q < (GLOBALV_MAPH * GLOBALV_MAPW); ++Q){
				printf("%d", *(w->allchambers + Q) != NULL);
				if (((Q + 1) % GLOBALV_MAPW) == 0)
					printf("\n");
			};
		#endif
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

static bool game_setup(GAME *const g){
	if (g != NULL){
		#ifdef DEBUG
			printf("D: trying to start WORLD structure...\n");
		#endif
		g->world = winit();

		if (g->world != NULL){
			#ifdef DEBUG
				printf("D: success, now will try to start PLAYER structure...\n");
			#endif
			g->player = pinit();

			if (g->player != NULL){
				#ifdef DEBUG
					printf("D: success, now will try to start COMMAND structure...\n");
				#endif
				g->command = cinit();

				if (g->command != NULL){
					#ifdef DEBUG
						printf("D: success on main structure initialization...\n");
					#endif
					return TRUE;
				} else {
					#ifdef DEBUG
						printf("D: failed to start COMMAND structure, aborting...\n");
					#endif
				};
				pdestroy(&g->player);
			} else {
				#ifdef DEBUG
					printf("D: failed to start PLAYER structure, aborting...\n");
				#endif
			};
			wdestroy(&g->world);
		} else {
			#ifdef DEBUG
				printf("D:failed to start WORLD structure, aborting...\n");
			#endif
		};
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

static bool ch_adjch_setup(CHAMBER *ch, byte num, ...){
	if (NULL != ch){
		ch->adjchambers = malloc(sizeof(CHAMBER *) * num);
		if (NULL != ch->adjchambers){
			va_list adjchambers;
			va_start(adjchambers, num);
			for (byte i = num; i > 0; *(ch->adjchambers + i - 1) = va_arg(adjchambers, CHAMBER *), --i);
			va_end(adjchambers);
			return TRUE;
		};
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};


static bool ch_iatcv_setup(CHAMBER *ch, byte num, ...){
	if (NULL != ch){
		ch->iactives = malloc(sizeof(IACTV *) * num);
		if (NULL != ch->iactives){
			va_list iactv;
			va_start(iactv, num);
			for (byte i = num; i > 0; *(ch->iactives + i - 1) = va_arg(iactv, IACTV *), --i);
			va_end(iactv);
			return TRUE;
		}
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

GAME *ginit(){
	GAME *g = malloc(sizeof(GAME));
	if (g != NULL){
		g->gsetup = &game_setup;

		#ifdef DEBUG
			printf("D: will start game main structure...\n");
		#endif

		if ((*g).gsetup(g))
			return g;
		free(g);
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return NULL;
};

WORLD *winit(){
	WORLD *w = malloc(sizeof(WORLD));
	if (NULL != w){
		w->allchambers = malloc(sizeof(CHAMBER *) * (GLOBALV_MAPW * GLOBALV_MAPH));
		if (NULL != w->allchambers){
			for (byte i = (GLOBALV_MAPW * GLOBALV_MAPH); i > 0; *(w->allchambers + i - 1) = NULL, --i);
			w->chsetup = &chamber_setup;
			return w;
		};
		free(w);
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return NULL;
};

CHAMBER *chinit(){
	CHAMBER *ch = malloc(sizeof(CHAMBER));
	if (NULL != ch){
		ch->adjch_setup = &ch_adjch_setup;
		ch->iactv_setup = &ch_iatcv_setup;
		ch->adjchambers = NULL;
		ch->iactives = NULL;
	};
	return ch;
};

bool wdestroy(WORLD **w){
	if (w != NULL && *w != NULL){
		if((*w)->allchambers != NULL){
			for(byte i = (GLOBALV_MAPW * GLOBALV_MAPH); i > 0; --i)
				if (NULL != *((*w)->allchambers + i - 1))
					free(*((*w)->allchambers + i - 1));
			free((*w)->allchambers);
		};
		free(*w);
		(*w) = NULL;
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

bool gdestroy(GAME **g){
	if (g != NULL && *g != NULL){
		if (NULL != (*g)->world)
			wdestroy(&(*g)->world);
		if (NULL != (*g)->player)
			pdestroy(&(*g)->player);
		if (NULL != (*g)->command)
			cdestroy(&(*g)->command);
		free(*g);
		(*g) = NULL;
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};