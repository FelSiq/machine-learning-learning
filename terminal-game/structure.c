#include "core.h"
#include "resources.h"
#include "structure.h"
#include "commands.h"
#include "player.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* World map sketch 
	10111000 
	11101100 
	00111000 
	11101000
*/

static bool chamber_setup(WORLD *w){
	if (NULL != w && NULL != w->allchambers){
		CHAMBER *aux;
		//Here's some hard codded number, which express exacly the world map sketch above, when in binary notation.
		uint ms = 3102488808, mask = (1 << (sizeof(uint) * BITS_IN_BYTE - 1));

		#ifdef DEBUG
			printf("D: will create chamber list in this model:\n");
			for(uint Q = (1 << (sizeof(uint)*BITS_IN_BYTE - 1)), counter = 0; Q > 0; Q >>= 1, ++counter){
				printf("%d", (ms & Q) >= 1);
				if ((counter + 1) % GLOBALV_MAPW == 0){
					printf("\n");
					Q >>= 2;
				};
			};
		#endif
		//Work here.
		for (byte i = 0; i < (GLOBALV_MAPH * GLOBALV_MAPW); ++i, mask >>= 1){
			if (ms & mask){
				aux = chinit();
				if (aux != NULL){
					++(w->nused);
					*(w->allchambers + i) = aux;
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

		#ifdef DEBUG
			printf("D: will start path construction on [%p] chamber...\n", ch);
		#endif

		CHAMBER *aux;
		PATH *p;
		ch->adjchambers = realloc(ch->adjchambers, sizeof(PATH *) * (ch->adjnum + num));
		if (NULL != ch->adjchambers){
			//ch->adjnum += num;
			va_list adjchambers;
			va_start(adjchambers, num);
			for (byte i = num; i > 0; --i){
				aux = va_arg(adjchambers, CHAMBER *);
				if (aux != NULL){
					p = malloc(sizeof(PATH));
					if (p != NULL){
						//Set the new path
						*(ch->adjchambers + ch->adjnum + i - 1) = p;
						p->a = ch;
						p->b = aux;
						p->open = TRUE;
						p->string = NULL;
						//Set the new path on the b chamber
						p->b->adjchambers = realloc(p->b->adjchambers, sizeof(PATH *) * (p->b->adjnum + 1));
						if (p->b->adjchambers != NULL){
							*(p->b->adjchambers + p->b->adjnum++) = p;

							#ifdef DEBUG
								printf("D: successfully set path between {[%p]<->[%p]} chambers...\n", p->a, p->b);
							#endif
						};
					} else {
						//Something went wrong, backtrack everything and abort.
						//ch->adjnum -= num;
						for (byte j = (i - 1); j < num; ++j)
							free(*(ch->adjchambers + ch->adjnum + j));
						if (ch->adjnum > 0)
							ch->adjchambers = realloc(ch->adjchambers, sizeof(PATH *) * ch->adjnum);
						else {
							free(ch->adjchambers);
							ch->adjchambers = NULL;
						}
						va_end(adjchambers);
						printf("E: \"%s\" failed.\n", __FUNCTION__);
						return FALSE;
					};
				};
			};
			#ifdef DEBUG
				printf("D: endded path construction on [%p] chamber...\n", ch);
			#endif
			ch->adjnum += num;
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
		ch->adjnum = 0;
	};
	return ch;
};

bool idestroy(IACTV **i){
	if (i != NULL && *i != NULL){
		free(*i);
		(*i) = NULL;
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

bool chdestroy(CHAMBER **ch){
	if (ch != NULL && *ch != NULL){
		if ((*ch)->adjchambers != NULL){
			CHAMBER *caux;
			PATH *aux;
			for (byte j = (*ch)->adjnum; j > 0; --j){
				aux = *((*ch)->adjchambers + j - 1);
				if (aux != NULL){
					caux = (aux->b == (*ch)) ? aux->a : aux->b;
					for(byte i = caux->adjnum; i > 0; --i){
						if (*(caux->adjchambers + i - 1) != aux)
							continue;
						*(caux->adjchambers + i - 1) = NULL;
					};
					free(aux);
				};
			};
			free((*ch)->adjchambers);
		};
		if ((*ch)->iactives != NULL){
			while(0 < (*ch)->actnum--)
				free(*((*ch)->iactives + (*ch)->actnum));
			free((*ch)->iactives);
		};
		free(*ch);
		(*ch) = NULL;
		return TRUE;
	};
	printf("E: \"%s\" failed.\n", __FUNCTION__);
	return FALSE;
};

bool wdestroy(WORLD **w){
	if (w != NULL && *w != NULL){
		if((*w)->allchambers != NULL){
			for(byte i = (GLOBALV_MAPW * GLOBALV_MAPH); i > 0; --i)
				if (NULL != *((*w)->allchambers + i - 1))
					chdestroy(&*((*w)->allchambers + i - 1));
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