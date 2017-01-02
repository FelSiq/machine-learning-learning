#include "core.h"
#include "resources.h"
#include "player.h"
#include "commands.h"
#include <stdlib.h>
#include <stdio.h>
#include "controls.h"
#include <unistd.h>
#include <string.h>

static bool func_playerGetname(PLAYER *p, FILE *fp){
	if (p != NULL && fp != NULL){
		if (p->name != NULL){
			free(p->name);
			p->name = NULL;
		};

		while((p->name = get_string(fp)) == NULL);
		*(p->name + strlen(p->name) - 1) = '\0';
		return TRUE;
	};
	err_exit;
};

static bool func_playerSetup(PLAYER *p, char const path[], char const path2[]){
	if (p != NULL){
		//Setting up main player stuff
		if (p->colectibles != NULL)
			free(p->colectibles);
		p->colectibles = malloc(sizeof(byte) * GLOBALV_PINV_STDSIZE);
		//Cleaning player's invectory up
		if (p->colectibles != NULL){
			FILE *fp = fopen(path, "r");
			byte aux = 0;
			#ifdef DEBUG
				if (access(path, R_OK) == -1)
					printf("D: file \"%s\" not found.\n", path);
				else
					printf("D: \"%s\" found and is readable.\n", path);
			#endif
			for (byte i = GLOBALV_PINV_STDSIZE; i > 0; --i)
				if (fp != NULL){
					fscanf(fp, "%hhu", &aux);
					decodify(&aux);
					*(p->colectibles + i - 1) = aux;
					aux = 0;
				} else *(p->colectibles + i - 1) = 0;

			#ifdef DEBUG
				if (fp != NULL){
					printf("D: loaded player's invenctory:\n");
					for(byte i = 0; i < GLOBALV_PINV_STDSIZE; printf("[%hu]", *(p->colectibles + i++)));
					printf("\n");
				};
				printf("D: end of player setup proccess.\n");
			#endif

			if (fp != NULL)
				fclose(fp);

			//Now colectible labels
			fp = fopen(path2, "r");
			if (fp != NULL){
				#ifdef DEBUG
					printf("D: opened \"%s\" file.", path2);
				#endif
				if (p->colnames != NULL)
					free(p->colnames); 
				p->colnames = malloc(sizeof(char *) * GLOBALV_MAXCOLNUM);
				if (p->colnames != NULL){
					char *aux;
					while(!feof(fp)){
						aux = get_string(fp);
						if (aux != NULL)
							*(p->colnames + p->colnamnum++) = aux;
					};
					p->colnames = realloc(p->colnames, sizeof(char *) * (p->colnamnum));
				} else printf("E: failed to malloc on p->colnames on %s.\n", __FUNCTION__);
				fclose(fp);
			} else printf("E: can't open \"%s\" on %s.\n", path2, __FUNCTION__);
			
			//Now player's note setup
			if (p->notes != NULL)
				list_destroy(&p->notes);
			p->notes = list_init();
			if (p->notes != NULL)
				return TRUE;
			else 
				printf("E: can't init player's notes setup (LIST) on \"%s\".\n", __FUNCTION__);
			
			//Something went wrong, abort.
			free(p->colectibles);
		};
	};
	err_exit;
};

PLAYER *pinit(){
	PLAYER *p = malloc(sizeof(PLAYER));
	if (p != NULL){
		//Setting player's methods
		(*p).psetup = &func_playerSetup;
		(*p).pgetname = &func_playerGetname;
		p->tasksdone = 0;
		p->colnamnum = 0;
		p->colectibles = NULL;
		p->enable = TRUE;
		p->name = NULL;
		p->notes = NULL;
		p->colnames = NULL;
		//Setting player's invectory up
		if ((*p).psetup(p, "./data/pinv", "./source/colnames"))
			return p;
		//At this point something went wrong
		free(p);
	};
	err_exit;
};

bool pdestroy(PLAYER **p){
	if (p != NULL && *p != NULL){
		if ((*p)->colectibles != NULL)
			free((*p)->colectibles);
		if ((*p)->name != NULL)
			free((*p)->name);
		if ((*p)->notes != NULL)
			list_destroy(&(*p)->notes);
		free(*p);
		(*p) = NULL;
		return TRUE;
	};
	err_exit;
};