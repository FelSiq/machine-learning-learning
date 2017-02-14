#ifndef __ID_TREE_
#define __ID_TREE_

typedef struct itm ITM;
typedef unsigned char byte;
typedef unsigned short int lbyte;

enum {
	PROGNAME,
	TRAINPATH,
	INPUTPATH,
	NUMARGS
};

ITM *itModel(byte **, lbyte, lbyte);
byte **dataGet(FILE *, lbyte *, lbyte *);
void dataPurge(byte **, lbyte);
void itPredict(ITM *, FILE *);
void itPurge(ITM **);

#endif