#ifndef __ID_TREE_
#define __ID_TREE_

typedef struct itm ITM;
typedef unsigned char byte;
typedef unsigned short int lbyte;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define DELTA 0.01
#define MARKED 255
#define HEADSIZE 10

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