#ifndef __ID_TREE_
#define __ID_TREE_

typedef struct itm ITM;
typedef unsigned char byte;
typedef unsigned short int lbyte;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define SWAP(X,Y) {byte Z = (X); (X) = (Y); (Y) = Z;}
#define DELTA 0.1
#define MARKED 255
#define HEADSIZE 10
#define ADJUST 0 //48
#define STD_OFFSET 3
#define TRUE 1
#define FALSE 0

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
void itPrint(ITM *);

#endif