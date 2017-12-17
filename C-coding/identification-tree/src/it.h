#ifndef __ID_TREE_
#define __ID_TREE_

typedef struct itm ITM;
typedef unsigned char byte;
typedef unsigned short int lbyte;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define SWAP(X,Y) {byte Z = (X); (X) = (Y); (Y) = Z;}
#define DELTA 0.05
#define MARKED 255
#define HEADSIZE 10
#define ADJUST 0
#define STD_OFFSET 3

enum {
	PROGNAME,
	TRAINPATH,
	INPUTPATH,
	NUMARGS
};

ITM *itModel(byte **, size_t, size_t);
byte **dataGet(FILE *, size_t *, size_t *);
void dataPurge(byte **, size_t);
void itPredict(ITM *, FILE *);
void itPurge(ITM **);
void itPrint(ITM *);

#endif