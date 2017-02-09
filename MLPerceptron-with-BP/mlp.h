#ifndef __TRUTH_T_NEURAL_
#define __TRUTH_T_NEURAL_

typedef unsigned char bool;
typedef unsigned char byte;
typedef unsigned int ui;

#define ABS(X) ((X) < 0 ? (-(X)) : (X))
#define sqr(X) ((X)*(X))
#define DELTA 0.1
#define RATE 0.1
#define VAR_NUM 2
#define TABLE_HEIGHT 4
#define NODE_NUM 4

#define STD_MIN -5
#define STD_MAX 5

typedef struct {
	bool *input, hidden, output;
	double *act;
	double *bias;
	double **weigth;
} network;

enum {
	INPUT_A,
	INPUT_B,
	HIDDEN,
	OUTPUT
};

enum {
	OUT_CNT,	
	HDN_CNT
};

enum {
	PROG_NAME,
	SEED,
	TRAIN_FILE,
	INPUT_FILE,
	MAX_ITERATION,
	ARG_NUM
};

#endif