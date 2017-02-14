#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "it.h"

typedef struct node {
	byte *conditions, *args;
	//Conditions stands for values to be check on this node, to
	//select the son that the input must follow in order to be identified.
	//Args stand for the values to be check.
	//This values must cover all the possibilities.
	char output;
	//This value is used only on leaf nodes. It represents the
	//input identification.
	//default value from nodeInit func is -1.
	struct node **sons;
	lbyte numSons;
} NODE;

struct itm{
	NODE *root;
	lbyte numNodes, argNum;
};

byte **dataGet(FILE *ftrain, lbyte *colNum, lbyte *examplesNum){
	byte **data = NULL;
	if (ftrain != NULL){
		//Get the argNum
		for(char c = 0; c != EOF && c != '\n'; c = fgetc(ftrain))
			*colNum += (c == ' ');
		if (ftell(ftrain) > 0)
			++(*colNum);
		fseek(ftrain, 0, SEEK_SET);
		if (*colNum > 0){
			while(!feof(ftrain)){
				data = realloc(data, sizeof(byte *) * (1 + *examplesNum));
				*(data + *examplesNum) = malloc(sizeof(byte) * *colNum);
				for(lbyte i = 0; i < *colNum; ++i)
					fscanf(ftrain, "%c%*c", (*(data + *examplesNum) + i));
				++(*examplesNum);
			}
		}
	}
	return data;
};

void dataPurge(byte **data, lbyte examplesNum){
	if (data != NULL){
		while(0 < examplesNum--)
			free(*(data + examplesNum));
		free(data);
	}
};

static NODE *nodeInit(){
	NODE *node = malloc(sizeof(node));
	if (node != NULL){
		node->conditions = NULL;
		node->args = NULL;
		node->sons = NULL;
		node->output = -1;
		node->numSons = 0;
	};
	return node;
};

static ITM *itInit(){
	ITM *model = malloc(sizeof(ITM));
	if (model != NULL){
		model->root = NULL;
		model->numNodes = 0;
	}
	return model;
};


//Work out these two functions belown

/*

	Data disorder (dd): -(Positives/Total)*log2(Positives/Total) -(Negatives/Total)*log2(Negatives/Total)
	To select a good subtree (ts): sum (dd(node) * (Elements inside node/Total of elements on the subtree))

	Has data with n rows and k columns (tests).
	Outcomes are a discrete set of values.
	
	1. begin loop til every dead-end node on tree == leaf, var i:
		need the total score of each tree, stored on a double vector named TS_vector, with size k - i.
		2. loop two: For each st
		ill unused col:
			- Get the number of possible labels, l;
			- create two vectors with size l, clean then up,
				one is positives vector and the other, negative vector;
			- Count up the positive outputs and negative outputs for every outcome.
			- Calculate the Data disorder of every outcome and store it (very important) on a double matrix.
			- Use these information to calculate total score of this characteristic, and store on
				the correspondent index on TS_vector.
		end loop two;
		- select the characteristic with the smallest total score (total disorder) on TS_vector,
			and put it up to the son of the newest node on the identification tree.
		- Set up new node's sons based on all his possible outcomes.
		- Verify the Data disorder of every outcome of the new node.
			if smaller than a DELTA, set its correspondent label a LEAF node.
	end loop;

*/

//Extension for numeric data:
	//take of the "every outcome" thing.
	//Get the median of all outcomes.
	//Median is the threshold. If element smaller than threshold, positive. If smaller, negative.
	//Same process then.

ITM *itModel(byte **data, lbyte colNum, lbyte examplesNum){
	ITM *model = itInit();
	if (model != NULL){

	}
	return model;
};

void itPredict(ITM *model, FILE *finput){
	if (model != NULL && finput != NULL){

	}
};

static void itPurge_rec(NODE *root){
	if (root != NULL){
		if (root->conditions != NULL)
			free(root->conditions);
		if (root->args != NULL)
			free(root->args);
		if (root->sons != NULL){
			while(0 < root->numSons--)
				itPurge_rec(*(root->sons + root->numSons));
			free(root->sons);
		}
		free(root);
	}
};

void itPurge(ITM **model){
	if (model != NULL && *model != NULL){
		itPurge_rec((*model)->root);
		free(*model);
		(*model) = NULL;
	}
};

int main(int argc, char const *argv[]){
	if (argc != NUMARGS){
		FILE *ftrain = fopen(*(argv + TRAINPATH), "r");
		if (ftrain != NULL){
			FILE *finput = fopen(*(argv + INPUTPATH), "r");
			if (finput != NULL){
				lbyte colNum = 0, examplesNum = 0;
				byte **data = dataGet(ftrain, &colNum, &examplesNum);
				if (data != NULL){
					ITM *model = itModel(data, colNum, examplesNum);
					if (model != NULL){
						itPredict(model, finput);
						itPurge(&model);
					}
					dataPurge(data, examplesNum);
					fclose(ftrain);
					fclose(finput);
					return 0;
				}
				printf("e: can't get data on \"%s\" path.\n", *(argv + TRAINPATH));
				fclose(ftrain);
				fclose(finput);
				return 4;
			}
			printf("e: cant open input file.\n");
			fclose(ftrain);
			return 3;
		}
		printf("e: can't open train file.\n");
		return 2;
	}
	printf("usage: %s <train path> <input path>\n", *(argv + PROGNAME));
	return 1;
};