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

		#ifdef DEBUG
			printf("d: number of parameters on each sample: %hu (include label)\n", *colNum);
		#endif

		if (*colNum > 0){
			while(!feof(ftrain)){
				data = realloc(data, sizeof(byte *) * (1 + *examplesNum));
				*(data + *examplesNum) = malloc(sizeof(byte) * *colNum);
				for(lbyte i = 0; i < *colNum; ++i)
					fscanf(ftrain, "%c%*c", (*(data + *examplesNum) + i));
				++(*examplesNum);
			}
		}

		#ifdef DEBUG
			printf("d: dataset height: %hu\nd: will print out head of dataset:\n", *examplesNum);
			register byte i, j;
			for(i = 0; i < MIN(HEADSIZE, *examplesNum); printf("\n"), ++i)
				for(j = 0; j < *colNum; 
					printf("%c ", *(*(data + i) + j)),
					++j);
		#endif
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

	Data disorder/entropy (dd): -(Positives/Total)*log2(Positives/Total) -(Negatives/Total)*log2(Negatives/Total)
	To select a good subtree (ts): sum (dd(node) * (Elements inside node/Total of elements on the subtree))

	Has data with n rows and k columns (tests).
	Outcomes are a discrete set of values.
	
	1. begin loop til every dead-end node on tree == leaf, var i:
		need the total score of each tree, stored on a double vector named TS_vector, with size k - i.
		2. loop two: For each still unused col:
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

ITM *itModel(byte **data, lbyte const colNum, lbyte const examplesNum){
	if (colNum > 1 && examplesNum > 0){
		ITM *model = itInit();
		if (model != NULL){
			byte FLAG = 1, k = (colNum - 1);
			////Vectors
			//vector of total Score (disorder coefficient), and for entropy on each parameter
			double *totalScore = NULL, *entropyCoef = NULL, aux0, aux1;
			//boolean vector of FLAGs to keep track of used parameters
			byte *vecUsed = malloc(sizeof(byte) * k);
			//Get all possible unique values from all parameters;
			//The first column of each row (index 0) is reserved for row size
			byte **uniqueOut = malloc(sizeof(byte *) * k);
			//POSITIVE and NEGATIVE vectors;
			//These vectors are used to entropy (disorder coefficient) calculation
			lbyte *vecPos = NULL;
			//counter Vector, to store a information necessary on entropy calculus
			lbyte *vecCounter = NULL;
			
			//Clean up vecUsed and set up uniqueOut
			register byte j, n, i;
			for (i = 0; i < k; *(vecUsed + i) = 1, ++i){
				*(uniqueOut + i) = malloc(sizeof(byte));
				**(uniqueOut + i) = 0;
				//Get unique outcomes
				for (j = 0; j < examplesNum; ++j){
					for (n = 0; 
						n < **(uniqueOut + i) && 
						(*(*(data + j) + i) != *(*(uniqueOut + i) + n + 1)); 
						++n);
							

					if (n == **(uniqueOut + i)){
						*(uniqueOut + i) = realloc(*(uniqueOut + i), 
							sizeof(byte *) * (1 + (**(uniqueOut + i))));
						*(*(uniqueOut + i) + **(uniqueOut + i) + 1) = *(*(data + j) + i);
						++(**(uniqueOut + i));
					}
				}
			}

			#ifdef DEBUG
				//DEBUG - Print unique outcomes
				printf("d: will now print UNIQUE outcomes:\n");
				for (j = 0; j < k; printf("\n"), ++j)
					for (n = 0; n < **(uniqueOut + j); 
						printf("%c ", *(*(uniqueOut + j) + n + 1)),
						++n);
			#endif

			//Loop to construct the tree itself
			do {
				totalScore = realloc(totalScore, sizeof(double) * (colNum - 1));
				#ifdef DEBUG
					printf("d: just started a new iteration...\n");
				#endif
				for(j = 0; j < (colNum - 1); ++j){
					if (*(vecUsed + j)){
						#ifdef DEBUG
							printf("d: started working on #%hhu argument.\n", j);
						#endif
						//Set up entropy vector for this parameter on this iteration
						entropyCoef = malloc(sizeof(double) * **(uniqueOut + j));
						//Set up pos/neg vectors
						vecPos = malloc(sizeof(lbyte) * **(uniqueOut + j));
						//Same with counter vector
						vecCounter = malloc(sizeof(lbyte) * **(uniqueOut + j));
						//clean both least vectors up
						for (n = 0; n < **(uniqueOut + j); ++n){
							*(vecPos + n) = 0;
							*(vecCounter + n) = 0;
						}

						//Fill up neg/pos vectors
						for(n = 0; n < examplesNum; ++n){
							if (*(*(data + n) + colNum) != MARKED){
								for (i = 0; 
									i < **(uniqueOut + j) && 
									*(*(uniqueOut + j) + i + 1) != *(*(data + n) + j);
									++i);
								*(vecPos + i) += (lbyte) MIN(1, (*(*(data + n) + colNum - 1) - 48)); 
								++(*(vecCounter + i));
							}
						}
						#ifdef DEBUG
							printf("d: positive vector on this iteration: [");
							for(i = 0; i < **(uniqueOut + j); 
								printf("%hu ", *(vecPos + i)),
								++i);
							printf("\b]\n");
						#endif

						//Caculate the totalScore for this argument on this iteration
						//First, clean totalScore from (previous iteration)/junk
						*(totalScore + j) = 0;
						//Its necessary to keep track of every coefficient,
						//to known if this verifycation leads to a LEAF.
						//Entropy = -(Positives/Total)*log2(Positives/Total) -(Negatives/Total)*log2(Negatives/Total)
						#ifdef DEBUG
							printf("d: (entropy calculus) ");
						#endif
						for (i = 0; i < **(uniqueOut + j); ++i){
							aux0 = *(vecPos + i)/(1.0 * *(vecCounter + i));
							aux1 = (*(vecCounter + i) - *(vecPos + i))/(1.0 * (*(vecCounter + i)));
							
							*(entropyCoef + i) = 0;
							if (aux0 > 0 && aux1 > 0)
								*(entropyCoef + i) = - (aux0)*log2(aux0) - (aux1)*log2(aux1);
							
							*(totalScore + j) += *(entropyCoef + i);

							#ifdef DEBUG
								printf("%lf ", *(entropyCoef + i));
							#endif
						}
						#ifdef DEBUG
							printf("- total: %lf\n", *(totalScore + j));
						#endif

						//free positive vector and vecCounter for the next iteration
						free(vecPos);
						free(vecCounter);

						//Work from here
						//Set up the new nodes on the tree
						//The promoted node will be the one who have the smallest entropy

						//demark the promoved parameter on the vecUsed.

						//Verify if all new "leaf" nodes are a leaf indeed.
						//Use the entropyCoef vector to this job
						//If TRUE, turn FLAG off. if not, MARK all data used on this iteration.

						//free entropy vector for the next iteration
						free(entropyCoef);
						#ifdef DEBUG
							printf("d: end of this argument.\n**************\n");
						#endif
					}
				}
				//"k" is a security counter, which keep track
				//if all arguments are used for the tree construction.
				//if true, force proccess end.
				--k;
				#ifdef DEBUG
					printf("d: end of iteration.\n###########################\n");
				#endif
			} while (FLAG && k > 0);

			//Freeing used memory
			if (uniqueOut != NULL)
				dataPurge(uniqueOut, (colNum - 1));
			if (totalScore != NULL)
				free(totalScore);
			if (vecUsed != NULL)
				free(vecUsed);
		}
		return model;
	}
	printf("e: no samples or parameters found on \"%s\" function.\n", __FUNCTION__);
	return NULL;
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
	if (argc == NUMARGS){
		FILE *ftrain = fopen(*(argv + TRAINPATH), "r");
		if (ftrain != NULL){
			FILE *finput = fopen(*(argv + INPUTPATH), "r");
			if (finput != NULL){
				lbyte colNum = 0, examplesNum = 0;
				byte **data = dataGet(ftrain, &colNum, &examplesNum);
				if (data != NULL){
					#ifdef DEBUG
						printf("d: will now construct ID tree model...\n");
					#endif
					ITM *model = itModel(data, colNum, examplesNum);
					#ifdef DEBUG
						printf("d: will start training and labelling process...\n");
					#endif
					if (model != NULL){
						itPredict(model, finput);
						itPurge(&model);
					}
					#ifdef DEBUG
						printf("d: now going to free used memory...\n");
					#endif
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