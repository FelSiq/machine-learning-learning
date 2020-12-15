#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "it.h"

typedef struct node {
	byte param, *args;
	size_t argNum;
	//Param stands for parameter to be check on this node, to
	//select the son that the input must follow in order to be identified.
	//Args stand for the values to be check.
	//This values should cover all the possibilities.
	//If not, the deeper sub-tree will be default selected.
	double entropy;
	//Entropy will store the accuracy of node, if it is a leaf,
	//to show up in the print function.
	char output;
	//This value is used only on leaf nodes. It represents the
	//input identification.
	//default value from nodeInit func is -1.
	struct node **sons;
	size_t numSons;
} NODE;

struct itm{
	NODE *root;
	size_t numNodes, argNum;
};

byte **dataGet(FILE *ftrain, size_t *colNum, size_t *examplesNum){
	byte **data = NULL;
	if (ftrain != NULL){
		//Get the argNum
		for(char c = 0; c != EOF && c != '\n'; c = fgetc(ftrain))
			*colNum += (c == ' ');
		if (ftell(ftrain) > 0)
			++(*colNum);
		fseek(ftrain, 0, SEEK_SET);

		#ifdef DEBUG
			printf("d: number of parameters on each sample: %lu (include label)\n", *colNum);
		#endif

		if (*colNum > 0){
			int aux;
			while(!feof(ftrain)){
				data = realloc(data, sizeof(byte *) * (1 + *examplesNum));
				*(data + *examplesNum) = malloc(sizeof(byte) * *colNum);
				for(size_t i = 0; i < *colNum; ++i){
					fscanf(ftrain, "%d%*c", &aux);
					*(*(data + *examplesNum) + i) = (byte) aux;
				}
				++(*examplesNum);
			}
		}

		#ifdef DEBUG
			printf("d: dataset height: %lu\nd: will print out head of dataset:\n", *examplesNum);
			size_t i, j;
			for(i = 0; i < MIN(HEADSIZE, *examplesNum); printf("\n"), ++i)
				for(j = 0; j < *colNum; 
					printf("%d ", *(*(data + i) + j)),
					++j);
		#endif
	}
	return data;
};

void dataPurge(byte **data, size_t examplesNum){
	if (data != NULL){
		while(0 < examplesNum--)
			free(*(data + examplesNum));
		free(data);
	}
};

void datalPurge(lbyte **data, size_t examplesNum){
	if (data != NULL){
		while(0 < examplesNum--)
			free(*(data + examplesNum));
		free(data);
	}
};

static NODE *nodeInit(){
	NODE *node = malloc(sizeof(NODE));
	if (node != NULL){
		node->args = NULL;
		node->sons = NULL;
		node->entropy = -1;
		node->output = -1;
		node->numSons = 0;
		node->argNum = 0;
		node->param = 0;
	};
	return node;
};

static ITM *itInit(){
	ITM *model = malloc(sizeof(ITM));
	if (model != NULL){
		model->root = NULL;
		model->numNodes = 0;
		model->argNum = 0;
	}
	return model;
};

inline static long int binSearch(byte *vector, byte key, size_t start, size_t size){
	long int i = start, j = size, middle;
	while (j >= i){
		middle = (i + j)*0.5;
		//if found, return the index
		if (*(vector + middle) == key)
			return middle;
		else {
			if (*(vector + middle) > key)
				j = middle - 1;
			else
				i = middle + 1;
		}
	}
	//If not found, a invalid index
	return (start - 1);
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

ITM *itModel(byte **data, size_t const colNum, size_t const examplesNum){
	if (colNum > 1 && examplesNum > 0){
		ITM *model = itInit();
		if (model != NULL){
			model->argNum = (colNum - 1);
			//This is the max value generalized, for a worst-case scenario, of entropy in every parameter
			double const entropyInit = 1.1;//(1.0 * examplesNum) + 0.1;
			NODE *newNode = NULL;
			size_t newNodeIndex = 0, newLeafIndex = 0;
			byte FLAG = 1;
			size_t k = (colNum - 1);
			////Vectors
			//vector of total Score (disorder coefficient), and for entropy on each parameter
			double *totalScore = malloc(sizeof(double) * k), 
			**entropyCoef = malloc(sizeof(double *) * k), 
			aux0 = 0, aux1 = 0, aux2 = 0;
			//boolean vector of FLAGs to keep track of used parameters
			byte *vecUsed = malloc(sizeof(byte) * k);
			//Get all possible unique values from all parameters;
			//The first column of each row (index 0) is reserved for row size
			byte **uniqueOut = malloc(sizeof(byte *) * k);
			//POSITIVE and NEGATIVE vectors;
			//These vectors are used to entropy (disorder coefficient) calculation
			lbyte **vecPos = malloc(sizeof(lbyte *) * k);
			//counter Vector, to store a information necessary on entropy calculus
			size_t **vecCounter = malloc(sizeof(size_t *) * k);
			//Parameter for entropy calculus
			size_t totalInfo = 0;
			
			//Clean up vecUsed and set up uniqueOut
			register size_t j = 0, n = 0, i = 0, m = 0;
			for (i = 0; i < k; *(vecUsed + i) = 1, ++i){
				*(uniqueOut + i) = malloc(sizeof(byte));
				**(uniqueOut + i) = 0;
				//Get unique outcomes
				for (m = 0; m < examplesNum; ++m){
					if (!binSearch(*(uniqueOut + i), *(*(data + m) + i), 1, **(uniqueOut + i))){
						*(uniqueOut + i) = realloc(*(uniqueOut + i), 
							sizeof(byte *) * (1 + (**(uniqueOut + i))));
						*(*(uniqueOut + i) + **(uniqueOut + i) + 1) = *(*(data + m) + i);
						++(**(uniqueOut + i));

						//Insert on the sorted position
						for(n = **(uniqueOut + i); 
							n > 1 && (*(*(uniqueOut + i) + n) < *(*(uniqueOut + i) + n - 1)); 
							--n)
							SWAP(*(*(uniqueOut + i) + n), *(*(uniqueOut + i) + n - 1));

					}
				}
			}

			#ifdef DEBUG
				//DEBUG - Print unique outcomes
				printf("d: will now print UNIQUE outcomes:\n");
				for (j = 0; j < k; printf("\n"), ++j)
					for (n = 0; n < **(uniqueOut + j); 
						printf("%d ", *(*(uniqueOut + j) + n + 1)),
						++n);
			#endif

			//Loop to construct the tree itself
			do {
				#ifdef DEBUG
					printf("d: just started a new iteration...\n");
				#endif
				for(j = 0; j < (colNum - 1); ++j){
					if (*(vecUsed + j)){
						byte const register cache_uniqueOut_j = **(uniqueOut + j);
						#ifdef DEBUG
							printf("d: started working on #%lu argument.\n", j);
						#endif
						//Set up entropy vector for this parameter on this iteration
						*(entropyCoef + j) = malloc(sizeof(double) * cache_uniqueOut_j);
						//Set up pos/neg vectors
						*(vecPos + j) = malloc(sizeof(lbyte) * cache_uniqueOut_j);
						//Same with counter vector
						*(vecCounter + j) = malloc(sizeof(size_t) * cache_uniqueOut_j);
						//clean vectors up
						for (n = 0; n < cache_uniqueOut_j; ++n){
							*(*(vecPos + j) + n) = 0;
							*(*(vecCounter + j) + n) = 0;
							*(*(entropyCoef + j) + n) = 0;
						}

						//Fill up neg/pos vectors
						for(m = 0; m < examplesNum; ++m){
							if (*(*(data + m) + colNum - 1) != MARKED){
								for (i = 0; i < cache_uniqueOut_j && 
									*(*(uniqueOut + j) + i + 1) != *(*(data + m) + j);
									++i);
								*(*(vecPos + j) + i) += (lbyte) MIN(1, (*(*(data + m) + colNum - 1) - ADJUST)); 
								++(*(*(vecCounter + j) + i));
							}
						}
						#ifdef DEBUG
							printf("d: positive vector on this iteration: [");
							for(i = 0; i < cache_uniqueOut_j; 
								printf("%hu ", *(*(vecPos + j) + i)),
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
						//Sum up the total amount of information on this node,
						//if its the first argument of this iteration
						if (totalInfo == 0)
							for (i = 0; i < cache_uniqueOut_j; 
								totalInfo += *(*(vecCounter + j) + i),
								++i); 

						for (i = 0; i < cache_uniqueOut_j; ++i){
							aux0 = *(*(vecPos + j) + i)/(1.0 * *(*(vecCounter + j) + i));
							aux1 = (*(*(vecCounter + j) + i) - *(*(vecPos + j) + i))/(1.0 * (*(*(vecCounter + j) + i)));
							aux2 = (*(*(vecCounter + j) + i))/(1.0 * totalInfo);
							*(*(entropyCoef + j) + i) = 0;
							if (aux0 > 0 && aux1 > 0)
								*(*(entropyCoef + j) + i) = -((aux0)*log2(aux0) + (aux1)*log2(aux1)) * (aux2);
							
							*(totalScore + j) += *(*(entropyCoef + j) + i);

							#ifdef DEBUG
								printf("%lf ", *(*(entropyCoef + j) + i));
							#endif
						}

						#ifdef DEBUG
							printf("- total: %lf\nd: end of this argument.\n**************\n", *(totalScore + j));
						#endif
					}
				}
				
				//Set up the new nodes on the tree
				//The promoted node will be the one who have the smallest entropy
				aux0 = entropyInit;//Entropy image is in [0,1].
				for (i = 0; i < (colNum - 1); ++i){
					if (*(vecUsed + i) && aux0 > *(totalScore + i)){
						aux0 = *(totalScore + i);
						newNodeIndex = i;
					}
				}
				//Verify if this is not the first iteration
				//if TRUE, set the newNode the new son of previous newNode. 
				if (newNode != NULL){
					*(newNode->sons + newNode->numSons - 1) = nodeInit();
					newNode = *(newNode->sons + newNode->numSons - 1);
				} else {
					model->root = nodeInit();
					newNode = model->root;
				}
				++(model->numNodes);
				
				//First, verify if it is not the last possible iteration.
				//If false, ignore this section.
				byte const register cache_uniqueOut_nni = **(uniqueOut + newNodeIndex);
				if (k > 1){
					//If true,
					//Select the most messet up subtree.
					//It will be the only one to be not a new leaf, except
					//if its entropy are small enought.
					aux0 = -1;
					for (i = 0; i < cache_uniqueOut_nni; ++i){
						if (aux0 < *(*(entropyCoef + newNodeIndex) + i)){
							aux0 = *(*(entropyCoef + newNodeIndex) + i);
							newLeafIndex = i;
						}
					}
					//
					if (*(*(entropyCoef + newNodeIndex) + newLeafIndex) < DELTA)
						newLeafIndex = MARKED;
				} else newLeafIndex = MARKED;

				//Set up the terminal/leaf nodes
				for (i = 0; i < cache_uniqueOut_nni; ++i){
					if (i != newLeafIndex){
						newNode->sons = realloc(newNode->sons, 
							sizeof(NODE *) * (newNode->numSons + 1));
						*(newNode->sons + newNode->numSons) = nodeInit();
						//get the "most common label" correspondent of this node
						(*(newNode->sons + newNode->numSons))->output = ADJUST + 
							(*(*(vecPos + newNodeIndex) + i) >= (*(*(vecCounter + newNodeIndex) + i) * 0.5));
						(*(newNode->sons + newNode->numSons))->entropy = *(*(entropyCoef + newNodeIndex) + i);
						++(newNode->numSons);

						//Add one unit to numNodes on model
						++(model->numNodes);

						#ifdef DEBUG
							printf("d: generated a new output node:\n\t- label value of %d.\n\t- entropy: %lf\n", 
								(*(newNode->sons + newNode->numSons - 1))->output,
								(*(newNode->sons + newNode->numSons - 1))->entropy);
						#endif
					}
				}
				//demark the promoted parameter on the vecUsed.
				*(vecUsed + newNodeIndex) = 0;

				//Set up all necessary parameters of this new node.
				newNode->argNum = cache_uniqueOut_nni;
				newNode->param = newNodeIndex;
				newNode->args = malloc(sizeof(byte) * newNode->argNum);
				newNode->entropy = *(*(entropyCoef + newNodeIndex) + newLeafIndex);//*(totalScore + newNodeIndex);
				for(i = 0; i < newNode->argNum; ++i)
					*(newNode->args + i) = *(*(uniqueOut + newNode->param) + 1 + i);
				#ifdef DEBUG
					printf("d: stabilished a new tree node:\n\t- column: %hhu\n\t- # of sons: %lu\n\t- total entropy: %lf\n",
						newNode->param,
						newNode->argNum,
						newNode->entropy);
				#endif

				//Verify if all new "leaf" nodes are a leaf indeed.
				//If TRUE, turn FLAG off. if not, MARK all data used on this iteration,
				//and incresease the size of sons of the newNode by one.

				if (newNode->argNum != newNode->numSons){
					#ifdef DEBUG
						printf("d: this is not the last iteration. Will mark used data.\n");
					#endif
					//Open space to the node of next iteration
					newNode->sons = realloc(newNode->sons,
						sizeof(NODE *) * (newNode->numSons + 1));
					*(newNode->sons + newNode->numSons) = NULL;
					++newNode->numSons;
					//Mark all used data
					//Remember that the deeper subtree will be in the (newNode->argnum - 1) index;
					for(m = 0; m < examplesNum; ++m){
						for (i = 0; i < (newNode->argNum - 1); ++i){
							if (*(*(data + m) + newNode->param) == *(newNode->args + i)){
								*(*(data + m) + colNum - 1) = MARKED;
								break;
							}
						}
					}
				} else {
					#ifdef DEBUG
						printf("d: < reached the deepest leaf node on the tree! >\n");
					#endif
					FLAG = 0;
				}
				
				//Free memory from value tables for the next iteration
				for(i = 0; i < (colNum - 1); ++i){
					free(*(entropyCoef + i));
					free(*(vecCounter + i));
					free(*(vecPos + i));
					*(entropyCoef + i) = NULL;
					*(vecCounter + i) = NULL;
					*(vecPos + i) = NULL;
				}
				//Clean up totalInfo counter for the next iteration
				totalInfo = 0;
				//"k" is a security lever, which keep track
				//if all arguments are used for the tree construction.
				//if true, force proccess end.
				--k;
				#ifdef DEBUG
					printf("d: end of iteration.\n###########################\n");
				#endif
			} while (FLAG && k > 0);

			//Freeing used memory
			if (entropyCoef != NULL)
				free(entropyCoef);
			if (vecCounter != NULL)
				free(vecCounter);
			if (vecPos != NULL)
				free(vecPos);
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
	if (model != NULL && model->root != NULL
		&& finput != NULL && !feof(finput)){
		lbyte *data = malloc(sizeof(lbyte) * (model->argNum));
		if (data != NULL){
			size_t i, counter = 0;
			long int index;
			while(!feof(finput)){
				//Get data from file
				for (i = 0; i < model->argNum; ++i)
					fscanf(finput, "%hu%*c", (data + i));

				//Travel throught the model
				NODE *traveller = model->root;
				while(traveller->sons != NULL){
					index = binSearch(traveller->args, *(data + traveller->param), 0, traveller->numSons);
					if (index == -1)
						index = (traveller->numSons - 1);
					traveller = *(traveller->sons + index);
				}
				printf("prediction #%lu: %d\n", ++counter, traveller->output);
			}
			free(data);
		} 
	}
};

static void itPurge_rec(NODE *root){
	if (root != NULL){
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

static void itPrint_rec(NODE *root, lbyte offset){
	if (root != NULL){
		if (root->numSons > 0){
			printf("<path node> # of sons: %lu (entropy: %lf)\n", 
				root->numSons,
				root->entropy);
			for (lbyte i = 0; i < root->numSons; ++i){
				for (lbyte i = 0; i < offset + STD_OFFSET; 
					printf(" "), ++i);
				printf("-> (#%hhu col = %d) ",
					root->param,
					*(root->args + i));
				itPrint_rec(*(root->sons + i), offset + STD_OFFSET);
			}
		} else {
			printf("<leaf> label: %d (entropy: %lf)\n", 
				root->output,
				root->entropy);
		}
	}
};

void itPrint(ITM *model){
	if (model != NULL){
		if (model->root != NULL){
			printf("total nodes on this model: %lu\n-> ", model->numNodes);
			itPrint_rec(model->root, 0);
		} else printf("-> this model is empty.\n");
	}
};