#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ht.h"
#include <time.h>

#define sqr(x)	((x)*(x))
#define ABS(x) ((x) < 0 ? (-(x)) : (x))
#define EPSOLON 0.00001

typedef unsigned char byte;

enum {
	PROG_NAME,
	K_PARAM,
	DATA_FILE,
	SEED,
	ARG_NUM
};

double SumEuclidDist(size_t obs_num, double *argsA, double *argsB){
	double dis = 0;
	while(0 < obs_num--)
		dis += sqr(*(argsA + obs_num) - *(argsB + obs_num));
	dis = pow(dis, 0.5);
	return dis;
};

#ifdef DEBUG
	void print_data(double **data, size_t rows, size_t obs_num){
		for(size_t k = 0; k < rows; printf("\n"), ++k){
			printf("%lu. ", k);
			for(size_t l = 0; l < obs_num; 
				printf("%lf\t", *(*(data + k) + l)), ++l);
		}
	}
#endif

double **getData(size_t *dataSize, size_t *paramNum, char const fname[]){
	FILE *fp = fopen(fname, "r");
	if (fp != NULL){
		double **dataTrain = NULL;
		//Get the paramNum
		for(char c = 0; c != EOF && c != '\n'; c = fgetc(fp))
			if (c == ' ')
				++(*paramNum);
		if (ftell(fp) > 0)
			++(*paramNum);
		#ifdef DEBUG
			printf("D: number of parameters: %lu\n", *paramNum);
		#endif
		//Get data
		fseek(fp, 0, SEEK_SET);
		if (*paramNum > 0){
			while(!feof(fp)){
				dataTrain = realloc(dataTrain, sizeof(double *) * (1 + *dataSize));
				*(dataTrain + *dataSize) = malloc(sizeof(double) * *paramNum);
				for (int k = 0; k < *paramNum; ++k)
					fscanf(fp, "%lf%*c", (*(dataTrain + *dataSize) + k));
				++(*dataSize);
			}
		}
		#ifdef DEBUG
			printf("D: dataSize: %lu\n", *dataSize);
		#endif
		fclose(fp);
		return dataTrain;

	} else printf("error: can't open \"%s\" on %s.\n", fname, __FUNCTION__);
	return NULL;
};

void destroyData (double **data, size_t dataSize){
	if (data != NULL){
		for (byte i = 0; i < dataSize; 
			free(*(data + i)),
			++i);
		free(data);
	}
};

void kmeans(double **data, size_t dataSize, 
	size_t paramNum, const byte k, const unsigned int seed){
	if (k > 0 && k < dataSize){
		//##########################################
		//Resources
		byte i, j, boolFlag, sampleIndex,
		*outputClass = malloc(sizeof(byte) * dataSize);
		//Malloc class vector and means matrix
		double **means = malloc(sizeof(double *) * k),
		**newMeans = malloc(sizeof(double *) * k), 
		dist, aux, diff;
		
		size_t *classCounter = malloc(sizeof(size_t) * k), n,
		randIndex, *selectedVals = malloc(sizeof(size_t) * k);
		//##########################################
		//Set the random seed
		srand(seed);
		//Select the k initial models,
		for (i = 0; i < k; ++i){
			*(means + i) = malloc(sizeof(double) * paramNum);
			*(newMeans + i) = malloc(sizeof(double) * paramNum);
			boolFlag = 1;

			//All samples must be different
			do {
				randIndex = (rand() % dataSize);
				for(j = 0; j < i; ++j)
					if(*(selectedVals + j) == randIndex)
						break;
				boolFlag = !(j == i);
			} while(boolFlag);
			//Refresh new index
			*(selectedVals + i) = randIndex;

			for(j = 0; j < paramNum; 
				*(*(means + i) + j) = *(*(data + randIndex) + j),
				++j);

			#ifdef DEBUG
				printf("D: selected initial sample index: %lu\n", randIndex);
				for (n = 0; n < paramNum; 
					printf("%lf\t", *(*(means + i) + n)),
					++n);
				printf("\n");
			#endif
		}

		do{
			//Clean up the classCounter and newMeans
			for(i = 0; i < k; ++i){
				*(classCounter + i) = 0;
				for(n = 0; n < paramNum; ++n)
				 	*(*(newMeans + i) + n) = 0;
			}

			//Get the new set of newMeans (nearest neighbour)
			for(n = 0; n < dataSize; ++n){
				dist = 1.0 * (1 << ((unsigned) (sizeof(unsigned int) * 8) - 2));
				for(i = 0; i < k; ++i){
					aux = SumEuclidDist(paramNum, *(data + n), *(means + i));
					if (dist > aux){
						dist = aux;
						sampleIndex = i;
					};
				};
				
				for(size_t z = 0; z < paramNum; ++z)
					*(*(newMeans + sampleIndex) + z) += *(*(data + n) + z);
				++(*(classCounter + sampleIndex));

				//Register the class for output purposes
				*(outputClass + n) = sampleIndex;
			}

			//Divide the summation by the number of elements to get the newMeans
			for(i = 0; i < k; ++i)
				if (*(classCounter + i) > 0)
					for(n = 0; n < paramNum; ++n)
						*(*(newMeans + i) + n) /= (1.0 * *(classCounter + i));
			#ifdef DEBUG
				printf("D: new means on this iteration:\n");
				for(i = 0; i < k; printf("\n"), ++i)
					for(n = 0; n < paramNum; ++n)
						printf("%lf ", newMeans[i][n]);
			#endif
			//Calculate the diff
			diff = 0;
			for (i = 0; i < k; ++i)
				diff += SumEuclidDist(paramNum, *(newMeans + i), *(means + i));
			diff /= (double) (1.0 * k);
			#ifdef DEBUG
				printf("D: diff on this iteration: %lf\n", diff);
			#endif
			//refresh means
			for (i = 0; i < k; ++i)
				for(n = 0; n < paramNum; ++n)
					*(*(means + i) + n) = *(*(newMeans + i) + n);
		} while (diff > EPSOLON);

		//Free a bunch of memory used on this section
		#ifdef DEBUG
			printf("D: clustering completed. freeing some memory...\n");
		#endif
		free(selectedVals);
		free(classCounter);
		destroyData(means, k);
		destroyData(newMeans, k);
		#ifdef DEBUG
			printf("D: now going to print result:\n");
			for(size_t TEST = 0; TEST < dataSize; ++TEST)
				printf("%hhu ", outputClass[TEST]);
			printf("\n");
		#endif
		//Print desired output
		//I'll use a hash table for speed.
		ht *hashTable = htInit();
		htAll(hashTable, outputClass, dataSize);
		htPrint(hashTable);
		htPurge(&hashTable);
		free(outputClass);

	} else printf("e: parameter \"k\" (%hhu) must be smaller than the size of dataset (%lu), and larger than 0.\n", k, dataSize);
};

int main(int argc, char const *argv[]){
	if (argc == ARG_NUM){
		size_t dataSize = 0, paramNum = 0;
		double **data = getData(&dataSize, &paramNum, *(argv + DATA_FILE));
		if (data != NULL){
			#ifdef DEBUG
				print_data(data, dataSize, paramNum);
			#endif
			long int seed = atol(*(argv + SEED));
			if (seed == -1){
				srand(time(NULL));
				seed = rand();
			} else seed = ABS(seed);

			kmeans(data, dataSize, paramNum, 
				ABS(atoi(*(argv + K_PARAM))), seed);
			destroyData(data, dataSize);
			return 0;
		}
		printf("e: can't read data.\n");
		return 2;
	}
	printf("usage: %s <K> <data path> <seed>\n", *(argv + PROG_NAME));
	return 1;
}