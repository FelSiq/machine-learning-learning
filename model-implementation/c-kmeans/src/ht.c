#include <stdlib.h>
#include <stdio.h>
#include "ht.h"

struct ht {
	unsigned int **matrix, groupNum;
};

ht *htInit(){
	ht *hashTable = malloc(sizeof(ht));
	if (hashTable != NULL){
		hashTable->matrix = NULL;
		hashTable->groupNum = 0;
	}
	return hashTable;
};

void htPrint(ht *hashTable){
	if (hashTable != NULL){
		for (size_t i = 0; i < hashTable->groupNum; printf("\n"), ++i){
			printf("group #%lu: ", i + 1);
			for (unsigned int j = 0; j < *(1 + *(hashTable->matrix + i)); 
				printf("%u ", *(*(hashTable->matrix + i) + j + 2)),
				++j);
		}
	}
};

static void destroyMatrix(unsigned int **matrix, size_t rows){
	if (matrix != NULL){
		if (rows > 0)
			while(0 < rows--)
				free(*(matrix + rows));
		free(matrix);
	}
};

void htAll(ht *hashTable, unsigned char *keys, size_t keysSize){
	if (hashTable != NULL && keys != NULL){
		if (hashTable->matrix != NULL)
			destroyMatrix(hashTable->matrix, hashTable->groupNum);
		hashTable->matrix = NULL;
		hashTable->groupNum = 0;
		for (register size_t i = 0; i < keysSize; ++i){

			//Expand hashTable keys
			if (*(keys + i) >= hashTable->groupNum){

				hashTable->matrix = realloc(hashTable->matrix, 
					sizeof(unsigned int *) * (1 + *(keys + i)));

				for(size_t k = hashTable->groupNum; k <= *(keys + i); ++k){
					*(hashTable->matrix + k) = malloc(sizeof(unsigned int) * (2 + STD_NEW_SIZE));
					**(hashTable->matrix + k) = STD_NEW_SIZE;
					*(1 + *(hashTable->matrix + k)) = 0;
					++(hashTable->groupNum);
				}
			}

			//Expand hashTable rows
			if (*(1 + *(hashTable->matrix + *(keys + i))) >= *(*(hashTable->matrix + *(keys + i)))){
				*(*(hashTable->matrix + *(keys + i))) *= 2;
				*(hashTable->matrix + *(keys + i)) = realloc(*(hashTable->matrix + *(keys + i)), 
					sizeof(unsigned int) * (2 + *(*(hashTable->matrix + *(keys + i)))));
			};

			//Hash indeces
			(hashTable->matrix)[keys[i]][hashTable->matrix[keys[i]][1] + 2] = i;
			++(hashTable->matrix[keys[i]][1]);
		}
	}
};

void htPurge(ht **hashTable){
	if (hashTable != NULL && *hashTable != NULL){
		destroyMatrix((*hashTable)->matrix, (*hashTable)->groupNum);
		free(*hashTable);
		(*hashTable) = NULL;
	}
};