#include <stdlib.h>
#include <stdio.h>
#include "it.h"

int main(int argc, char const *argv[]){
	if (argc == NUMARGS){
		FILE *ftrain = fopen(*(argv + TRAINPATH), "r");
		if (ftrain != NULL){
			FILE *finput = fopen(*(argv + INPUTPATH), "r");
			if (finput != NULL){
				size_t colNum = 0, examplesNum = 0;
				byte **data = dataGet(ftrain, &colNum, &examplesNum);
				if (data != NULL){
					#ifdef DEBUG
						printf("d: will now construct ID tree model...\n");
					#endif
					ITM *model = itModel(data, colNum, examplesNum);
					#ifdef DEBUG
						printf("d: model complete. result:\n");
						itPrint(model);
						printf("d: will start predict process...\n");
					#endif
					if (model != NULL){
						itPredict(model, finput);
						#ifdef DEBUG
							printf("d: now going to destroy model to free memory.\n");
						#endif
						itPurge(&model);
					} else printf("e: can't create model, something went wrong.\n");
					
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
			printf("e: can't open input file.\n");
			fclose(ftrain);
			return 3;
		}
		printf("e: can't open train file.\n");
		return 2;
	}
	printf("usage: %s <train path> <input path>\n", *(argv + PROGNAME));
	return 1;
};