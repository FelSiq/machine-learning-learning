#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mlp.h"

static double sigmoidFunc(double x, double lambda){
	return 1.0/(1.0 + exp(-lambda * x));
};

bool **getData(char const fname[]){
	FILE *fp = fopen(fname, "r");
	if (fp != NULL){
		bool **dataTrain = malloc(sizeof(bool *) * (1 + VAR_NUM));

		if (dataTrain != NULL){
			byte i, j;
			for (i = 0; i <= VAR_NUM; 
				*(dataTrain + i) = malloc(sizeof(bool) * TABLE_HEIGHT),
				++i);
			for(i = 0; i < TABLE_HEIGHT; ++i)
				for(j = 0; j <= VAR_NUM; 
					fscanf(fp, "%hhu%*c", (*(dataTrain + j) + i)), 
					++j);
		}

		fclose(fp);
		return dataTrain;

	} else printf("error: can't open \"%s\" on %s.\n", fname, __FUNCTION__);
	return NULL;
};

void destroyData(bool **data){
	if (data != NULL){
		for (byte i = 0; i <= VAR_NUM; 
			free(*(data + i)),
			++i);
		free(data);
	}
};

#ifdef DEBUG
	void printData(bool **data){
		if (data != NULL){
			for (byte i = 0; i < TABLE_HEIGHT; printf("\n"), ++i)
				for(byte k = 0; k <= VAR_NUM; 
					printf("%hhu ", *(*(data + k) + i)),
					++k);
		}
	};
#endif

void neuralTraining(network *map, bool **dataTrain, ui max_it, double delta, double rate){
	double totalError, aux;
	ui it = 0;
	register byte i;
	while(max_it > it++){
		totalError = 0;
		printf("iteration #%u of %u...\n", it, max_it);
		for(ui k = 0; k < TABLE_HEIGHT; ++k){
			//Set the input values on the input nodes
			#ifdef DEBUG
				printf("D: Now on table row #%hhu...\nD: receiving inputs...\n", k);
			#endif
			for(i = 0; i < VAR_NUM; ++i)
				*(map->input + i) = *(*(dataTrain + i) + k);
			//Calculate the value on hidden layer (sum of input values x weigth)
			#ifdef DEBUG
				printf("Inputs: %hhu and %hhu.\nD: now starting hidden node calculation...\n", 
					*(map->input + INPUT_A), *(map->input + INPUT_B));
			#endif
			*(map->act + HIDDEN) = 0;
			for(i = 0; i < VAR_NUM; ++i)
				*(map->act + HIDDEN) += *(map->input + i) * *(*(map->weigth + i) + HDN_CNT);
			*(map->act + HIDDEN) = sigmoidFunc(*(map->act + HIDDEN), 1.0);
			//Calculate the value on outputLayer
			#ifdef DEBUG
				printf("D: now starting output node calculation...\n");
			#endif
			*(map->act + OUTPUT) = 0;
			for(i = 0; i <= VAR_NUM; ++i)
				*(map->act + OUTPUT) += *(map->act + i) * *(*(map->weigth + i) + OUT_CNT);
			*(map->act + OUTPUT) = sigmoidFunc(*(map->act + OUTPUT), 1.0);
			//Calculate the output error
			#ifdef DEBUG
				printf("D: error calculation: ");
			#endif
			aux = *(*(dataTrain + VAR_NUM) + k) - *(map->act + OUTPUT);
			totalError += sqr(aux);
			#ifdef DEBUG
				printf("%lf\nD: now adjusting weights...\n", aux);
			#endif
			//Adjust the weights of conections to Outputlayer
			for(i = 0; i <= VAR_NUM; ++i)
				*(*(map->weigth + i) + OUT_CNT) = *(*(map->weigth + i) + OUT_CNT) + (aux * rate);
			//Adjust the weigths on the conections to HiddenLayer
			for(i = 0; i < VAR_NUM; ++i)
				*(*(map->weigth + i) + HDN_CNT) = *(*(map->weigth + i) + HDN_CNT) + (aux * rate);
			//Repeat
			#ifdef DEBUG
				printf("D: row completed.\n");
			#endif
		};
		#ifdef DEBUG
			printf("D: totalError on this iteration: %lf\tmean:%lf\n", totalError, totalError/(1.0 * TABLE_HEIGHT));
		#endif
		if (totalError/(1.0 * TABLE_HEIGHT) < delta){
			#ifdef DEBUG
				printf("D: totalError is smaller than delta, training completed.\n");
			#endif
			it = (max_it + 1);
		}
	}
	#ifdef DEBUG
		printf("D: training is completed.\n");
	#endif
};

void neuralPredict(network *map, FILE *fi){
	register byte i;
	while(!feof(fi)){
		fscanf(fi, "%hhu%*c%hhu%*c", (map->input + INPUT_A), (map->input + INPUT_B));
		//Calculate the output on inputA and inputB
		//Calculate the value on hidden layer (sum of input values x weigth)
		*(map->act + HIDDEN) = 0;
		for(i = 0; i < VAR_NUM; ++i)
			*(map->act + HIDDEN) += *(map->input + i) * *(*(map->weigth + i) + HDN_CNT);
		*(map->act + HIDDEN) = sigmoidFunc(*(map->act + HIDDEN), 1.0);
		//Calculate the value on outputLayer
		*(map->act + OUTPUT) = 0;
		for(i = 0; i <= VAR_NUM; ++i)
			*(map->act + OUTPUT) += *(map->act + i) * *(*(map->weigth + i) + OUT_CNT);
		*(map->act + OUTPUT) = sigmoidFunc(*(map->act + OUTPUT), 1.0);
		printf("neural prediction: %lf\n", *(map->act + OUTPUT));
	}
};

long int randRange(int min, int max){
	return (rand() % (max + 1 - min)) + min;
};

network *neuralInit(){
	network *map = malloc(sizeof(network));
	if (map != NULL){
		map->act = malloc(sizeof(double) * NODE_NUM);
		map->bias = malloc(sizeof(double) * NODE_NUM);
		map->input = malloc(sizeof(bool) * VAR_NUM);

		map->weigth = malloc(sizeof(double *) * (NODE_NUM - 1));

		*(map->weigth + INPUT_A) = malloc(sizeof(double) * 2);
		map->weigth[INPUT_A][0] = randRange(STD_MIN, STD_MAX);
		map->weigth[INPUT_A][1] = randRange(STD_MIN, STD_MAX);

		*(map->weigth + INPUT_B) = malloc(sizeof(double) * 2);
		map->weigth[INPUT_B][0] = randRange(STD_MIN, STD_MAX);
		map->weigth[INPUT_B][1] = randRange(STD_MIN, STD_MAX);

		*(map->weigth + HIDDEN) = malloc(sizeof(double));
		map->weigth[HIDDEN][0] = randRange(STD_MIN, STD_MAX);
	}
	return map;
};

void neuralPurge(network **map){
	if (map != NULL && *map != NULL){
		if ((*map)->input != NULL)
			free((*map)->input);
		if ((*map)->bias != NULL)
			free((*map)->bias);
		if ((*map)->act != NULL)
			free((*map)->act);
		if ((*map)->weigth != NULL){
			for(byte i = 0; i < (NODE_NUM - 1); 
				free(*((*map)->weigth + i)),
				++i);
			free((*map)->weigth);
		}
		free(*map);
		(*map) = NULL;
	}
};

int main(int argc, char const *argv[]){
	if (argc == ARG_NUM){
		//Set user random seed
		srand(ABS(atol(*(argv + SEED))));

		//Read training data
		bool **dataTrain = getData(*(argv + TRAIN_FILE));
		if (dataTrain != NULL){
			//Verify if input file exists and is readable
			FILE *fi = fopen(*(argv + INPUT_FILE), "r");
			if (fi != NULL){
				#ifdef DEBUG
					printData(dataTrain);
				#endif
				network *map = neuralInit();
				//Train neural network
				neuralTraining(map, dataTrain, atoi(argv[MAX_ITERATION]), DELTA, RATE);
				//Predict inputs
				neuralPredict(map, fi);
				//Free all remaining RAM memory used
				neuralPurge(&map);
				destroyData(dataTrain);
				fclose(fi);
				return 0;
			}
			printf("error: can not open input file.\n");
			return 3;
		}
		printf("error: can not get data.\n");
		return 2;
	}
	printf("usage: %s <seed> <train path> <input path> <max iterations>\n", *(argv + PROG_NAME));
	return 1;
};