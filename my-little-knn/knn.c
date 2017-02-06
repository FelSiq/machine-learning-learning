#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

enum {
	PROG_NAME,
	K_PARAM,
	OBS_NUM,
	TEST_NAME,
	DATA_NAME,
	ARG_NUM
};

long double euclid_dis(unsigned int obs_num, long double *argsA, long double *argsB){
	long double dis = 0;

	#ifdef DEBUG
		for(int i = 0; i < obs_num; 
			printf("%LF/%LF\n", *(argsA + i), *(argsB + i)), ++i);
	#endif

	while(0 < obs_num--)
		dis += pow(*(argsA + obs_num) - *(argsB + obs_num), 2);
	dis = pow(dis, 0.5);
	return dis;
};

static void rec_qck(long double *vector, unsigned int *vecaux, int start, int end){
	int i = start, j = end;
	long double pivot = *(vector + (i + j)/2), aux;
	while(j >= i){
		while(i <= end && *(vector + i) < pivot) ++i;
		while(j >= start && *(vector + j) > pivot) --j;
		if (j >= i){
			aux = *(vector + i);
			*(vector + i) = *(vector + j);
			*(vector + j) = aux;

			aux = *(vecaux + i);
			*(vecaux + i) = *(vecaux + j);
			*(vecaux + j) = aux;

			++i;
			--j;
		}
	}
	if (i < end)
		rec_qck(vector, vecaux, i, end);
	if (j > start)
		rec_qck(vector, vecaux, start, j);
};

void qckSort(long double *vector, unsigned int *aux, size_t size){
	rec_qck(vector, aux, 0, size - 1);
};

long double **get_data(FILE *fdata, size_t *dt_size, unsigned int obs_num){
	long double **dt_matrix = NULL;
	while(!feof(fdata)){
		dt_matrix = realloc(dt_matrix, sizeof(long double *) * (1 + *dt_size));
		*(*dt_size + dt_matrix) = malloc(sizeof(long double) * obs_num);
		for(unsigned int i = obs_num; i > 0; --i)
			fscanf(fdata, "%LF", (*(*dt_size + dt_matrix) + i - 1));
		++(*dt_size);
	};
	return dt_matrix;
};

#ifdef DEBUG
	void print_data(long double **data, unsigned int rows, unsigned int obs_num){
		for(unsigned int k = 0; k < rows; printf("\n"), ++k)
			for(unsigned int l = 0; l < obs_num; 
				printf("%LF ", *(*(data + k) + l)), ++l);
	}
#endif

void destroy_data(long double **data, unsigned int rows){
	while(0 < rows--)
		free(*(data + rows));
	free(data);
};

void knn (unsigned int K, FILE *finput, FILE *fdata, unsigned int obs_num){
	size_t dt_size = 0;
	
	long double **dt_matrix = get_data(fdata, &dt_size, obs_num), 
	*distances = malloc(sizeof(long double) * dt_size),
	*obs = malloc(sizeof(long double) * obs_num),
	sum = 0;

	unsigned int *indeces = malloc(sizeof(unsigned int) * dt_size), i;

	while(!feof(finput)){
		for(i = 0; i < dt_size; *(indeces + i) = i, ++i);
		//Load all data from fdata in a matrix k
		//begin repeat until feof(finput)k
		//Get one single case from finput k
		//calculate all euclid_dis from this case to the data from fdata and store in a vector
		//Sort the vector
		//Get the K first results and do a mean
		//Print mean and repeat.
		//print_data(dt_matrix, dt_size, obs_num);
		for(i = obs_num; i > 0; --i)
			fscanf(finput, "%LF", (obs + i - 1));
		for(i = dt_size; i > 0; 
			*(distances + i - 1) = euclid_dis(obs_num, obs, *(dt_matrix + i - 1)), 
			--i);
		qckSort(distances, indeces, dt_size);
		printf("group(s): ");
		for(i = 0; i < K; printf("[%u]", *(indeces + i)), sum += *(distances + i++));
		printf("\t(total distance: %LF)\n", sum/(1.0 * K));
		sum = 0;
	};
	free(obs);
	free(indeces);
	free(distances);
	destroy_data(dt_matrix, dt_size);
};

int main(int argc, char const *argv[]){
	extern int errno;
	if (argc == ARG_NUM && OBS_NUM > 0 && K_PARAM > 0){
		FILE *finput = fopen(*(TEST_NAME + argv), "r");
		if (finput != NULL){
			FILE *fdata = fopen(*(DATA_NAME + argv), "r");
			if (fdata != NULL){
				knn(atol(*(K_PARAM + argv)),finput, fdata, atol(argv[OBS_NUM]));
				fclose(finput);
				fclose(fdata);
				return 0;
			}
			fclose(finput);
			printf("E#%hd with \"%s\": %s\n", errno, *(DATA_NAME + argv), strerror(errno));
			return errno;
		}

		printf("E#%hd with \"%s\": %s\n", errno, *(TEST_NAME + argv), strerror(errno));
		return errno;
	};

	printf("usage: %s <K> <obs num> <test path> <data path>\n", *(PROG_NAME + argv));
	return 1;
};