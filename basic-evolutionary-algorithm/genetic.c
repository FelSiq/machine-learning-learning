#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define SWAP_LF(A,B) {double C = (A); (A) = (B); (B) = C;}
#define MODULUS(A) ((A) < 0) ? (-(A)) : (A)

int *polinit(int order){
	int *polcoefs = malloc(sizeof(int) * (1 + order));
	if (NULL != polcoefs){
		char c = 97;
		for(int k = 0; k <= order; 
			printf("type '%c' (%d/%d) coeficient: ", c++, k + 1, order + 1), 
			scanf("%d%*c", (polcoefs + k++)));
	};
	return polcoefs;
};

void printvec(double const *vec, size_t size){
	for (register size_t i = 0; i < size; printf("%lf ", *(vec + i++)));
	printf("\n");
};

double *resinit(size_t popnum){
	double *results = malloc(sizeof(double) * popnum);
	#ifdef DEBUG
		size_t k = popnum;
	#endif
	if (NULL != results)
		while(0 < popnum--)
			*(results + popnum) = pow(-1, popnum) * (double) (rand() % 2000);
	#ifdef DEBUG
		printvec(results, k);
	#endif
	return results;
};

static void qsort_rec(double *newresults, double *newgen, int start, int end){
	double pivot = (*(newresults + start) + *(newresults + end)) * 0.5;
	int i = start, j = end;
	while(i <= j){
		while (i <= end && *(newresults + i) < pivot) i++;
		while (j >= start && *(newresults + j) > pivot) j--;
		if (i <= j){
			SWAP_LF(*(newresults + i), *(newresults + j));
			SWAP_LF(*(newgen + i), *(newgen + j));
			i++;
			j--;
		};
	};
	if (i < end)
		qsort_rec(newresults, newgen, i, end);
	if (start < j)
		qsort_rec(newresults, newgen, start, j);
};

void qcksort(double *newresults, double *newgen, int popnum){
	qsort_rec(newresults, newgen, 0, popnum - 1);
};

void crossover (int const *polcoefs, int order, double *results, int popnum){
	/*
		The crossover section generates exactly newgen_size - 1 (constant value) new samples. 
		The additional + 1 is justified by the use of elitism property.
	*/

	size_t const newgen_size = 1 + (popnum * (popnum - 1) * 0.5);
	double 	*newgen = malloc(sizeof(double) * newgen_size),
			*newresults = malloc(sizeof(double) * newgen_size);

	//Crossover
	int counter = 0;
	for(int i = 0; i < (popnum - 1); ++i){
		for(register int j = (i + 1); j < popnum; ++j){
			//There's no crossover without a couple, i != j.
			*(newgen + counter++) = (*(results + i) + *(results + j)) * 0.5;
		};
	};
	//The best result of previous generation is considered aswell (elitism).
	*(newgen + newgen_size - 1) = *results;

	//Calculates the smallest error
	double error = 0;
	for(register int j = 0; j < order; ++j)
		error += (pow(*results, (order - j))) * (*(polcoefs + j));
	error += *(polcoefs + order);

	//Calculate de polynomial values, with mutations, and store in newresults vector
	for(register int i = 0; i < newgen_size; ++i){
		//Sum of all polynomial terms except the independent term
		*(newresults + i) = 0;
		for(register int j = 0; j < order; ++j)
			*(newresults + i) += (pow(*(newgen + i), (order - j))) * (*(polcoefs + j));
		//Add the independent term
		*(newresults + i) += *(polcoefs + order);
		//Add mutation
		if (i % popnum == 0)
			*(newresults + i) += ((rand() % 3) - 1) * sqrt(error);
		//Modularize (because we want as next as possible to 0)
		*(newresults + i) = MODULUS(*(newresults + i));
	};
	//Sort the newresults vector alongside its correspondents samples
	qcksort(newresults, newgen, popnum);
	/*
		Get the popnum first samples of newgen that generates the "as close to 0 as possible" results, 
		and update de result vector.	
	*/

	for(register int i = 0; i < popnum; 
		*(results + i) = *(newgen + i), ++i);

	free(newresults);
	free(newgen);
};

int main(int argc, char const *argv[]){
	//Set random seed to initialize results
	srand(time(NULL));
	//Get all parameters
	printf("type de polynomial order (0,10], please:\n");
	int order = 0;
	scanf("%d%*c", &order);

	if (order <= 0 || order > 10){
		printf("invalid order, abort.\n");
		exit(1);
	};

	int *polcoefs = polinit(order);

	printf("type the number of the initial population:\n");
	int popnum = 0;
	scanf("%d%*c", &popnum);

	if (popnum <= 1){
		printf("there is must exist at least two exemples on initial population, abort.\n");
		exit(3);
	};

	double *results = resinit(popnum);

	printf("give the number of generations:\n");
	int gennum = 0;
	scanf("%d%*c", &gennum);

	if (gennum <= 0){
		printf("number of generations must be a positive integer, abort.\n");
		exit(2);
	};
	//Loop learning
	while(0 < gennum--){
		crossover(polcoefs, order, results, popnum);
	};

	//Show result
	printvec(results, popnum);

	//Free used memory
	free(results);
	free(polcoefs);
	return 0;
};