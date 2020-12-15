#include <stdlib.h>
#include <stdio.h>
#include <time.h>

enum{
	PROGNAME,
	NUMCOL,
	NUMROW,
	MAXVALS,
	NUMARGS
};

int main(int argc, char const *argv[]){
	srand(time(NULL));
	if (argc == NUMARGS){
		int numCol = atol(argv[NUMCOL]);
		int numRow = atol(argv[NUMROW]);
		int maxVal = atol(argv[MAXVALS]);
		size_t register j;
		for (size_t i = 0; i < numRow; ++i){
			for (j = 0; j < numCol - 1; ++j)
				printf("%d ", rand() % maxVal);
			printf("%d\n", rand() % 2);
		}
		return 0;
	}
	printf("usage: %s <# of cols> <# of rows> <# of classes>\n", argv[PROGNAME]);
	return 1;
};