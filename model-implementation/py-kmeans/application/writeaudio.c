#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum {
	FILENAME,
	INPUTPATH,
	ARGNUM
};

int main(int argc, char const *argv[]) {
	if (argc < ARGNUM) {
		printf("usage: %s <input path>\n", argv[FILENAME]);
		return 2;
	}

	char *outname = malloc(sizeof(char) * (5 + strlen(argv[INPUTPATH])));
	strcat(outname, argv[INPUTPATH]); 
	strcat(outname, ".raw"); 

	FILE *fp = fopen(argv[INPUTPATH], "r");
	FILE *fout = fopen(outname, "w");

	if (fp != NULL) {
		unsigned int aux;
		float freq;

		while (!feof(fp)) {
			if (fscanf(fp, "%f", &freq)) {
				aux = (unsigned int) freq;
				fwrite(&aux, 1, sizeof(unsigned char), fout);
			}
		}
		free(outname);
		return fclose(fp);
	}
	
	free(outname);
	printf("e: can't read \"%s\" file.\n", argv[INPUTPATH]);
	return 1;
}