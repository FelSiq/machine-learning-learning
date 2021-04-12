#include <stdlib.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {
	FILE *fp = fopen("./audio.raw", "r");

	if (fp != NULL) {
		unsigned char byte;

		while (!feof(fp)) {
			if (fread(&byte, 1, sizeof(char), fp))
				printf("%d\n", byte);
		}

		return fclose(fp);
	}
	printf("e: can't read \"audio.raw\" file.\n");
	return 1;
}