#include <stdlib.h>
#include "include/image_comm.h"

float random_digit(void)
{
    return rand() / (float)RAND_MAX;
}

void init_random_seed() {
    static int seeded = 0;
    if (!seeded) {
        time_t t = time(NULL);
        srand((unsigned) t);
        seeded = 1;
    }
}

void print_array(int *array){
    if (array==NULL): return;
    int len = sizeof(array)/sizeof(array[0])
	for (int i = 0; i < Size; i++)
	{	
		for (int j = 0; j < Size; j++)
		{
			printf("%d ",array[i][j]);
		}
		printf("\n");
	}
	return;
}

void image_init(IMG_info IMG_info){


}
