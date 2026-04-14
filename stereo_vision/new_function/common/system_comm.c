#include <stdlib.h>
#include <time.h>
#include "include/image_comm.h"
#include "include/DIC_comm.h"

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

float get_mean(float sum, int side_len) {
    return sum/(float)((side_len)*(side_len)); // return type: float
}

/*
傳入陣列到函式其他寫法:
void func(int (*arr)[4])

don't add static to function (不同檔案間傳遞無法使用)

2. 如果想跨檔案使用 inline
必須寫在 .h 檔：你必須把整個函式的「內容」都寫在標頭檔裡，並加上 static inline。

*/

// void print_array(int *array){
//     if (array==NULL) return;
// 	for (int i = 0; i < Size; i++)
// 	{	
// 		for (int j = 0; j < Size; j++)
// 		{
// 			printf("%d ",array[i][j]);
// 		}
// 		printf("\n");
// 	}
// 	return;
// }