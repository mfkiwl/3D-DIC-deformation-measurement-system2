#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "include/DIC_comm.h"
#include "include/PSO_comm.h"
#include "include/image_comm.h"

#define SYS_FAILURE               -1
#define SYS_SUCCESS               1
#define SYS_DBG_EN                1

#define square(x) ((x)*(x))
#define mean(x) ((x)/(DIC_SUBSET_SIDE_LEN*DIC_SUBSET_SIDE_LEN))

#define IMG_WIDTH_640                           640
#define IMG_HEIGHT_480                          480 
#define IMG_WIDTH_1280                          1280
#define IMG_HEIGHT_720                          720 
#define IMG_WIDTH_1920                          1920
#define IMG_HEIGHT_1080                         1080 



#define DIC_SUBSET_SIDE_LEN                        31
#define DIC_SUBSET_SIDE_LEN_HALF (DIC_SUBSET_SIDE_LEN-1)/2 
#define DIC_SCAN_SIDE_LEN 31

#define PSO_POPULATION 20 
#define PSO_DIMENSION 2
#define PSO_ITERATION 4
#define PSO_ITERATION_REC (1.0/PSO_ITERATION)
#define PSO_MATRIX_VER_POINTS 3
#define PSO_MATRIX_HOR_POINTS 3
#define PSO_MATRIX_ALL_POINTS PSO_MATRIX_VER_POINTS*PSO_MATRIX_HOR_POINTS
#define PSO_MATRIX_SIDE_LEN DIC_SCAN_SIDE_LEN/2
#define PSO_MATRIX_INTERVAL (PSO_MATRIX_SIDE_LEN)/(PSO_MATRIX_VER_POINTS-1) 
#define PSO_MATRIX_SIDE_LEN_HALF PSO_MATRIX_INTERVAL*(PSO_MATRIX_VER_POINTS-1)/2
#define PSO_BOUND_LEN 0.5*(DIC_SCAN_SIDE_LEN-1)
#define PSO_V_MAX 0.5*PSO_BOUND_LEN
#define PSO_V_INI 0.2*PSO_V_MAX
#define PSO_W_UPPER 0.9 
#define PSO_W_LOWER 0.4
#define PSO_DECREASE_FACTOR 1 
#define PSO_INCREASE_FACTOR 1.05
#define PSO_COGNITION_FACTOR 1.0 
#define PSO_SOCIAL_FACTOR 1.0 



#if defined(SYS_DBG_EN)
#define SYS_DBG               SYS_PRT
#else
#define SYS_DBG(format, args...)	do {} while (0) // erase: SYS_DBG(<all things you input>)
#endif

#define SYS_PRT(format, ...)	printf("[%s][%d]"format, __func__, __LINE__, ##__VA_ARGS__)
// args... similar to ...

typedef struct {
	int **args_ptr;
	int args_num;
	int args_idx;
} SYS_ARGS_INFO;

typedef struct {
	int dev_num;
	int dev_idx;
	SYS_ARGS_INFO args_info;
	PSO_context pso_ctx;
	DIC_context dic_ctx;
} SYS_INFO;




/*
===== Note =====
args...  表示所有參數: 0~無限
...      等同args...


*/