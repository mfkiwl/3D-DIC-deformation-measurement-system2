// run PSO & DIC
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "include/DIC_comm.h"
#include "include/PSO_comm.h"
#include "include/system_comm.h"

int main(int argc, char *argv[]) {
    return 0;
}

__declspec(dllexport)
void process_image(double *ref_img, double *cur_img, int width, int height,\
                     int population, int subset_side_len,\
                     double *img_ref_pt, double *img_cur_pt, double *result){
    if (ref_img == NULL || cur_img == NULL) return;
    struct SYS_INFO *info = SYS_create();
    if (!info) return;
    
    // initial image info
    info->pso_ctx.img_info.ref_data         = ref_img;
    info->pso_ctx.img_info.cur_data         = cur_img;
    info->pso_ctx.img_info.width            = width;
    info->pso_ctx.img_info.height           = height;
    info->pso_ctx.config.population         = population;
    info->dic_ctx.config.subset_side_len    = subset_side_len;
    info->dic_ctx.img_ref_pt_pos            = img_ref_pt;
    info->dic_ctx.img_cur_pt_pos            = img_cur_pt;

    // assign algo type
    info->pso_ctx.algo_ops                  = &standard_pso_algo_ops;

    if (info) {
        run_PSO(info);
        SYS_DBG("global_best_pos: (x,y) = (%.2f,%.2f)\n", info->pso_ctx.global_best.position[1], info->pso_ctx.global_best.position[0]);
        result[0] = info->pso_ctx.global_best.position[0]; // y
        result[1] = info->pso_ctx.global_best.position[1]; // x
        result[2] = info->pso_ctx.global_best.value; // coef
        SYS_clean(info);
    }
}


// =====
    // assign parameters
    // printf("Please input particle number: ");
    // scanf("%d", &(info->pso_ctx.config.population));
    // printf("Please input subset_side_len: ");
    // scanf("%d", &info->dic_ctx.config.subset_side_len);

    // // initial image info
    // info->pso_ctx.img_info.ref_data     = ref_img;
    // info->pso_ctx.img_info.cur_data     = cur_img;
    // info->pso_ctx.img_info.width        = width;
    // info->pso_ctx.img_info.height       = height;

// ZNCC_ctx_init(info);
// info->pso_ctx.priv = zncc_ctx;

// struct PSO_context *pso_ctx_ptr = PSO_create_factory(PSO_TYPE_ZNCC, info);

// // run
// if (execute_pso_algorithm(pso_ctx_ptr) != 0) {
//     SYS_DBG("execute_pso_algorithm failed!!\n");
//     free(zncc_ctx);
//     free(info);
//     return SYS_FAILURE;
// }

// // result
// printf("global_best_pos: (x,y) = (%.2f,%.2f)\n", info->pso_ctx.global_best.position[1], info->pso_ctx.global_best.position[0]);

// // free struct
// info->pso_ctx.algo_ops->cleanup(pso_ctx_ptr);
// free(zncc_ctx);
// free(info);

// return SYS_SUCCESS;


/* 
Note:

呼叫順序:
1.建立私有資料 與 通用框架
2.將私有資料放進框架
3.在功能函數中從共通框架取回


stack空間比heap小很多，以空間考量前提下盡量用heap
stack空間不夠容易發生stack overflow


cost_ops->cost_function = zncc_func
總之 以架構來說 只要分配malloc到cost_ops->cost_function這裡
剩下函數zncc_func 要由函數自己內容寫malloc去分配


to do list

導入內聯函數 (inline): 函數前加 static inline
使用「位元欄位」 (Bit-fields) 處理開關 不要int
struct 的順序會影響佔用的記憶體大小
使用 const 與 restrict 關鍵字

*/

// void Print_Args(SYS_INFO *info)
// {
// 	int i = 0;
// 	SYS_DBG("\n");

// 	SYS_DBG("=====================\n");
// 	for (i = 0; i < info->args_info.args_num; i++) {
// 		SYS_DBG("Argument[%02d/%02d]     : %s\n", i, info->args_info.args_num, info->args_info.args_ptr[i]);
// 	}
// 	info->args_info.args_ptr[info->args_info.args_num] = NULL;
// 	SYS_DBG("=====================\n");
// }

