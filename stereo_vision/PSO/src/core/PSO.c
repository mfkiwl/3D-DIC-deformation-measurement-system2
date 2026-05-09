// run PSO & DIC
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "DIC_comm.h"
#include "PSO_comm.h"
#include "system_comm.h"

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
