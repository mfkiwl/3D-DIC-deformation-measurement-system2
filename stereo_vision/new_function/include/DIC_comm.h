#ifndef DIC_COMM_H
#define DIC_COMM_H
#include <stdlib.h>
#include <stdint.h>
#include "include/PSO_comm.h"

#define square(x) ((x)*(x))

typedef struct DIC_config{
    uint16_t subset_side_len;
    uint16_t scan_side_len;
} DIC_config;

typedef struct DIC_context {
    struct DIC_config config;
    int *img_ref_pt_pos;
    void *priv;
} DIC_context;

typedef struct DIC_subset_info {
    float *subset_data;
    float mean;
    int side_len;
    int total_pixels;
} DIC_subset_info;

typedef struct DIC_ZNCC_context {
    struct DIC_subset_info ref_subset_info;
    struct DIC_subset_info cur_subset_info;
    float subset_pt_cur_pos[2];                 // Pi_x, Pi_y relative correct coordinate
    // float subset_pt_ref_pos[2];
    int img_pt_ref_pos[2];                      // object point
    // int *IMG_aft;
    // int *IMG_bef;
    // int IMG_pt_cur_pos[2];
    // int IMG_width;
    // int IMG_height;
    // int corr_coef;
} DIC_ZNCC_context;

typedef struct DIC_SSD_context {
    struct DIC_subset_info ref_subset_info;
    struct DIC_subset_info cur_subset_info;
    float subset_pt_cur_pos[2]; 
    int img_pt_ref_pos[2]; 
}DIC_SSD_context;

struct PSO_cost_func_ops {
    int (*init)(SYS_INFO *info);
    float (*cost_function)(PSO_context *ctx);
    int (*cleanup)(PSO_context *ctx);
};

float get_subset_mean(const DIC_subset_info *subset);
void ZNCC_ctx_init(struct SYS_INFO *info);
int ZNCC_subset_alloc(struct DIC_ZNCC_context *zncc_ctx);
int ZNCC_init(struct SYS_INFO *info);
float ZNCC_cost_function(struct PSO_context *ctx) ;
int ZNCC_cleanup(struct PSO_context *ctx);
int SSD_init(struct SYS_INFO *info);
float SSD_cost_function(struct PSO_context *ctx);
int SSD_cleanup(struct PSO_context *ctx);

extern void ZNCC_ctx_init(struct SYS_INFO *info);
extern struct PSO_cost_func_ops zncc_ops;
extern struct PSO_cost_func_ops ssd_ops;

/*
private 為特定任務中所需的資料 (pointer: 8bytes)
*/

#endif
