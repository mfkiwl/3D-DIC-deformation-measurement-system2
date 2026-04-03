#include <stdio.h>
#include <stdint.h>
#include "include/image_comm.h"

struct PSO_LayoutConfig{
    int rows;                       // matrix_ver_point
    int cols;                       // matrix_hor_point
    int grid_spacing;               // matrix_side_len
    int side_length;                // matrix_interval

    //int matrix_all_point;
    //int matrix_side_len_half;
};

struct PSO_config{
    int population;
    int dimension;
    int max_iter;
    // float iteration_rec; 

    int boundary_size;              
    int v_max;                      // velocity_max
    int v_min;                      // velocity_min
    float w_max;                    // w_upper
    float w_min;                    // w_lower

    float c1;                       // cognition_factor
    float c2;                       // social_factor

    float dec_factor;                 // Decrease_factor
    float inc_factor;                 // Increase_factor
    struct PSO_LayoutConfig layout;
};

struct PSO_particle {
    float *position;
    float *velocity;
    float *best_position;
};

struct PSO_algorithm_ops {
    void (*update_pt_vel)(PSO_context *ctx);
    void (*cost_function)(PSO_context *ctx, int *pos);
};

typedef struct PSO_context {
    struct PSO_config config;
    struct PSO_algorithm_ops *algo_ops;
    struct PSO_particle *particle;
    struct IMG_info img_cfg;
    void *priv; // private 為特定任務中所需的資料 (pointer: 8bytes)
}PSO_context;




/*
Note:
使用指標可能使快取命中率下降

*/