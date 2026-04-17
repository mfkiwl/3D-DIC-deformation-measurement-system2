#ifndef PSO_comm_H
#define PSO_comm_H

#include <stdio.h>
#include <stdint.h>
#include "include/image_comm.h"
#include "include/DIC_comm.h"

#define PSO_FAILURE         -1
#define PSO_SUCCESS         0

typedef struct PSO_context PSO_context;
typedef struct SYS_INFO SYS_INFO;     // 告訴編譯器，這隻大白鯊晚點會出現
typedef struct PSO_context PSO_context;

typedef enum {
    VELOCITY,
	POSITION,
} PSO_BOUND_CHECK_TYPE;

typedef enum {
    PSO_TYPE_ZNCC,
    PSO_TYPE_SSD
} PSO_COST_TYPE;

struct PSO_global_best {
    double position[2]; // x, y
    double value;
    int index;
};

struct PSO_LayoutConfig{
    int side_len_pt;                    // matrix_side_len_points
    double grid_spacing;                 // matrix_interval_pixel
    int fixed_particles;                // inital all fixed particles
    int shift_side_len_half;                    // correct the y corridate of point
    //int side_len;                     // matrix_side_len (pixel) (from priv)
    //int matrix_all_point;
    //int matrix_side_len_half;
};

struct PSO_config{
    struct PSO_LayoutConfig layout;
    int population;
    int dimension;
    int max_iter;
    // double iteration_rec; 
    int v_max;                        // velocity_max
    int v_min;                        // velocity_min
    int v_ini;                        // velocity_initial
    double w_max;                      // w_upper
    double w_min;                      // w_lower
    double c1;                         // cognition_factor
    double c2;                         // social_factor
    double dec_factor;                 // Decrease_factor
    double inc_factor;                 // Increase_factor
    double border_side_len;            // border length 
};

struct PSO_particle {
    double position[2];
    double velocity[2];
    double ind_best_pos[2];
    double ind_best_value;
};

typedef struct PSO_context {
    struct PSO_config config;
    struct IMG_info img_info;
    struct PSO_algorithm_ops *algo_ops;     // 與priv不同 填入特定的ops: 表示:怎麼做 (指標指向靜態Struct)
    struct PSO_cost_func_ops *cost_ops;
    struct PSO_particle *particle;
    struct PSO_global_best global_best; 
    void *priv;                             // private 為特定任務中所需的"資料" 表示:用什麼做
}PSO_context;

struct PSO_algorithm_ops {
    int (*init)(PSO_context *ctx);
    int (*run)(PSO_context *ctx);
    int (*cleanup)(PSO_context *ctx);
};

// declare implementation
void run_PSO(struct SYS_INFO *info);
void PSO_create_factory(PSO_COST_TYPE type, struct SYS_INFO *info);
int PSO_destroy(struct PSO_context *pso_ctx);
int execute_pso_algorithm(struct PSO_context *ctx);
int st_pso_algo_init(struct PSO_context *ctx);
int st_pso_algo_run(struct PSO_context *ctx);
int st_pso_algo_cleanup(struct PSO_context *ctx);
double double_rand_nor(void);

extern struct PSO_cost_func_ops zncc_ops;
extern struct PSO_cost_func_ops ssd_ops;
extern struct PSO_algorithm_ops standard_pso_algo_ops;

#endif

