#include "include/PSO_comm.h"
#include "ZNCC_core.c"

void pso_context_init(struct PSO_context *ctx) {
    if (ctx == NULL) return;

    ctx->config.population =            100;
    ctx->config.dimension  =            2;      
    ctx->config.max_iter   =            50;
    
    ctx->config.v_max      =            10;
    ctx->config.v_min      =            -10;
    ctx->config.w_max      =            0.9f;    
    ctx->config.w_min      =            0.4f;

    ctx->config.c1         =            2.0f;
    ctx->config.c2         =            2.0f;

    ctx->config.dec_factor =            1.0f;
    ctx->config.inc_factor =            1.0f;

    // 初始化內部的 layout
    // pso_layout_init(&ctx->config.layout); 
}

void st_update_pt_vel_algorithm(struct PSO_context *ctx){
    // standard pso
}

struct PSO_algorithm_ops standard_pso_ops = {
    .update_pt_vel = st_update_pt_vel_algorithm,
    .cost_function = ZNCC_cost_function
};


/*
用法:
struct PSO_algorithm_ops zncc_ops = { .fitness_function = zncc_calc };
struct PSO_algorithm_ops ssd_ops  = { .fitness_function = ssd_calc };

// 根據使用者設定切換
if (user_choice == USE_ZNCC) {
    ctx.algo_ops = &zncc_ops;
} else {
    ctx.algo_ops = &ssd_ops;
}

// 核心演算法永遠只呼叫這一行，它會自動跑正確的演算法
float score = ctx.algo_ops->fitness_function(&ctx, current_pos);
*/