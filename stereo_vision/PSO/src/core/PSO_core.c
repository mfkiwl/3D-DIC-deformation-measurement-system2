#include "include/system_comm.h"
#include "include/PSO_comm.h"
#include <float.h>

double double_rand_nor(void) {
    return (rand() / (double)RAND_MAX);
}

static inline void clamp_to_bounds(double *value, double max, double min) {
	if (!value) return;
	if (*value > max) *value = max;
	else if (*value < min) *value = min;
}

int execute_pso_algorithm(struct PSO_context *ctx) {
	// 1. check
	if (!ctx || !ctx->algo_ops || !ctx->algo_ops->run) {
		return PSO_FAILURE;
	}
	// 2. initial (optional)
	if (ctx->algo_ops->init) {
		if (ctx->algo_ops->init(ctx) != 0) {
			SYS_DBG("algo init failed!\n");
			return PSO_FAILURE;
		}
	}
	// 3. run (necessary)
	int run_ret = ctx->algo_ops->run(ctx);
	if (run_ret != 0) {
		SYS_DBG("algo run failed!\n");
		return PSO_FAILURE;
	}
	// 4. clean (optional)
	if (ctx->algo_ops->cleanup) {
		int clean_ret = ctx->algo_ops->cleanup(ctx);
		if (clean_ret != 0) {
			SYS_DBG("algo cleanup failed!\n");
			return PSO_FAILURE;
		}
	}
	return PSO_SUCCESS;
}

void st_pso_config_init(struct PSO_context *ctx) {
    // ctx->config.population 			= 		50; // depends on input
    ctx->config.dimension 				= 		2;
    ctx->config.max_iter 				= 		10; 

    ctx->config.v_max 					= 		4;
    ctx->config.v_min 					= 		-4;
	ctx->config.v_ini					=		1;

    ctx->config.w_max 					= 		0.9f;
    ctx->config.w_min 					= 		0.4f;

    ctx->config.c1 						= 		2.0f;
    ctx->config.c2 						=		2.0f;

    ctx->config.dec_factor 				= 		1.0f;
    ctx->config.inc_factor 				=		1.0f;

    ctx->config.border_side_len 		= 		21; 		// PSO MAX displacement (pixel)
	ctx->global_best.value				=		-FLT_MAX;
}

void st_pso_layout_init(struct PSO_context *ctx) {
	// init st_pso_config_init() first
	struct DIC_ZNCC_context *zncc_ctx = (struct DIC_ZNCC_context *)ctx->priv;
	if (zncc_ctx == NULL) return;
    ctx->config.layout.side_len_pt 			= 		5;
    ctx->config.layout.grid_spacing 		= 		(ctx->config.border_side_len - 1) / (ctx->config.layout.side_len_pt - 1);
    ctx->config.layout.fixed_particles 		= 		ctx->config.layout.side_len_pt * ctx->config.layout.side_len_pt;
	ctx->config.layout.shift_side_len_half 	= 		((ctx->config.layout.side_len_pt-1)/2) * ctx->config.layout.grid_spacing;
}

int st_pso_memory_alloc(struct PSO_context *ctx) {
	ctx->particle = malloc(sizeof(struct PSO_particle)*ctx->config.population);
	if (!ctx->particle) { 
		SYS_DBG("malloc particle failed!!\n");
		return -1;
	}
	return 0;
}

struct PSO_algorithm_ops standard_pso_algo_ops = {
    .init           = 	st_pso_algo_init,
    .run  			= 	st_pso_algo_run,
	.cleanup		= 	st_pso_algo_cleanup
};

int st_pso_algo_init(struct PSO_context *ctx) {
	SYS_DBG("st_pso_algo_init\n");
    if (ctx == NULL) return -1;
	st_pso_config_init(ctx);
	st_pso_layout_init(ctx);
	if (st_pso_memory_alloc(ctx) != 0) {
		SYS_DBG("Allocating memory failed!!\n");
		return -1;
	}
	if (ctx->config.population < ctx->config.layout.fixed_particles) {
		SYS_DBG("population < fixed_particles!\n");
		return -1;
	}
	return 0;
}

int st_pso_algo_run(struct PSO_context *ctx) {
	SYS_DBG("st_pso_algo_run\n");
	if (ctx == NULL) return -1;
	double c1 = ctx->config.c1;
	double c2 = ctx->config.c2;
	double border_side_len = ctx->config.border_side_len;
	double border_side_len_half = 0.5 * (border_side_len - 1);
	struct DIC_ZNCC_context *zncc_ctx = (struct DIC_ZNCC_context *)ctx->priv;
	struct PSO_particle *particles = ctx->particle;
	double spacing = ctx->config.layout.grid_spacing;
	double half_len = (double)ctx->config.layout.shift_side_len_half;
	// initialize random var (only do one time)
	init_random_seed();
	// init setup
	for (int p_idx = 0; p_idx < ctx->config.population; p_idx++) {
		// initialize fixed points
		if (p_idx < (ctx->config.layout.fixed_particles)) {
			double x_pos = (double)(p_idx % ctx->config.layout.side_len_pt) * spacing - (double)half_len;
			double y_pos = (double)(p_idx / ctx->config.layout.side_len_pt) * spacing - (double)half_len;
			particles[p_idx].position[1] = x_pos;
			particles[p_idx].position[0] = y_pos;
			for (int dim=0; dim<ctx->config.dimension; dim++) {
				particles[p_idx].velocity[dim] = ctx->config.v_ini*(double_rand_nor());
				particles[p_idx].ind_best_pos[dim] = particles[p_idx].position[dim];
			}
			zncc_ctx->subset_pt_cur_pos[0] = particles[p_idx].position[0]; // update pos for cost_function
    		zncc_ctx->subset_pt_cur_pos[1] = particles[p_idx].position[1]; // update pos for cost_function
			double cost = ctx->cost_ops->cost_function(ctx);
			SYS_DBG("cost = %.4f\n", cost);
			particles[p_idx].ind_best_value = cost;
			if (cost > ctx->global_best.value) {
				ctx->global_best.value = cost;
				ctx->global_best.index = p_idx;
				for (int dim=0; dim<ctx->config.dimension; dim++) {
					ctx->global_best.position[dim] = particles[p_idx].position[dim];
				}
			}
		} 
		// initialize random points
		else {
			for (int dim=0; dim<ctx->config.dimension; dim++) {
				particles[p_idx].velocity[dim] = ctx->config.v_ini*(double_rand_nor()); // should initialize in .init
				particles[p_idx].position[dim] = border_side_len_half * (double_rand_nor());
				particles[p_idx].ind_best_pos[dim] = particles[p_idx].position[dim];
			}
			zncc_ctx->subset_pt_cur_pos[0] = particles[p_idx].position[0]; // update pos for cost_function
    		zncc_ctx->subset_pt_cur_pos[1] = particles[p_idx].position[1]; // update pos for cost_function
			double cost = ctx->cost_ops->cost_function(ctx);
			particles[p_idx].ind_best_value = cost;
			if (cost > ctx->global_best.value) {
				ctx->global_best.value = cost;
				ctx->global_best.index = p_idx;
				for (int dim=0; dim<ctx->config.dimension; dim++) {
					ctx->global_best.position[dim] = particles[p_idx].position[dim];
				}
			}
		}
	}
	// iteration strat
	for (int iter = 0; iter < ctx->config.max_iter; iter++) {
		for (int p_idx = 0; p_idx < ctx->config.population; p_idx++) {
			for (int dim = 0; dim<ctx->config.dimension; dim++) {
				/* update pt*/
				double weight = ctx->config.w_max - ((double)(iter + 1) / ctx->config.max_iter)*(ctx->config.w_max - ctx->config.w_min);
				particles[p_idx].velocity[dim] = weight * particles[p_idx].velocity[dim] +\
												 c1 * (double_rand_nor()) * (particles[p_idx].ind_best_pos[dim] - particles[p_idx].position[dim]) +\
												 c2 * (double_rand_nor()) * (ctx->global_best.position[dim] - particles[p_idx].position[dim]);
				
				clamp_to_bounds(&particles[p_idx].velocity[dim], ctx->config.v_max, ctx->config.v_min);
				particles[p_idx].position[dim] += particles[p_idx].velocity[dim];
				clamp_to_bounds(&particles[p_idx].position[dim], border_side_len_half, -border_side_len_half);
			}
			zncc_ctx->subset_pt_cur_pos[0] = particles[p_idx].position[0]; // update pos for cost_function
    		zncc_ctx->subset_pt_cur_pos[1] = particles[p_idx].position[1]; // update pos for cost_function
			double cost = ctx->cost_ops->cost_function(ctx);

			if (cost > particles[p_idx].ind_best_value) {
				particles[p_idx].ind_best_value = cost;
				for (int dim = 0; dim<ctx->config.dimension; dim++) {
					particles[p_idx].ind_best_pos[dim] = particles[p_idx].position[dim];
				}
				// Global best
				if (cost > ctx->global_best.value) {
					ctx->global_best.value = cost;
					ctx->global_best.index = p_idx;
					for (int dim = 0; dim<ctx->config.dimension; dim++) {
						ctx->global_best.position[dim] = particles[p_idx].position[dim];
					}
				}
			}
		}
	}
	// Result is in ctx->global_best
	SYS_DBG("[INFO] PSO finished!\n");
	return SYS_SUCCESS;
}

int st_pso_algo_cleanup(struct PSO_context *ctx) {
	SYS_DBG("st_pso_algo_cleanup\n");
	if (ctx == NULL) return -1;
	if (ctx->particle) {
		free(ctx->particle);
		ctx->particle = NULL;
	}
	return SYS_SUCCESS;
}
