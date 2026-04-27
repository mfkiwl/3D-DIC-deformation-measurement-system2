#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "include/system_comm.h"
#include "include/DIC_comm.h"
#include "include/PSO_comm.h"

extern int interp_safe_margin();
extern double bilinear(double *data, int width, int height, double x, double y);

void run_PSO(struct SYS_INFO *info) {
	// create PSO_context and put DIC info in it
	PSO_create_factory(PSO_TYPE_ZNCC, info);

	if (!info->pso_ctx.priv) return;

    if (execute_pso_algorithm(&info->pso_ctx) != 0) {
        SYS_DBG("execute_pso_algorithm failed!!\n");
        return;
    }
}

double get_subset_mean(const DIC_subset_info *subset) {
	if (!subset) return NAN;
    double sum = 0.0f;
    for (int i = 0; i < subset->total_pixels; i++) {
        sum += (double)subset->subset_data[i];
    }
    return sum/(subset->total_pixels);
}

static void get_ref_subset_data(struct SYS_INFO *info) {
	struct DIC_ZNCC_context *zncc_ctx = info->pso_ctx.priv;
	double *pos_ptr 				= info->dic_ctx.img_ref_pt_pos;
	double *ref_img_data 			= info->pso_ctx.img_info.ref_data;
	int subset_side_len				= info->dic_ctx.config.subset_side_len;
	int img_width 					= info->pso_ctx.img_info.width;
	int img_height 					= info->pso_ctx.img_info.height;
	double img_pt_y_shift 			= pos_ptr[0] - (subset_side_len - 1) / 2.0f;
	double img_pt_x_shift 			= pos_ptr[1] - (subset_side_len - 1) / 2.0f;
	double *ref_subset_data 		= zncc_ctx->ref_subset_info.subset_data;
	for (int row = 0; row < subset_side_len; row++) {
		for (int col = 0; col < subset_side_len; col++) {
			double img_pt_y_shift_row = img_pt_y_shift + row;
			double img_pt_x_shift_col = img_pt_x_shift + col;
			int pixel_idx = row * subset_side_len + col;
			ref_subset_data[pixel_idx] = bilinear(ref_img_data, img_width, img_height,
												img_pt_x_shift_col, img_pt_y_shift_row);
		}
	}
	zncc_ctx->ref_subset_info.mean = get_subset_mean(&zncc_ctx->ref_subset_info);
}

void ZNCC_ctx_init(struct SYS_INFO *info) {
	if (!info) return;
	struct DIC_ZNCC_context *zncc_ctx = info->pso_ctx.priv;
	if (zncc_ctx == NULL) return;
	zncc_ctx->ref_subset_info.side_len 		= 	info->dic_ctx.config.subset_side_len;
	zncc_ctx->cur_subset_info.side_len 		= 	info->dic_ctx.config.subset_side_len;
	zncc_ctx->img_pt_ref_pos[0]				=	info->dic_ctx.img_ref_pt_pos[0];
	zncc_ctx->img_pt_ref_pos[1]				=	info->dic_ctx.img_ref_pt_pos[1];
	zncc_ctx->img_pt_cur_pos[0]				=	info->dic_ctx.img_cur_pt_pos[0];
	zncc_ctx->img_pt_cur_pos[1]				=	info->dic_ctx.img_cur_pt_pos[1];
	zncc_ctx->cur_subset_info.subset_data 	= 	NULL; // calculate in runtime		
	zncc_ctx->ref_subset_info.total_pixels 	= 	square(zncc_ctx->ref_subset_info.side_len);
	zncc_ctx->cur_subset_info.total_pixels 	= 	square(zncc_ctx->cur_subset_info.side_len);
}

int ZNCC_subset_alloc(struct DIC_ZNCC_context *zncc_ctx) {
	int subset_memory_size = square(zncc_ctx->cur_subset_info.side_len) * sizeof(double);
	zncc_ctx->cur_subset_info.subset_data = malloc(subset_memory_size);
	zncc_ctx->ref_subset_info.subset_data = malloc(subset_memory_size);
	if (!zncc_ctx->cur_subset_info.subset_data || !zncc_ctx->ref_subset_info.subset_data) { 
		SYS_DBG("Malloc subset_data failed!!\n");
		free(zncc_ctx->cur_subset_info.subset_data);
    	free(zncc_ctx->ref_subset_info.subset_data);
		return -1;
	}
	return 0;
}

struct PSO_cost_func_ops zncc_ops = {
	.init			= ZNCC_init,
    .cost_function  = ZNCC_cost_function,
	.cleanup		= ZNCC_cleanup
};

int ZNCC_init(struct SYS_INFO *info) {
	SYS_DBG("ZNCC_init\n");
	if (!info) return -1;
	struct DIC_ZNCC_context *zncc_ctx = info->pso_ctx.priv;
	if (zncc_ctx == NULL) return -1;

	ZNCC_ctx_init(info);

	if (ZNCC_subset_alloc(zncc_ctx) != 0) {
		SYS_DBG("Allocating memory failed!!\n");
		return -1;
	}

	get_ref_subset_data(info); // calculate ref_subset_data and put it into: zncc_ctx->ref_subset_info.subset_data
	
	return 0;
}

double ZNCC_cost_function(struct PSO_context *ctx) {
	// SYS_DBG("ZNCC_cost_function\n");
	if (ctx == NULL) return -10.0f; // result range: -1 ~ +1
	struct DIC_ZNCC_context *zncc_ctx = (struct DIC_ZNCC_context *)ctx->priv;
	double rel_pso_pt_y 				= zncc_ctx->subset_pt_cur_pos[0]; 	// Pi_x, Pi_y relative corrected coordinate
	double rel_pso_pt_x 				= zncc_ctx->subset_pt_cur_pos[1]; 	// Pi_x, Pi_y relative corrected coordinate
	double img_cur_pt_y 				= zncc_ctx->img_pt_cur_pos[0]; 		// start point (fixed in whole PSO process)
	double img_cur_pt_x 				= zncc_ctx->img_pt_cur_pos[1]; 		// start point (fixed in whole PSO process)

	int img_width 						= ctx->img_info.width;
	int img_height 						= ctx->img_info.height;
	double *cur_img_data 				= ctx->img_info.cur_data;
	int subset_side_len 				= zncc_ctx->ref_subset_info.side_len;
	double *ref_subset_data 			= zncc_ctx->ref_subset_info.subset_data;
	double *cur_subset_data 			= zncc_ctx->cur_subset_info.subset_data;

	double img_pso_pt_y 				= img_cur_pt_y + rel_pso_pt_y; 		// pso particle absolute coordinate
	double img_pso_pt_x 				= img_cur_pt_x + rel_pso_pt_x; 		// pso particle absolute coordinate

	double img_pso_pt_y_shift 			= img_pso_pt_y - ((subset_side_len - 1) / 2.0f); // shift to left_top_point, means (0,0) point in subset matrix(n,n)
	double img_pso_pt_x_shift			= img_pso_pt_x - ((subset_side_len - 1) / 2.0f);

	if ((img_pso_pt_y_shift + subset_side_len + interp_safe_margin()) > img_height || (img_pso_pt_y_shift + interp_safe_margin()) < 0) {
		SYS_DBG("shifted PSO particle image pos_y out of bound!\n");
		SYS_DBG("img_pso_pt_y_shift: %.2f\n", img_pso_pt_y_shift);
		return NAN; // result range: -1 ~ +1
	} else if ((img_pso_pt_x_shift + subset_side_len + interp_safe_margin()) > img_width || (img_pso_pt_x_shift + interp_safe_margin()) < 0) {
		SYS_DBG("shifted PSO particle image pos_x out of bound!\n");
		SYS_DBG("img_pso_pt_x_shift: %.2f\n", img_pso_pt_x_shift);
		return NAN; // result range: -1 ~ +1
	}
	
	for (int row = 0; row < subset_side_len; row++) {
		for (int col = 0; col < subset_side_len; col++) {
			double img_pso_pt_y_shift_row = img_pso_pt_y_shift + row;
			double img_pso_pt_x_shift_col = img_pso_pt_x_shift + col;
			int pixel_idx = row * subset_side_len + col;
			cur_subset_data[pixel_idx] = bilinear(cur_img_data, img_width, img_height,
												img_pso_pt_x_shift_col, img_pso_pt_y_shift_row);
		}
	}

	double sum = 0.0f, ref_sum_den = 0.0f, cur_sum_den = 0.0f;
	double cur_subset_mean = get_subset_mean(&(zncc_ctx->cur_subset_info));
	double ref_subset_mean = zncc_ctx->ref_subset_info.mean;
	if (!ref_subset_data) {
		SYS_DBG("Error: ref_subset_data is NULL!\n");
		return NAN;
	}

	for (int row = 0; row < subset_side_len; row++) {
		for (int col = 0; col < subset_side_len; col++) {
			double mean_substrat_ref_subset = ((ref_subset_data[row * subset_side_len + col]) - ref_subset_mean);
			double mean_substrat_cur_subset = ((cur_subset_data[row * subset_side_len + col]) - cur_subset_mean);
			sum += mean_substrat_ref_subset * mean_substrat_cur_subset;
			ref_sum_den += square(mean_substrat_ref_subset);
			cur_sum_den += square(mean_substrat_cur_subset);
		}
	}
	if (ref_sum_den == 0 || cur_sum_den == 0) return NAN;
	return (sum / (sqrt(ref_sum_den * cur_sum_den)));
}

int ZNCC_cleanup(struct PSO_context *ctx) {
	SYS_DBG("ZNCC_cleanup\n");
	if (ctx == NULL || ctx->priv == NULL) return -1;

	struct DIC_ZNCC_context *zncc_ctx = (struct DIC_ZNCC_context *)ctx->priv;
    if (zncc_ctx->cur_subset_info.subset_data) {
		free(zncc_ctx->cur_subset_info.subset_data);
		zncc_ctx->cur_subset_info.subset_data = NULL;
	}
	if (zncc_ctx->ref_subset_info.subset_data) {
		free(zncc_ctx->ref_subset_info.subset_data);
		zncc_ctx->ref_subset_info.subset_data = NULL;
	}
    free(zncc_ctx);
    ctx->priv = NULL;
    SYS_DBG("ZNCC_cleanup: Resources released\n");
	return 0;
}

// SSD Critiria
struct PSO_cost_func_ops ssd_ops = {
	.init			= SSD_init,
    .cost_function  = SSD_cost_function,
	.cleanup		= SSD_cleanup
};

int SSD_init(struct SYS_INFO *info) {
	return 0;
}

double SSD_cost_function(struct PSO_context *ctx) {
	return 0;
}

int SSD_cleanup(struct PSO_context *ctx) {
	return 0;
}