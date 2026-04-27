#include "include/system_comm.h"
#include "include/PSO_comm.h"

void PSO_create_factory(PSO_COST_TYPE type, struct SYS_INFO *info) {
    // SELECT PSO COST FUNC TYPE
    if (type == PSO_TYPE_ZNCC) {
        struct DIC_ZNCC_context *zncc_ctx = malloc(sizeof(struct DIC_ZNCC_context));
        info->pso_ctx.priv       = zncc_ctx;
        info->pso_ctx.cost_ops   = &zncc_ops;
    } 
    else if (type == PSO_TYPE_SSD) {
        struct DIC_SSD_context *ssd_ctx = malloc(sizeof(struct DIC_SSD_context));
        info->pso_ctx.priv       = ssd_ctx;
        info->pso_ctx.cost_ops   = &ssd_ops;
    }

    if (info->pso_ctx.cost_ops->init(info) != 0) {
        SYS_DBG("PSO cost_func init failed!\n");
        free(info->pso_ctx.priv);
        return;
    }
}

int PSO_destroy(struct PSO_context *pso_ctx) {
    if (!pso_ctx) return PSO_FAILURE;
    if (pso_ctx->cost_ops && pso_ctx->cost_ops->cleanup) {
        pso_ctx->cost_ops->cleanup(pso_ctx->priv);
    }
    free(pso_ctx->priv);
    return PSO_SUCCESS;
}