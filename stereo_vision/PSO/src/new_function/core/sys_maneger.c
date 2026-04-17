
#include "include/system_comm.h"
#include "include/DIC_comm.h"


struct SYS_INFO *SYS_create() {
    struct SYS_INFO *info = malloc(sizeof(struct SYS_INFO));
    if (!info) {
        SYS_DBG("fail to create system memory!\n");
        return NULL;
    } else {
        SYS_DBG("Allocate memory (%p) for info successfully\n", info);
    }
    return info;
}

int SYS_clean(SYS_INFO *info) {
    if (!info) {
        SYS_DBG("SYS_clean failed!!\n");
        return SYS_FAILURE;
    } else {
        free(info);
    }
    return SYS_SUCCESS;
}