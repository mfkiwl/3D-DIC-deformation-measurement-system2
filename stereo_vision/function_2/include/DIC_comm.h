
#include <stdlib.h>
#include <stdint.h>


#define square(x) ((x)*(x))

typedef struct DIC_config{
    uint16_t subset_side_len;
    //uint16_t subset_side_len_half;
    uint16_t scan_side_len;

} DIC_config;

typedef struct DIC_context {
    struct DIC_config config;
    void *priv;
} DIC_context;


/*
private 為特定任務中所需的資料 (pointer: 8bytes)
*/