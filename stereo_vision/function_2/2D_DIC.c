// run PSO & DIC
#include <stdio.h>
#include <stdlib.h>
#include "include/DIC_comm.h"
#include "include/PSO_comm.h"
#include "include/system_comm.h"


void Print_Args(SYS_INFO *info)
{
	int i = 0;
	INNO_UI_DBG("\n");

	INNO_UI_DBG("=====================\n");
	for (i = 0; i < info->args_info.args_num; i++) {
		INNO_UI_DBG("Argument[%02d/%02d]     : %s\n", i, info->args_info.args_num, info->args_info.args_ptr[i]);
	}
	info->args_info.args_ptr[info->args_info.args_num] = NULL;
	INNO_UI_DBG("=====================\n");
}

int main(){
    // allocate memory for struct
    int i;
    int ret = SYS_SUCCESS;
    SYS_INFO *info = malloc(sizeof(SYS_INFO));
    if (!info) {
        SYS_DBG("Unable to allocate memory to info!\n");
        return SYS_FAILURE;
    }else {
        SYS_DBG("Allocate memory (%p) for info successfully\n", info);
    }

    //Print_Args(info);
    
    

    // free struct

    if (ret)? return 0 : return 1;


    
}



/* 
Note:
stack空間比heap小很多，以空間考量前提下盡量用heap
stack空間不夠容易發生stack overflow



to do list

導入內聯函數 (inline): 函數前加 static inline
使用「位元欄位」 (Bit-fields) 處理開關 不要int
struct 的順序會影響佔用的記憶體大小
使用 const 與 restrict 關鍵字

*/

