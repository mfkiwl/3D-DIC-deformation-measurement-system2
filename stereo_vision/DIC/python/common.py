from enum import IntEnum
from stereo_vision import config_user as CF_user
from copy import deepcopy
from stereo_vision.config_DIC import (
    DIC_config, DIC_Image, Img_Ref_Pt_Pos, 
    Stereo_DIC_Init_Param, Subset_Info, 
    Img_Grad_Info, Coarse_Search_Method, PSO_Config
)

class DIC_search_pt_type(IntEnum):
    initial         = 0
    normal          = 1


def get_subset(img, pt_x, pt_y, subset_side_len):
    
    return


def set_base_dic_cfg():
    base_cfg = DIC_config (
        coarse_method       = Coarse_Search_Method.PSO,
        coarse_method_cfg   = PSO_Config(CF_user.PSO_population),

        dic_image           = DIC_Image(ref = None, cur = None),
        img_ref_pt          = None,
        init_param          = Stereo_DIC_Init_Param(translate = None),
        subset_ref_info     = None,
        subset_cur_info     = Subset_Info(None, subset_side_len = None),
        img_grad            = None,

        search_type         = DIC_search_pt_type.normal
    )
    return base_cfg


def build_dic_cfg_1B2B(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img2_ref_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = DIC_search_pt_type.initial
    return cfg

def build_dic_cfg_1B1A(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img1_cur_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = DIC_search_pt_type.normal
    return cfg

def build_dic_cfg_2B2A(session, pt_x, pt_y, sub_ref, H_inv, J, trans=0):
    cfg = deepcopy(set_base_dic_cfg())
    cfg.dic_image           = DIC_Image(ref = session.img_buf.img2_ref_rec_gray, cur = session.img_buf.img2_cur_rec_gray)
    cfg.img_ref_pt          = Img_Ref_Pt_Pos(pt_x, pt_y)
    cfg.init_param          = Stereo_DIC_Init_Param(translate = (trans))
    cfg.subset_ref_info     = Subset_Info(sub_ref, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A)
    cfg.subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A)
    cfg.img_grad            = Img_Grad_Info(H_inv_mat=H_inv, J_mat=J)
    cfg.search_type         = DIC_search_pt_type.normal
    return cfg


def data_init_1B2B():
    return

def data_init_1B1A():
    return

def data_init_2B2A():
    return