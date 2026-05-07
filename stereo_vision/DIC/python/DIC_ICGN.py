## ===== DIC (digital image correlation) ===== ##
import numpy as np
import ctypes
import stereo_vision.config as CF
from stereo_vision import config_user as CF_user
from stereo_vision.config_DIC import DIC_config
from stereo_vision.DIC.python.common import DIC_search_pt_type
from ctypes import cdll, c_int, c_double, POINTER

# update target_img_subset(subset_size_len * subset_size_len). if not deformed, use default setting
def update_target_img_subset(subset_size_len, img, point_ini, warp_coef=None):
    img = np.asarray(img, dtype=np.float64)
    img_flat = img.flatten(order='C') # C:n row major
    height, width = img.shape
    if warp_coef is None:
        warp_coef = np.eye(3, dtype=np.float64)
    target_matrix_g_flat = np.zeros(subset_size_len*subset_size_len, dtype=np.float64) # create new 1d array
    m = cdll.LoadLibrary(f'{CF.BUILD_DIR}/ICGN.dll')
    
    m.update_target_img_subset.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, c_int, c_int
    ]
    m.update_target_img_subset.restype = None

    img_flat_ptr                    = img_flat.ctypes.data_as(POINTER(c_double))
    target_matrix_g_flat_ptr        = target_matrix_g_flat.ctypes.data_as(POINTER(c_double))
    point_ini_ptr                   = point_ini.ctypes.data_as(POINTER(c_double))
    warp_coef_ptr                   = warp_coef.ctypes.data_as(POINTER(c_double))

    m.update_target_img_subset(img_flat_ptr,
                                target_matrix_g_flat_ptr,
                                point_ini_ptr,
                                warp_coef_ptr,
                                width,
                                height,
                                subset_size_len)

    target_matrix_g = target_matrix_g_flat.reshape((subset_size_len, subset_size_len))
    return target_matrix_g


def run_DIC(dic_config: DIC_config, lib_PSO):
       img_ref                                   = dic_config.dic_image.ref
       img_cur                                   = dic_config.dic_image.cur
       img_ref_x                                 = dic_config.img_ref_pt.pt_x
       img_ref_y                                 = dic_config.img_ref_pt.pt_y
       subset_size_len                           = dic_config.subset_ref_info.subset_side_len
       subset_ref_data                           = dic_config.subset_ref_info.subset_data
       H_inv_mat                                 = dic_config.img_grad.H_inv_mat
       J_mat                                     = dic_config.img_grad.J_mat
       translation                               = dic_config.init_param.translate
       population                                = dic_config.coarse_method_cfg.population
       search_type                               = dic_config.search_type

       if search_type == DIC_search_pt_type.initial:
              img_ref_pt_x_guess                 = img_ref_x - translation
       else:
              img_ref_pt_x_guess                 = img_ref_x
       img_ref_pt_y_guess                        = img_ref_y
       
       h, w                                      = img_ref.shape
       subset_side_len_half                      = int(0.5*(subset_size_len-1)) 
       img_ref                                   = np.asarray(img_ref, dtype=np.double)
       img_cur                                   = np.asarray(img_cur, dtype=np.double)
       result_buffer                             = np.zeros(3, dtype=np.double) # [y, x, coef]
       img_ref_pt_pos                            = np.array((img_ref_y, img_ref_x), dtype=np.double)
       img_cur_pt_pos                            = np.array((img_ref_pt_y_guess, img_ref_pt_x_guess), dtype=np.double)

       # ===== measure interger displacment ===== 
       # lib = ctypes.CDLL(f"{CF.BUILD_DIR}/PSO.dll")
       # parm type
       # lib.process_image.argtypes = [
       #        ctypes.POINTER(ctypes.c_double),    # ref_img
       #        ctypes.POINTER(ctypes.c_double),    # cur_img
       #        ctypes.c_int,                       # width
       #        ctypes.c_int,                       # height
       #        ctypes.c_int,                       # population
       #        ctypes.c_int,                       # subset_side_len
       #        ctypes.POINTER(ctypes.c_double),    # img_ref_pt
       #        ctypes.POINTER(ctypes.c_double),    # img_cur_pt
       #        ctypes.POINTER(ctypes.c_double)     # result_buffer
       # ]
       # lib.process_image.restype = None

       img_ref_ptr                               = img_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_cur_ptr                               = img_cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_ref_pt_pos_ptr                        = img_ref_pt_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_cur_pt_pos_ptr                        = img_cur_pt_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       result_buffer_ptr                         = result_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) 
             
       # run_PSO
       lib_PSO.process_image(
              img_ref_ptr, 
              img_cur_ptr, 
              w, 
              h, 
              population,
              subset_size_len,
              img_ref_pt_pos_ptr,
              img_cur_pt_pos_ptr,
              result_buffer_ptr
       )

       PSO_dis_y = result_buffer[0]
       PSO_dis_x = result_buffer[1]
       correlation_coef = result_buffer[2]

       """ ===== ICGN ===== """
       ref_mat_f = subset_ref_data
       ref_mat_f_mean = np.mean(ref_mat_f)
       delta_f = np.std(ref_mat_f, ddof=0)

       # define displacement vector: P
       x = PSO_dis_x
       xx = 0
       xy = 0
       y = PSO_dis_y
       yx = 0
       yy = 0
       # warp function coefficient of deformed subset
       warp_function = np.array([(1+xx, xy, x),\
                                 (yx, 1+yy, y),\
                                 (0, 0, 1)], dtype=np.float64)

       cnt = 0
       limit = CF_user.DIC_ICGN_ACCURACY_INIT
       eps = 1e-12
       # [NOTICE] FROM NOW ON IS (X,Y), NOT (Y,X) anymore!!
       point_ini = np.array((img_ref_pt_x_guess, img_ref_pt_y_guess), dtype=np.float64)
       while limit > CF_user.DIC_ICGN_ACCURACY and cnt < CF_user.DIC_ICGN_CNT:
              target_mat_g = update_target_img_subset(subset_size_len, img_cur, point_ini, warp_function)
              target_mat_g_mean = np.mean(target_mat_g)
              delta_g = np.std(target_mat_g, ddof=0)

              corelation_sum = np.zeros(6, dtype=np.float64) 
              ratio = delta_f / (delta_g + eps) # eps: prevent zero
              residual_F_G = (ref_mat_f - ref_mat_f_mean) - ratio*(target_mat_g - target_mat_g_mean)
              corelation_sum = np.tensordot(J_mat, residual_F_G, axes=([0,1],[0,1]))

              corelation_sum = corelation_sum.reshape(6,1) 
              delta_P = (-H_inv_mat @ corelation_sum).flatten() # flatten turn 2d array to 1d array to get scalar
              limit = np.sqrt(np.square(delta_P[0]) + np.square(delta_P[1]*subset_side_len_half)+
                              np.square(delta_P[2]*subset_side_len_half) + np.square(delta_P[3])+
                              np.square(delta_P[4]*subset_side_len_half) + np.square(delta_P[5]*subset_side_len_half))

              warp_inc_function = np.array([[1+delta_P[1], delta_P[2], delta_P[0]],
                                           [delta_P[4], 1+delta_P[5], delta_P[3]],
                                           [0, 0, 1]], dtype=np.float64)

              warp_inc_function_inv = np.linalg.inv(warp_inc_function)
              # update warp function
              warp_function = warp_function @ warp_inc_function_inv
              cnt += 1
              
       X = warp_function[0][2]
       Y = warp_function[1][2]
       # print(f"(X,Y)=({X:.3f},{Y:.3f})")
       img_cur_y = Y + img_ref_pt_y_guess
       img_cur_x = X + img_ref_pt_x_guess
       return img_cur_x, img_cur_y



