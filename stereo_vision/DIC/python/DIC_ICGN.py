## ===== DIC (digital image correlation) ===== ##
import numpy as np
import time
import ctypes
import stereo_vision.config as CF
from stereo_vision import config_user as CF_user
from stereo_vision.config_DIC import DIC_config
from stereo_vision.DIC.python.common import DIC_search_pt_type
from ctypes import cdll, c_int, c_double, POINTER

def run_DIC(dic_config: DIC_config, lib_PSO, lib_ICGN, ICGN_proc, PSO_proc):
       
       img_ref                                   = dic_config.dic_image.ref
       img_cur                                   = dic_config.dic_image.cur
       img_ref_x                                 = dic_config.img_ref_pt.pt_x
       img_ref_y                                 = dic_config.img_ref_pt.pt_y
       subset_side_len                           = dic_config.subset_ref_info.subset_side_len
       subset_ref_data                           = dic_config.subset_ref_info.subset_data
       H_inv_mat                                 = dic_config.img_grad.H_inv_mat
       J_mat                                     = dic_config.img_grad.J_mat
       translation                               = dic_config.init_param.translate
       population                                = dic_config.coarse_method_cfg.population
       search_type                               = dic_config.search_type

       if img_ref.dtype != np.float64 or not img_ref.flags['C_CONTIGUOUS']:
              img_ref = np.ascontiguousarray(img_ref, dtype=np.float64)
       if img_cur.dtype != np.float64 or not img_cur.flags['C_CONTIGUOUS']:
              img_cur = np.ascontiguousarray(img_cur, dtype=np.float64)
       h, w = img_ref.shape

       img_ref_ptr = img_ref.ctypes.data_as(ctypes.POINTER(c_double))
       img_cur_ptr = img_cur.ctypes.data_as(ctypes.POINTER(c_double))

       img_ref_pt_x_guess = img_ref_x - translation if search_type == DIC_search_pt_type.initial else img_ref_x
       img_ref_pt_y_guess = img_ref_y

       PSO_proc._ref_pt_pos[0] = img_ref_y; PSO_proc._ref_pt_pos[1] = img_ref_x
       PSO_proc._cur_pt_pos[0] = img_ref_pt_y_guess; PSO_proc._cur_pt_pos[1] = img_ref_pt_x_guess
       start_pso = time.time()
       lib_PSO.process_image(
              img_ref_ptr, 
              img_cur_ptr, 
              w, 
              h, 
              population,
              subset_side_len,
              PSO_proc._ref_pt_ptr,
              PSO_proc._cur_pt_ptr,
              PSO_proc._result_ptr
       )
       PSO_dis_y, PSO_dis_x = PSO_proc._result_buf[0], PSO_proc._result_buf[1]
       end_pso = time.time()
       time_pso = end_pso - start_pso
       print(f"time_pso: {time_pso:.5f}")
       # result_buffer = PSO_proc()

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
       warp_function = np.array([(1+xx, xy, x),
                                 (yx, 1+yy, y),
                                 (0, 0, 1)], dtype=np.float64)

       cnt = 0
       limit = CF_user.DIC_ICGN_ACCURACY_INIT
       eps = 1e-12
       # [NOTICE] FROM NOW ON IS (X,Y), NOT (Y,X) anymore!!
       point_ini = np.array((img_ref_pt_x_guess, img_ref_pt_y_guess), dtype=np.float64)
       subset_side_len_half = int(0.5*(subset_side_len-1)) 
       start_ICGN = time.time()
       while limit > CF_user.DIC_ICGN_ACCURACY and cnt < CF_user.DIC_ICGN_CNT:
              target_mat_g = ICGN_proc.update_target_img_subset(img_cur, point_ini, lib_ICGN, warp_function)
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
       end_ICGN = time.time()
       time_ICGN = end_ICGN - start_ICGN
       # print(f"time_ICGN: {time_ICGN:.5f}")
       
       X = warp_function[0][2]
       Y = warp_function[1][2]
       # print(f"(X,Y)=({X:.3f},{Y:.3f})")
       img_cur_y = Y + img_ref_pt_y_guess
       img_cur_x = X + img_ref_pt_x_guess
       return img_cur_x, img_cur_y



