## ===== DIC (digital image correlation) ===== ##
import numpy as np
import ctypes
import stereo_vision.config as CF
import time
import stereo_vision.DIC.python.ICGN
from stereo_vision.DIC.python.common import DIC_search_pt_type
import cv2


def run_DIC(img_ref,
            img_cur,
            img_ref_x, 
            img_ref_y,
            subset_size_len,
            H_inv_mat,
            J_mat,
            translation,
            population,
            search_type):
       
       print(f"search_type={search_type}")
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
       lib = ctypes.CDLL(f"{CF.BUILD_DIR}/2D_DIC.dll")
       # parm type
       lib.process_image.argtypes = [
              ctypes.POINTER(ctypes.c_double),    # ref_img
              ctypes.POINTER(ctypes.c_double),    # cur_img
              ctypes.c_int,                       # width
              ctypes.c_int,                       # height
              ctypes.c_int,                       # population
              ctypes.c_int,                       # subset_side_len
              ctypes.POINTER(ctypes.c_double),    # img_ref_pt
              ctypes.POINTER(ctypes.c_double),    # img_cur_pt
              ctypes.POINTER(ctypes.c_double)     # result_buffer
       ]
       lib.process_image.restype = None

       img_ref_ptr                               = img_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_cur_ptr                               = img_cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_ref_pt_pos_ptr                        = img_ref_pt_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       img_cur_pt_pos_ptr                        = img_cur_pt_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
       result_buffer_ptr                         = result_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) 
             
       # run_PSO
       lib.process_image(
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
       print(f"PSO_dis_y={PSO_dis_y}")
       print(f"PSO_dis_x={PSO_dis_x}")
       print(f"correlation_coef={correlation_coef}")
       # print(f"(PSO_dis_x,PSO_dis_y)=({PSO_dis_x},{PSO_dis_y})")

       # Reference subset
       ref_mat_f = img_ref[img_ref_pt_y_guess - subset_side_len_half : img_ref_pt_y_guess + subset_side_len_half + 1,
                           img_ref_pt_x_guess - subset_side_len_half : img_ref_pt_x_guess + subset_side_len_half + 1]
       
       # Mean of Reference subset
       ref_mat_f_mean = np.mean(ref_mat_f)
       # Delta_f
       delta_f = np.std(ref_mat_f, ddof=0)

       # define displacement vector: P
       x = PSO_dis_x # obtain from PSO
       xx = 0
       xy = 0
       y = PSO_dis_y # obtain from PSO
       yx = 0
       yy = 0
       # warp function coefficient of deformed subset
       warp_function = np.array([(1+xx, xy, x),\
                                 (yx, 1+yy, y),\
                                 (0, 0, 1)], dtype=np.float64)

       """ ===== ICGN ===== """
       cnt = 0
       limit = 0.1
       eps = 1e-12
       # [NOTICE] FROM NOW ON IS (X,Y), NOT (Y,X) anymore!!
       point_ini = np.array((img_ref_pt_x_guess, img_ref_pt_y_guess), dtype=np.float64)
       while limit > 0.0001 and cnt < 20:
              target_mat_g = stereo_vision.DIC.python.ICGN.update_target_img_subset(subset_size_len, img_cur, point_ini, warp_function)
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
       print(f"(X,Y)=({X:.3f},{Y:.3f})")
       img_cur_y = Y + img_ref_pt_y_guess
       img_cur_x = X + img_ref_pt_x_guess
       return img_cur_x, img_cur_y



       # img_ref_sub_f32 = img_ref[img_ref_y-subset_side_len_half:img_ref_y+subset_side_len_half+1,\
       #                             img_ref_x-subset_side_len_half:img_ref_x+subset_side_len_half+1]
       # img_ref_sub_f32_mean = np.array(np.mean(img_ref_sub_f32), dtype=np.float64)
       # img_cur_sub_f32 = np.zeros((subset_size_len, subset_size_len), dtype=np.float32)