
print("\n<< Stereo_DIC_PSO_ICGN >>")
import numpy as np
import cv2 as cv
import os
import sys
import time
from stereo_vision import config_user as CF_user
import stereo_vision.tools.math.src.hessian
import stereo_vision.DIC.python.DIC_ICGN as DIC_ICGN
import stereo_vision.camera_calibration.python.image_calibration as img_cal
from stereo_vision.DIC.python.DIC_session import create_session
from stereo_vision.tools.vision.src.click_tool import get_click_point
from stereo_vision.tools.vision.src.processor import rotate_image, extract_patch_bicubic
from stereo_vision.DIC.python.common import DIC_search_pt_type

from stereo_vision.config_DIC import (
    DIC_config, DIC_Image, Img_Ref_Pt_Pos, 
    Stereo_DIC_Init_Param, Subset_Info, 
    Img_Grad_Info, Coarse_Search_Method, PSO_Config
)

from stereo_vision.DIC.python.DIC_session import (
    DIC_user_config, create_session,
)

from stereo_vision.config_user import (
    Test_Mode
)

mode = Test_Mode.in_plane

pt_mat_len = int(np.sqrt(CF_user.TEST_POINT_ARRAY))
pt_mat_len_half = int((pt_mat_len - 1) / 2)

file_name = f"{CF_user.LOAD_MIN}_{CF_user.LOAD_MAX}kg_image1.jpg"
config = DIC_user_config(pt_mat_len, CF_user.TEST_SUBSET_SIZE_1B1A, CF_user.TEST_SUBSET_SIZE_2B2A)
session = create_session(config)
(
    session.load_stereo_images_ref(file_name)
            .pre_process()
            .get_img_sobel()
)

C1_B_x_ini, C1_B_y_ini = get_click_point(session.img_buf.img1_ref_rec_show, 'img1_ref_rec_show', 'set a reference point on img_1B')
C2_B_x_ini, C2_B_y_ini = get_click_point(session.img_buf.img2_ref_rec_show, 'img2_ref_rec_show', 'set a reference point on img_2B')
session.free_show_image()
session.get_reprojection_info()

lib_ICGN        = session.lib.ICGN
lib_PSO         = session.lib.PSO
# lib_interp      = session.lib.interp

for ROW in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
    for COL in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
        row = ROW + pt_mat_len_half
        col = COL + pt_mat_len_half

        C1_B_x = int(CF_user.TEST_INTERVAL*COL + C1_B_x_ini)
        C1_B_y = int(CF_user.TEST_INTERVAL*ROW + C1_B_y_ini)
        session.dic_buf.C1B_points[row][col][0] = C1_B_y
        session.dic_buf.C1B_points[row][col][1] = C1_B_x
        # ===== Image gradient =====
        subset_len_1B2B_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B2B-1))
        img_grad_1B2B_y = session.img_buf.img1_ref_sobel_y[C1_B_y - subset_len_1B2B_half:C1_B_y + subset_len_1B2B_half + 1,\
                                                           C1_B_x - subset_len_1B2B_half:C1_B_x + subset_len_1B2B_half + 1]
        img_grad_1B2B_x = session.img_buf.img1_ref_sobel_x[C1_B_y - subset_len_1B2B_half:C1_B_y + subset_len_1B2B_half + 1,\
                                                           C1_B_x - subset_len_1B2B_half:C1_B_x + subset_len_1B2B_half + 1]
        H_inv_1B2B, J_1B2B = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_1B2B, img_grad_1B2B_x, img_grad_1B2B_y)
        C1B_subset_center_pt = np.array((C1_B_x,C1_B_y), dtype=np.float64)
        img_1B_sub = DIC_ICGN.update_target_img_subset(CF_user.TEST_SUBSET_SIZE_1B1A, session.img_buf.img1_ref_rec_gray, C1B_subset_center_pt, lib_ICGN)
        session.dic_buf.img_1B_sub_zone[row][col][:][:] = img_1B_sub
        
        dic_config = DIC_config (
            coarse_method       = Coarse_Search_Method.PSO,
            coarse_method_cfg   = PSO_Config(CF_user.PSO_population),
            dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img2_ref_rec_gray),
            img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C1_B_x, pt_y=C1_B_y),
            init_param          = Stereo_DIC_Init_Param(translate = (C1_B_x_ini - C2_B_x_ini)),
            subset_ref_info     = Subset_Info(img_1B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B),
            subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B),
            img_grad            = Img_Grad_Info(H_inv_mat=H_inv_1B2B, J_mat=J_1B2B),
            search_type         = DIC_search_pt_type.initial
        )
        C2_B_x, C2_B_y = DIC_ICGN.run_DIC(dic_config, lib_PSO, lib_ICGN)
        X_ref, Y_ref, Z_ref = session.disparity_to_3d_pt(C1_B_x, C1_B_y, C2_B_x)
        session.dic_buf.C2B_points[row][col] = (C2_B_y, C2_B_x)
        session.dic_buf.WC_bef_zone[row][col] = (X_ref, Y_ref, Z_ref)
        
        # ===== 1B1A ===== #
        subset_len_1B1A_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1))
        img_grad_1B1A_x = session.img_buf.img1_ref_sobel_x[C1_B_y - subset_len_1B1A_half:C1_B_y + subset_len_1B1A_half+1,\
                                                           C1_B_x - subset_len_1B1A_half:C1_B_x + subset_len_1B1A_half+1]
        img_grad_1B1A_y = session.img_buf.img1_ref_sobel_y[C1_B_y - subset_len_1B1A_half:C1_B_y + subset_len_1B1A_half+1,\
                                                           C1_B_x - subset_len_1B1A_half:C1_B_x + subset_len_1B1A_half+1]
        H_inv_1B1A, J_1B1A = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_1B1A, img_grad_1B1A_x, img_grad_1B1A_y)
        session.dic_buf.H1B1A_inv_all[row][col][:][:]   = H_inv_1B1A[:][:]
        session.dic_buf.J1B1A_all[row][col][:][:][:]    = J_1B1A[:][:][:]
        
        # ===== 2B2A ===== #
        subset_side_len_2B2A_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1))
        C2B_subset_center_pt = np.array((C2_B_x,C2_B_y), dtype=np.float64)
        img_2B_sub = DIC_ICGN.update_target_img_subset(CF_user.TEST_SUBSET_SIZE_2B2A, session.img_buf.img2_ref_rec_gray, C2B_subset_center_pt, lib_ICGN)
        
        pad = 1  # Sobel need more 1 pixel to expand boarder
        img_2B_sub_pad = cv.copyMakeBorder(img_2B_sub, pad, pad, pad, pad, borderType=cv.BORDER_REFLECT)
        img_2B_sobel_y = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 0, 1)*0.125
        img_2B_sobel_x = cv.Sobel(img_2B_sub_pad, cv.CV_64F, 1, 0)*0.125
        img_grad_2B2A_y = img_2B_sobel_y[pad:-pad, pad:-pad]
        img_grad_2B2A_x = img_2B_sobel_x[pad:-pad, pad:-pad]
        
        H_inv_2B2A, J_2B2A = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_2B2A, img_grad_2B2A_x, img_grad_2B2A_y) 
        session.dic_buf.H2B2A_inv_all[row][col][:][:]   = H_inv_2B2A[:][:]
        session.dic_buf.J2B2A_all[row][col][:][:][:]    = J_2B2A[:][:][:]
        # save img_2B_sub for each point
        session.dic_buf.img_2B_sub_zone[row][col][:][:] = img_2B_sub
        
        session.img_buf.img1_ref_rec = cv.circle(session.img_buf.img1_ref_rec, (int(C1_B_x), int(C1_B_y)), 5,\
                                (0, 255, 255), 1)  
        session.img_buf.img2_ref_rec = cv.circle(session.img_buf.img2_ref_rec, (int(C2_B_x), int(C2_B_y)), 5,\
                                (0, 255, 255), 1)

cv.imshow('session.img_buf.img1_ref_rec', session.img_buf.img1_ref_rec)
cv.imshow('session.img_buf.img2_ref_rec', session.img_buf.img2_ref_rec)
cv.waitKey(0)
cv.destroyAllWindows()

dis_sum = 0
total_time = 0

base_cfg = DIC_config (
    coarse_method       = Coarse_Search_Method.PSO,
    coarse_method_cfg   = PSO_Config(CF_user.PSO_population),
    dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img1_cur_rec_gray),
    img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C1_B_x, pt_y=C1_B_y),
    init_param          = Stereo_DIC_Init_Param(translate = None),
    subset_ref_info     = Subset_Info(img_1B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
    subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
    img_grad            = Img_Grad_Info(H_inv_mat=H_inv_1B1A, J_mat=J_1B1A),
    search_type         = DIC_search_pt_type.normal
)

for img_idx in range(1, CF_user.TEST_TARGET_IMG_PAIR_NUM + 1,1):
    loaded_file_name = f"{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image{img_idx}.jpg"
    print(f"loaded_file_name: {loaded_file_name}")
    (
        session.load_stereo_images_cur(loaded_file_name)
                .pre_process_cur()
    )

    for ROW in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
        for COL in range(-pt_mat_len_half, pt_mat_len_half + 1, 1):
            row = ROW + pt_mat_len_half
            col = COL + pt_mat_len_half
            C1_B_y, C1_B_x = session.dic_buf.C1B_points[row][col]
            C2_B_y, C2_B_x = session.dic_buf.C2B_points[row][col]

            # ===== 1B1A ===== #
            start = time.time()
            H_inv_1B1A[:][:]        = session.dic_buf.H1B1A_inv_all[row][col][:][:]
            J_1B1A[:][:][:]         = session.dic_buf.J1B1A_all[row][col][:][:][:]
            img_1B_sub              = session.dic_buf.img_1B_sub_zone[row][col]

            dic_config = DIC_config (
                coarse_method       = Coarse_Search_Method.PSO,
                coarse_method_cfg   = PSO_Config(CF_user.PSO_population),
                dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img1_cur_rec_gray),
                img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C1_B_x, pt_y=C1_B_y),
                init_param          = Stereo_DIC_Init_Param(translate = None),
                subset_ref_info     = Subset_Info(img_1B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
                subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
                img_grad            = Img_Grad_Info(H_inv_mat=H_inv_1B1A, J_mat=J_1B1A),
                search_type         = DIC_search_pt_type.normal
            )
            C1_A_x, C1_A_y = DIC_ICGN.run_DIC(dic_config, lib_PSO, lib_ICGN)
            
            # ===== 2B2A ===== #
            H_inv_2B2A[:][:]        = session.dic_buf.H2B2A_inv_all[row][col][:][:]
            J_2B2A[:][:][:]         = session.dic_buf.J2B2A_all[row][col][:][:][:]
            img_2B_sub              = session.dic_buf.img_2B_sub_zone[row][col]

            
            dic_config = DIC_config (
                coarse_method       = Coarse_Search_Method.PSO,
                coarse_method_cfg   = PSO_Config(CF_user.PSO_population),
                dic_image           = DIC_Image(ref = session.img_buf.img2_ref_rec_gray, cur = session.img_buf.img2_cur_rec_gray),
                img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C2_B_x, pt_y=C2_B_y),
                init_param          = Stereo_DIC_Init_Param(translate = None),
                subset_ref_info     = Subset_Info(img_2B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A),
                subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A),
                img_grad            = Img_Grad_Info(H_inv_mat=H_inv_2B2A, J_mat=J_2B2A),
                search_type         = DIC_search_pt_type.normal
            )
            start_DIC = time.time()
            C2_A_x, C2_A_y = DIC_ICGN.run_DIC(dic_config, lib_PSO, lib_ICGN)
            end_DIC = time.time()
            time_dic = end_DIC - start_DIC
            # print(f"time_dic: {time_dic:.4f}")

            """ current world coordinate  """
            X_cur, Y_cur, Z_cur = session.disparity_to_3d_pt(C1_A_x, C1_A_y, C2_A_x)
            
            # Displacement between reference point and target point
            session.dic_buf.WC_aft_zone[row][col] = (X_cur, Y_cur, Z_cur)
            session.result_buf.disM[row][col][:] = session.dic_buf.WC_aft_zone[row][col][:] - session.dic_buf.WC_bef_zone[row][col][:]
            # out:z, in1:x(right+), in2:y(down+)
            dis_out = session.dic_buf.WC_aft_zone[row][col][2]-session.dic_buf.WC_bef_zone[row][col][2]
            dis_in_1 = session.dic_buf.WC_aft_zone[row][col][0]-session.dic_buf.WC_bef_zone[row][col][0]
            dis_in_2 = session.dic_buf.WC_aft_zone[row][col][1]-session.dic_buf.WC_bef_zone[row][col][1]
            dis_in_sum = np.sqrt(dis_in_1**2 + dis_in_2**2)
            
            if CF_user.TEST_MODE == Test_Mode.in_plane.value: # in plane
                # print(np.round(dis_in_sum, 6))
                dis_sum += dis_in_sum
            else: # out of plane
                # print(np.round(dis_out, 6))
                dis_sum += dis_out
            
            end = time.time()
            increase_time = end - start
            # print(f"increase_time: {increase_time:.4f}")
            total_time += increase_time

            session.img_buf.img1_cur_rec = cv.circle(session.img_buf.img1_cur_rec, (int(C1_A_x), int(C1_A_y)), 5,\
                                                    (0, 255, 255), 1)  
            session.img_buf.img2_cur_rec = cv.circle(session.img_buf.img2_cur_rec, (int(C2_A_x), int(C2_A_y)), 5,\
                                                    (0, 255, 255), 1)  
            
            session.result_buf.disM_out[row][col] = dis_out
            session.result_buf.disM_in_1[row][col] = dis_in_1
            session.result_buf.disM_in_2[row][col] = dis_in_2
    
cv.imshow('img_1A_rec', session.img_buf.img1_cur_rec)
cv.imshow('img_2A_rec', session.img_buf.img2_cur_rec)
cv.waitKey(0)
cv.destroyAllWindows()

print('Average time per point: ', total_time / (CF_user.TEST_POINT_ARRAY * CF_user.TEST_TARGET_IMG_PAIR_NUM))
print('Average dis:', dis_sum / (CF_user.TEST_TARGET_IMG_PAIR_NUM * CF_user.TEST_POINT_ARRAY))
print("End")

