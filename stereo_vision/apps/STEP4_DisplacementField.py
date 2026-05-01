
print("\n<< Stereo_DIC_PSO_ICGN >>")
import numpy as np
import cv2 as cv
import os
import sys
import time
from stereo_vision import config as CF
from stereo_vision import config_user as CF_user
import stereo_vision.tools.math.src.hessian
import stereo_vision.DIC.python.ICGN as ICGN
import stereo_vision.DIC.python.DIC_ICGN as DIC_ICGN
import stereo_vision.camera_calibration.python.image_calibration as img_cal
from stereo_vision.DIC.python.DIC_session import create_session
from stereo_vision.tools.vision.src.click_tool import click_recorder
from stereo_vision.tools.vision.src.processor import rotate_image, extract_patch_bicubic
from stereo_vision.DIC.python.common import DIC_search_pt_type

from stereo_vision.config_DIC import (
    DIC_config, DIC_Image, Img_Ref_Pt_Pos, 
    Stereo_DIC_Init_Param, Subset_Info, 
    Img_Grad_Info, Coarse_Search_Method, PSO_Config
)

from stereo_vision.DIC.python.DIC_session import (
    DIC_user_config, Stereo_DIC_session,
    create_session, img_buffer, DIC_buffer, system_config,
)

## Set the number of analysis points.
pt_mat_side_len = int(np.sqrt(CF_user.TEST_POINT_ARRAY))
pt_mat_side_len_half = int((pt_mat_side_len - 1) / 2)

# initial session
file_name = f"{CF_user.LOAD_MIN}_{CF_user.LOAD_MAX}kg_image1.jpg"
config = DIC_user_config(pt_mat_side_len, CF_user.TEST_SUBSET_SIZE_1B1A, CF_user.TEST_SUBSET_SIZE_2B2A)
session = create_session(config)
(
    session.load_stereo_images_ref(file_name)
            .pre_process()
            .get_img_sobel()
)

## Select first point in left image
cv.putText(session.img_buf.img1_ref_rec_show, 'set a reference point on img_1B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.namedWindow("session.img_buf.img1_ref_rec_show", cv.WINDOW_NORMAL)
cv.namedWindow("session.img_buf.img2_ref_rec_show", cv.WINDOW_NORMAL)
cv.imshow("session.img_buf.img1_ref_rec_show", session.img_buf.img1_ref_rec_show)
cv.imshow("session.img_buf.img2_ref_rec_show", session.img_buf.img2_ref_rec_show)

coor_1B = click_recorder()
print('Please set a reference point in session.img_buf.img1_ref_rec_show by clicking on the image.')
cv.setMouseCallback('session.img_buf.img1_ref_rec_show', coor_1B.callback_cam1, session.img_buf.img1_ref_rec_show)
cv.waitKey(0) 
cv.destroyAllWindows()
print("done\n")

## Select corresponding points in right image
cv.putText(session.img_buf.img2_ref_rec_show, 'set a corresponding point on img_2B', (20, 60),\
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv.namedWindow("session.img_buf.img1_ref_rec_show", cv.WINDOW_NORMAL)
cv.namedWindow("session.img_buf.img2_ref_rec_show", cv.WINDOW_NORMAL)
cv.imshow("session.img_buf.img1_ref_rec_show", session.img_buf.img1_ref_rec_show)
cv.imshow("session.img_buf.img2_ref_rec_show", session.img_buf.img2_ref_rec_show)

coor_2B = click_recorder()
print('Please set a reference point in session.img_buf.img2_ref_rec_show by clicking on the image.')
cv.setMouseCallback('session.img_buf.img2_ref_rec_show', coor_2B.callback_cam2, session.img_buf.img2_ref_rec_show)
cv.waitKey(0) 
cv.destroyAllWindows()

session.free_show_image()

## Read the image calibration file and obtain the projection matrix.
cv_file = cv.FileStorage()
cv_file.open(CF.STEREO_MAP_PATH, cv.FileStorage_READ)
Q = cv_file.getNode('Q').mat()
cv_file.release()

""" =============== parameters ==============="""
# start point : (C1_B_x_ini, C1_B_y_ini)
C1_B_x_ini = coor_1B.x
C1_B_y_ini = coor_1B.y
C2_B_x_ini = coor_2B.x
C2_B_y_ini = coor_2B.y

C1_B_x_ini = 468
C1_B_y_ini = 273
C2_B_x_ini = 164
C2_B_y_ini = 259

## check if C1_B_x_ini, y and C2_B_x_ini, y not defined
for var_name in ['C1_B_x_ini', 'C1_B_y_ini', 'C2_B_x_ini', 'C2_B_y_ini']:
    if var_name not in globals() or globals()[var_name] is None:
        print(f"[ERROR] {var_name} not defined or is None!")

print(f"C1_B_x_ini: {C1_B_x_ini}")
print(f"C1_B_y_ini: {C1_B_y_ini}")
print(f"C2_B_x_ini: {C2_B_x_ini}")
print(f"C2_B_y_ini: {C2_B_y_ini}")
print("")

# Due to the large distance (disparity) between the two cameras, 
# a translational distance is set to facilitate DIC in quickly finding corresponding points in the 2B image.
translate_1B2B = C1_B_x_ini - C2_B_x_ini
print(f"translate_1B2B:{translate_1B2B}")
if translate_1B2B < 0:
    print(f"translate_1B2B: {translate_1B2B} < 0 !!")
    exit(1)

focal = Q[2][3]                 # focal (unit:pixel)
baseline = 1/Q[3][2]            # baseline (unit:mm)
principal_x = -Q[0][3]          # The pt center coor of the camera.
principal_y = -Q[1][3]          # The pt center coor of the camera.

print(f"C1_B_x_ini: {C1_B_x_ini}")
print(f"C1_B_y_ini: {C1_B_y_ini}")
print(f"C2_B_x_ini: {C2_B_x_ini}")
print(f"C2_B_y_ini: {C2_B_y_ini}")
print("")

## Corrsponding points
for ROW in range(-pt_mat_side_len_half, pt_mat_side_len_half + 1, 1): # -2 ~ +2
    for COL in range(-pt_mat_side_len_half, pt_mat_side_len_half + 1, 1):
        C1_B_x = int(CF_user.TEST_INTERVAL*COL + C1_B_x_ini)
        C1_B_y = int(CF_user.TEST_INTERVAL*ROW + C1_B_y_ini)
        session.dic_buf.C1B_points[ROW + pt_mat_side_len_half][COL + pt_mat_side_len_half][0] = C1_B_y
        session.dic_buf.C1B_points[ROW + pt_mat_side_len_half][COL + pt_mat_side_len_half][1] = C1_B_x
        ## ========== Compute image gradient ========== """
        subset_side_len_1B2B_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B2B-1))
        img_grad_1B2B_y = session.img_buf.img1_ref_sobel_y[C1_B_y - subset_side_len_1B2B_half:C1_B_y + subset_side_len_1B2B_half + 1,\
                                                           C1_B_x - subset_side_len_1B2B_half:C1_B_x + subset_side_len_1B2B_half + 1]
        img_grad_1B2B_x = session.img_buf.img1_ref_sobel_x[C1_B_y - subset_side_len_1B2B_half:C1_B_y + subset_side_len_1B2B_half + 1,\
                                                           C1_B_x - subset_side_len_1B2B_half:C1_B_x + subset_side_len_1B2B_half + 1]
        H_inv_1B2B, J_1B2B = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_1B2B, img_grad_1B2B_x, img_grad_1B2B_y)
        C1B_subset_center_pt = np.array((C1_B_x,C1_B_y), dtype=np.float64)
        img_1B_sub = ICGN.update_target_img_subset(CF_user.TEST_SUBSET_SIZE_1B1A, session.img_buf.img1_ref_rec_gray, C1B_subset_center_pt)
        session.dic_buf.img_1B_sub_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:] = img_1B_sub
        
        dic_config = DIC_config (
            coarse_method       = Coarse_Search_Method.PSO,
            dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img2_ref_rec_gray),
            img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C1_B_x, pt_y=C1_B_y),
            init_param          = Stereo_DIC_Init_Param(translate = translate_1B2B),
            subset_ref_info     = Subset_Info(img_1B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B),
            subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B2B),
            img_grad            = Img_Grad_Info(H_inv_mat=H_inv_1B2B, J_mat=J_1B2B),
            pso_config          = PSO_Config(CF_user.PSO_population),
            search_type         = DIC_search_pt_type.initial
        )
        C2_B_x, C2_B_y = DIC_ICGN.run_DIC(dic_config)

        session.dic_buf.C2B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0] = C2_B_y
        session.dic_buf.C2B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1] = C2_B_x
        ## initial 3d coordinate
        disparity_1B2B = (C1_B_x - C2_B_x) # get disparity: xl-xr (unit:pixel)
        disparity_1B2B_reci = np.divide(1, disparity_1B2B)
        # 3D coordinate of Reference point (initial)
        X_origin = (C1_B_x - principal_x) * baseline * disparity_1B2B_reci
        Y_origin = (C1_B_y - principal_y) * baseline * disparity_1B2B_reci
        Z_origin = focal * baseline * disparity_1B2B_reci
        session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0] = X_origin
        session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1] = Y_origin
        session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][2] = Z_origin
        
        # pre-calculate Hessian & Jacobian for 1B1A & 2B2A
        # ===== 1B1A ===== #
        subset_side_len_1B1A_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_1B1A-1))
        img_grad_1B1A_x = session.img_buf.img1_ref_sobel_x[C1_B_y - subset_side_len_1B1A_half:C1_B_y + subset_side_len_1B1A_half+1,\
                                                           C1_B_x - subset_side_len_1B1A_half:C1_B_x + subset_side_len_1B1A_half+1]
        img_grad_1B1A_y = session.img_buf.img1_ref_sobel_y[C1_B_y - subset_side_len_1B1A_half:C1_B_y + subset_side_len_1B1A_half+1,\
                                                           C1_B_x - subset_side_len_1B1A_half:C1_B_x + subset_side_len_1B1A_half+1]
        H_inv_1B1A, J_1B1A = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_1B1A, img_grad_1B1A_x, img_grad_1B1A_y)
        session.dic_buf.H1B1A_inv_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:]   = H_inv_1B1A[:][:]
        session.dic_buf.J1B1A_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:][:]    = J_1B1A[:][:][:]
        
        # ===== 2B2A ===== #
        subset_side_len_2B2A_half = int(0.5*(CF_user.TEST_SUBSET_SIZE_2B2A-1))
        C2B_subset_center_pt = np.array((C2_B_x,C2_B_y), dtype=np.float64)
        img_2B_sub = ICGN.update_target_img_subset(CF_user.TEST_SUBSET_SIZE_2B2A, session.img_buf.img2_ref_rec_gray, C2B_subset_center_pt)
        img_grad_2B2A_x = extract_patch_bicubic(session.img_buf.img2_ref_sobel_x, C1_B_x, C1_B_y, CF_user.TEST_SUBSET_SIZE_2B2A)
        img_grad_2B2A_y = extract_patch_bicubic(session.img_buf.img2_ref_sobel_y, C1_B_x, C1_B_y, CF_user.TEST_SUBSET_SIZE_2B2A)
        H_inv_2B2A, J_2B2A = stereo_vision.tools.math.src.hessian.get_Hinv_jacobian(CF_user.TEST_SUBSET_SIZE_2B2A, img_grad_2B2A_x, img_grad_2B2A_y) 
        session.dic_buf.H2B2A_inv_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:]   = H_inv_2B2A[:][:]
        session.dic_buf.J2B2A_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:][:]    = J_2B2A[:][:][:]
        # save img_2B_sub for each point
        session.dic_buf.img_2B_sub_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:] = img_2B_sub
        
        session.img_buf.img1_ref_rec = cv.circle(session.img_buf.img1_ref_rec, (int(C1_B_x), int(C1_B_y)), 5,\
                                (0, 255, 255), 1)  
        session.img_buf.img2_ref_rec = cv.circle(session.img_buf.img2_ref_rec, (int(C2_B_x), int(C2_B_y)), 5,\
                                (0, 255, 255), 1)

cv.imshow('session.img_buf.img1_ref_rec', session.img_buf.img1_ref_rec)
cv.imshow('session.img_buf.img2_ref_rec', session.img_buf.img2_ref_rec)
cv.waitKey(0)
cv.destroyAllWindows()

# sys.exit(0)

dis_sum = 0
for img_idx in range(1, CF_user.TEST_TARGET_IMG_CNT + 1,1):
    loaded_file_name = f"{CF_user.LOAD_CUR}_{CF_user.LOAD_MAX}kg_image{img_idx}.jpg"
    print(f"loaded_file_name: {loaded_file_name}")
    (
        session.load_stereo_images_cur(loaded_file_name)
                .pre_process_cur()
    )

    start2 = time.time()
    for ROW in range(-pt_mat_side_len_half, pt_mat_side_len_half + 1, 1):
        for COL in range(-pt_mat_side_len_half, pt_mat_side_len_half + 1, 1):
            C1_B_y = session.dic_buf.C1B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0] #integer
            C1_B_x = session.dic_buf.C1B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1] #integer
            C2_B_y = session.dic_buf.C2B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0] #decimal
            C2_B_x = session.dic_buf.C2B_points[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1] #decimal

            # ===== 1B1A ===== #
            start = time.time()
            start_1B1A = time.time()

            H_inv_1B1A[:][:]        = session.dic_buf.H1B1A_inv_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:]
            J_1B1A[:][:][:]         = session.dic_buf.J1B1A_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:][:]
            img_1B_sub              = session.dic_buf.img_1B_sub_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half]

            dic_config = DIC_config (
                coarse_method       = Coarse_Search_Method.PSO,
                dic_image           = DIC_Image(ref = session.img_buf.img1_ref_rec_gray, cur = session.img_buf.img1_cur_rec_gray),
                img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C1_B_x, pt_y=C1_B_y),
                init_param          = Stereo_DIC_Init_Param(translate = None),
                subset_ref_info     = Subset_Info(img_1B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
                subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_1B1A),
                img_grad            = Img_Grad_Info(H_inv_mat=H_inv_1B1A, J_mat=J_1B1A),
                pso_config          = PSO_Config(CF_user.PSO_population),
                search_type         = DIC_search_pt_type.normal
            )
            C1_A_x, C1_A_y = DIC_ICGN.run_DIC(dic_config)

            end_1B1A = time.time()
            time_1B1A = end_1B1A - start_1B1A
            # print('time_1B1A:',time_1B1A)
            
            # ===== 2B2A ===== #
            start_2B2A = time.time()
        
            H_inv_2B2A[:][:]        = session.dic_buf.H2B2A_inv_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:]
            J_2B2A[:][:][:]         = session.dic_buf.J2B2A_all[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:][:][:]
            img_2B_sub              = session.dic_buf.img_2B_sub_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half]
            # print(f"img_2B_sub={img_2B_sub}")
            # print('debugging...')
            # sys.exit(0)

            dic_config = DIC_config (
                coarse_method       = Coarse_Search_Method.PSO,
                dic_image           = DIC_Image(ref = session.img_buf.img2_ref_rec_gray, cur = session.img_buf.img2_cur_rec_gray),
                img_ref_pt          = Img_Ref_Pt_Pos(pt_x=C2_B_x, pt_y=C2_B_y),
                init_param          = Stereo_DIC_Init_Param(translate = None),
                subset_ref_info     = Subset_Info(img_2B_sub, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A),
                subset_cur_info     = Subset_Info(None, subset_side_len = CF_user.TEST_SUBSET_SIZE_2B2A),
                img_grad            = Img_Grad_Info(H_inv_mat=H_inv_2B2A, J_mat=J_2B2A),
                pso_config          = PSO_Config(CF_user.PSO_population),
                search_type         = DIC_search_pt_type.normal
            )
            C2_A_x, C2_A_y = DIC_ICGN.run_DIC(dic_config)

            end_2B2A = time.time()
            time_2B2A = end_2B2A - start_2B2A
            # print('time_2B2A:',time_2B2A)
            
            """ current world coordinate  """
            # get disparity: xl-xr (unit:pixel)
            disparity_1A2A = (C1_A_x - C2_A_x)
            disparity_1A2A_reci = np.divide(1, disparity_1A2A)
            X_after = (C1_A_x-principal_x)*baseline*disparity_1A2A_reci
            Y_after = (C1_A_y-principal_y)*baseline*disparity_1A2A_reci
            Z_after = focal*baseline*disparity_1A2A_reci
            
            # Displacement between reference point and target point
            session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0] = X_after
            session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1] = Y_after
            session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][2] = Z_after
            session.dic_buf.disM[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:] = session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:] - session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][:]
            # out:z, in1:x(right+), in2:y(down+)
            dis_out = session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][2]-session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][2]
            #dis_out2 = np.dot(disM[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half],nVector)
            dis_in_1 = session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0]-session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][0]
            dis_in_2 = session.dic_buf.WC_aft_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1]-session.dic_buf.WC_bef_zone[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half][1]
            dis_in_sum = np.sqrt(dis_in_1**2 + dis_in_2**2)
            
            if CF_user.TEST_MODE_EN == 0: # in plane
                print(np.round(dis_in_sum, 6))
                dis_sum += dis_in_sum
            else: # out of plane
                print(np.round(dis_out, 6))
                dis_sum += dis_out
            
            session.img_buf.img1_cur_rec = cv.circle(session.img_buf.img1_cur_rec, (int(C1_A_x), int(C1_A_y)), 5,\
                                                    (0, 255, 255), 1)  
            session.img_buf.img2_cur_rec = cv.circle(session.img_buf.img2_cur_rec, (int(C2_A_x), int(C2_A_y)), 5,\
                                                    (0, 255, 255), 1)  
            
            end = time.time()
            total_time = end - start 
            
            session.dic_buf.disM_out[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half] = dis_out
            session.dic_buf.disM_in_1[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half] = dis_in_1
            session.dic_buf.disM_in_2[ROW+pt_mat_side_len_half][COL+pt_mat_side_len_half] = dis_in_2

    end2 = time.time()
    total_time2 = end2 - start2
    
cv.imshow('img_1A_rec', session.img_buf.img1_cur_rec)
cv.imshow('img_2A_rec', session.img_buf.img2_cur_rec)
cv.waitKey(0)
cv.destroyAllWindows()

print('Average time per point: ', total_time / (CF_user.TEST_POINT_ARRAY*10))
print('Average dis:', dis_sum / (CF_user.TEST_TARGET_IMG_CNT * CF_user.TEST_POINT_ARRAY))
print("End")

