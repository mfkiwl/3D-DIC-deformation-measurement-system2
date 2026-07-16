import cv2
import numpy as np
import stereo_vision.config_DIC as cfg_dic
from stereo_vision.DIC.python.common import DIC_search_pt_type
import stereo_vision.DIC.python.DIC_session as DIC_session

def run_lk_pyramid_core(session: DIC_session.Stereo_DIC_session, dic_config: cfg_dic.DIC_config):

    img_ref                                   = dic_config.dic_image.ref
    img_cur                                   = dic_config.dic_image.cur
    sub_len                                   = dic_config.subset_ref_info.subset_side_len
    translation                               = dic_config.init_param.translate
    search_type                               = dic_config.search_type

    pt_ref_mat_lk = session.dic_buf.C1B_points.astype(np.float32).reshape(-1, 1, 2)  # astype: copy new ram

    if search_type == DIC_search_pt_type.initial:
        pt_ref_mat_lk[..., 0] -= translation

    lk_params = dict(
        winSize=(sub_len, sub_len),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    pt_cur_mat_lk, st, err = cv2.calcOpticalFlowPyrLK(
        img_ref,
        img_cur,
        pt_ref_mat_lk,
        None,
        **lk_params
    )
    if st is None or st.sum() == 0:
        print("[ERROR] LK Pyramid FAIL!")
        return None
    
    if pt_cur_mat_lk.shape[0] != session.cfg.pt_mat_side_len * session.cfg.pt_mat_side_len:
        print("[ERROR] point size mismatch")
        return None

    session.dic_buf.C2B_points = pt_cur_mat_lk.reshape(session.cfg.pt_mat_side_len, session.cfg.pt_mat_side_len, 2)
    return None


def run_lk_pyramid_core_single(session, dic_config):
    img_ref                                   = dic_config.dic_image.ref
    img_cur                                   = dic_config.dic_image.cur
    img_ref_x                                 = dic_config.img_ref_pt.pt_x
    img_ref_y                                 = dic_config.img_ref_pt.pt_y
    sub_len                                   = dic_config.subset_ref_info.subset_side_len
    translation                               = dic_config.init_param.translate
    search_type                               = dic_config.search_type

    img_ref = img_ref.astype(np.uint8)
    img_cur = img_cur.astype(np.uint8)

    # print(img_ref.dtype, img_ref.shape)
    # print(img_cur.dtype, img_cur.shape)
    # print(sub_len)

    pt_ref_lk = np.array(
        [[img_ref_x, img_ref_y]],
        dtype=np.float32
    ).reshape(-1, 1, 2)
    # astype: copy new ram

    if search_type == DIC_search_pt_type.initial:
        pt_ref_lk[..., 0] -= translation

    lk_params = dict(
        winSize=(sub_len, sub_len),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    pt_cur_lk, st, err = cv2.calcOpticalFlowPyrLK(
        img_ref,
        img_cur,
        pt_ref_lk,
        None,
        **lk_params
    )
    if st is None or st.sum() == 0:
        print("[ERROR] LK Pyramid FAIL!")
        return None

    img_cur_x, img_cur_y = pt_cur_lk[0, 0]
    coarse_x = img_cur_x - img_ref_x
    coarse_y = img_cur_y - img_ref_y
    # print(f"coarse_x:{coarse_x}")
    # print(f"coarse_y:{coarse_y}")
    return coarse_x, coarse_y