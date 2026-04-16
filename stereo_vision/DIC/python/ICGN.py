
import stereo_vision.config as CF
import numpy as np
from ctypes import cdll, c_int, c_double, POINTER
import os

dll_path = f'{CF.BUILD_DIR}/dll/ICGN.dll'
if not os.path.exists(dll_path):
    print(f"file not found:{dll_path}")

# update target_img_subset(subset_size_len * subset_size_len). if not deformed, skip warp_coef
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