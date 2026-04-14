import ctypes
import numpy as np
import cv2

lib = ctypes.CDLL('./2D_DIC.dll')

# parm type
lib.process_image.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), # ref_img
    ctypes.POINTER(ctypes.c_uint8), # cur_img
    ctypes.c_int,                   # width
    ctypes.c_int,                   # height
    ctypes.c_int,                   # population
    ctypes.c_int,                   # subset_side_len
    ctypes.POINTER(ctypes.c_int),   # img_ref_pt
    ctypes.POINTER(ctypes.c_float)  # result_buffer (對應 C 的 float*)
]

# return type
lib.process_image.restype = None

# parameters
population = 40
subset_side_len = 31

def main():
    img_ref = cv2.imread('image/img1.jpg', cv2.IMREAD_GRAYSCALE)
    img_cur = cv2.imread('image/img1_shefted_5_5.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img_ref is None or img_cur is None:
        print("Error: no images!")
        return

    h, w = img_ref.shape
    img_ref_pt_x = 426
    img_ref_pt_y = 320
    ref_ptr = img_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    cur_ptr = img_cur.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    img_ref_pt = np.array((img_ref_pt_y,img_ref_pt_x), dtype=np.int32)
    img_ref_pt_ptr = img_ref_pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result_buffer = np.zeros(3, dtype=np.float32)
    result_buffer_ptr = result_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.process_image(
        ref_ptr, 
        cur_ptr, 
        w, 
        h, 
        population,
        subset_side_len,
        img_ref_pt_ptr,
        result_buffer_ptr
    )

    result_y = result_buffer[0]
    result_x = result_buffer[1]
    coef  = result_buffer[2]

    print(f"Result from C: X={result_x:.3f}, Y={result_y:.3f}, Coef={coef:.3f}")

if __name__ == "__main__":
    main()