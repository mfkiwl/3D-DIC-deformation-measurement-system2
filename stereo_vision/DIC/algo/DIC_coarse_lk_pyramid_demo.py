import cv2
import numpy as np

img1 = cv2.imread("stereo_vision/DIC/python/tmp/img1_shefted_5_5.jpg", 0)
img2 = cv2.imread("stereo_vision/DIC/python/tmp/img1.jpg", 0)

p0 = np.array([[[240, 320]]], dtype=np.float32)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)

p1, st, err = cv2.calcOpticalFlowPyrLK(
    img1,
    img2,
    p0,
    None,
    **lk_params
)

if st[0][0] == 1:

    old_x, old_y = p0[0][0]
    new_x, new_y = p1[0][0]

    print(f"Old point : ({old_x}, {old_y})")
    print(f"New point : ({new_x}, {new_y})")

else:
    print("Tracking failed")


# if __name__ == "__main__":
#     # read image (grayscale)
#     img = cv2.imread("stereo_vision/data/1.jpg", cv2.IMREAD_GRAYSCALE)

#     if img is None:
#         raise ValueError("Image not found")

#     pyramid = build_pyramid(img, levels=4)

#     show_pyramid(pyramid)