## PATH
import os
import Config_user as CF_user
cwd = os.getcwd()
WORKSPACE                       = os.path.join(cwd, "stereo_vision")
DLL_DIR                         = os.path.join(WORKSPACE, "dll")
FUNC_DIR                        = os.path.join(WORKSPACE, "function")
FUNC_ALGO_DIR                   = os.path.join(WORKSPACE, "function", "algorithm")
FUNC_INTERP_DIR                 = os.path.join(WORKSPACE, "function", "interpolation")
IMAGE_DIR                       = os.path.join(WORKSPACE, "image")
IMAGE_CAL_DIR                   = os.path.join("IMAGE_DIR", "Calibration")
IMAGE_CAL_LEFT_DIR              = os.path.join("IMAGE_DIR", "Calibration", "StereoLeft")
IMAGE_CAL_RIGHT_DIR             = os.path.join("IMAGE_DIR", "Calibration", "StereoRight")
IMAGE_TARGET_DIR                = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR)
IMAGE_TARGET_IN_DIR             = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "in")
IMAGE_TARGET_OUT_DIR            = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "out")
IMAGE_TARGET_IN_CAM1_DIR        = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "in", "cam1")
IMAGE_TARGET_IN_CAM2_DIR        = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "in", "cam2")
IMAGE_TARGET_OUT_CAM1_DIR       = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "out", "cam1")
IMAGE_TARGET_OUT_CAM2_DIR       = os.path.join("IMAGE_DIR", CF_user.TEST_IMG_DIR, "out", "cam2")
STEREP_MAP_PATH                 = f"{WORKSPACE}/stereoMap.xml"



