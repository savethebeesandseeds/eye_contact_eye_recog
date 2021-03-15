import os
import sys
import dlib
from time import time, sleep

import utils_get_faces
import utils_for_files

t0_start = time()
# --- get file root path
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TEMP_FOLDER"] = os.path.normpath(os.path.join(ROOT, "../temp"))
# --- get file root path
image_path = "../DATA/img_people/many_faces_1.jpg"
image_path = "../DATA/img_people/faces_360_1.jpg"
image_path = os.path.normpath(os.path.join(ROOT, image_path))
# --- get file for landmark detection
landmark_path = "../dlib-19.21/download_from_source/shape_predictor_68_face_landmarks.dat"
landmark_path = os.path.normpath(os.path.join(ROOT, landmark_path))
# --- print paths
print("Proyect ROOT path: {}".format(ROOT))
print("Proyect TEMP folder: {}".format(os.environ["TEMP_FOLDER"]))
print("IMAGE path: {}".format(image_path))
# --- ----
# --- make faces class
DET_F = utils_get_faces.DETECT_FACES(
    img_path_=image_path, 
    temp_folder=os.environ["TEMP_FOLDER"], 
    face_detector_method='hog', 
    lm_predictor_path=landmark_path)
# --- get image
t_start = time()
DET_F.load_image_file()
print("execution time to load_image: {}".format(time()-t_start))
# --- get faces
t_start = time()
DET_F.get_faces(upsample_factor=2)
print("execution time to get_faces (2): {}".format(time()-t_start))
# --- get face locations
t_start = time()
DET_F.face_locations_to_pixel_corners(print_flag = False, save_flag = True)
print("execution time to locate_face_pixels: {}".format(time()-t_start))
# --- get face landmarks
t_start = time()
DET_F.get_land_marks()
print("execution time to get_land_marks: {}".format(time()-t_start))
# --- final report
print("total execution time: {}".format(time()-t0_start))

print("STOP")
dlib.hit_enter_to_continue()