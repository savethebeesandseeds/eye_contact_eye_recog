import os
import sys
import dlib
from time import time, sleep

import utils_get_faces
import utils_for_files
import utils_for_video
import utils_for_bodies
import utils_for_yolo

t0_start = time()
# --- get file root path
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TEMP_FOLDER"] = os.path.normpath(os.path.join(ROOT, "../temp"))
utils_for_files.reset_folder(os.environ["TEMP_FOLDER"], just_content=True)
# --- get file root path
source_path = "../DATA/img_people/many_faces_1.jpg"
source_path = "../DATA/img_people/faces_360_1.jpg"
source_path = "../DATA/vid_1.mp4"
source_path = os.path.normpath(os.path.join(ROOT, source_path))
# --- cnn path
cnn_path = os.path.normpath(os.path.join(ROOT, '../dlib-19.21/download_from_source/mmod_human_face_detector.dat'))
# --- get file for landmark detection
landmark_path = "../dlib-19.21/download_from_source/shape_predictor_68_face_landmarks.dat"
landmark_path = os.path.normpath(os.path.join(ROOT, landmark_path))
# --- print paths
print("Proyect ROOT path: {}".format(ROOT))
print("Proyect TEMP folder: {}".format(os.environ["TEMP_FOLDER"]))
print("SOURCE path: {}".format(source_path))

# --- ----
DET_V = utils_for_video.video_mech(
    source=source_path, 
    process_at_fps=0.25, 
    temp_folder=os.environ["TEMP_FOLDER"])
success = DET_V.get_frame()
img_path = DET_V.save_c_image()
# --- ---- YOLO - TINY
size = 512
min_precision = 0.1
max_overlap = 0.01
only_humans = True

yolo_model = utils_for_yolo.Yolo_mech(
    isTiny=True, 
    size=size, 
    min_precision=min_precision, 
    max_overlap=max_overlap, 
    only_humans=only_humans, 
    state_dict='./yolo/models/yolov3_tiny_coco_01.h5'
)
print('img_path:', img_path)
yolo_model.process_img(img_path)
yolo_model.print_img('yolo_output.png')
input("yolo finish, stop...")
# --- ---- object candidates
utils_for_bodies.object_candidates(image_file=img_path, temp_folder=os.environ["TEMP_FOLDER"])
# --- ----
input("STOP!!")
# --- make faces class
DET_F = utils_get_faces.DETECT_FACES(
    img_path_=img_path, 
    temp_folder=os.environ["TEMP_FOLDER"], 
    face_detector_method='hog', # hog/cnn 
    lm_predictor_path=landmark_path,
    cnn_path=cnn_path,
    reset_env=True)
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

input("STOP #n!\n")
dlib.hit_enter_to_continue()