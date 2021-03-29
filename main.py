# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import sys
import dlib
import random
from time import time, sleep

import utils_get_faces
import utils_for_files
import utils_for_video
import utils_for_bodies
import utils_for_yolo
import utils_for_blur
import utils_for_counting
import utils_for_cipher
t0_start = time()
# Download paths for yolo models
# https://drive.google.com/file/d/1Vng1GL9mJf06z5gcahtSsinzrXajqmls/view
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
# --- get file root path
ROOT = os.path.dirname(os.path.abspath(__file__))
# --- system security check
cipher = utils_for_cipher.CIPHER_MECH()
if(not cipher.check_is_RPI()): # check plataform
    exit()
if(not cipher.check_KEY()):
    exit()
# --- get auxiliar paths
os.environ["TEMP_FOLDER"] = os.path.normpath(os.path.join(ROOT, "../temp"))
os.environ["REPORT_FOLDER"] = os.path.normpath(os.path.join(ROOT, "../report"))
utils_for_files.reset_folder(os.environ["TEMP_FOLDER"], just_content=True)
utils_for_files.assert_folder(os.environ["REPORT_FOLDER"])
# --- get file root path
source_path = "../DATA/img_people/many_faces_1.jpg"
source_path = "../DATA/img_people/faces_360_1.jpg"
source_path = "../DATA/classroom.mp4"
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
# --- ---- Config YOLO
yolo_shape = 'v3-tiny'
yolo_shape = 'v4-crodhuman-tiny'
coco_names_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/coco.names"
if(yolo_shape=='v3-tiny'):
    yolo_weight_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/yolov3-tiny.weights"
    yolo_config_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/yolov3-tiny.cfg"
elif(yolo_shape=='v4-crodhuman-tiny'):
    yolo_weight_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/crowdhuman/yolov4-tiny-crowdhuman-416x416_best.weights"
    yolo_config_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/crowdhuman/yolov4-tiny-crowdhuman-416x416.cfg"
else:
    yolo_weight_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/yolov3.weights"
    yolo_config_file = "/home/work/img_recognition/count_eye_contact_with_camera_video/yolo/models/yolov3.cfg"
assert os.path.isfile(coco_names_file), 'No neural model parametrization coco file found .names'
assert os.path.isfile(yolo_config_file), 'No neural model configuration file found .cfg'
assert os.path.isfile(yolo_weight_file), 'No neural model hiperparameters file found .weights'
# --- ---- MECH'S
source_mech = utils_for_video.video_mech(
    source=source_path, 
    process_at_fps=1.0, 
    temp_folder=os.environ["TEMP_FOLDER"],
)
yolo_mech = utils_for_yolo.Yolo_mech(
    resize_size=512, 
    confidence=0.1, 
    threshold=0.1, 
    max_overlap=1.0,
    only_humans=False, 
    do_good_boxes=True, 
    do_new_boxes=True, 
    new_boxes_discriminant=0.025,
    coco_names_file = coco_names_file,
    yolo_weight_file = yolo_weight_file,
    yolo_config_file = yolo_config_file,
    temp_folder=os.environ["TEMP_FOLDER"],
)
# face_mech = utils_get_faces.DETECT_FACES(
#     temp_folder=os.environ["TEMP_FOLDER"], 
#     face_detector_method='hog', # hog/cnn 
#     lm_predictor_path=landmark_path,
#     cnn_path=cnn_path,
#     reset_env=True
# )
blur_mech = utils_for_blur.BLUR_MECH(
    blur_factor=(50,50),
    updown_scale=0.25,
    temp_folder=os.environ["TEMP_FOLDER"],
)
counting_mech = utils_for_counting.COUNTER_MECH(
    report_folder=os.environ["REPORT_FOLDER"],
    report_filename='REPORT_FILE.txt', 
    file_label='COUNTING REPORT',
)
past_img_path = None
idx_counter = 0
reset_temp_folder_prob = 0.05
while(True):
    if(reset_temp_folder_prob is not None and reset_temp_folder_prob!= 0 and random.randint(0,100)<=reset_temp_folder_prob*100):
        utils_for_files.reset_folder(os.environ["TEMP_FOLDER"], just_content=True)
    s_t = time()
    success = source_mech.get_frame(idx=idx_counter)
    if(not success):
        print('>> END: not more frames')
        break
    img_path, image_id = source_mech.save_c_image()
    print('---- ---- NEW IMG UUID:', image_id)
    print(">> RESULTS: enlapsed time (get frame): {}".format(time()-s_t))
    s_t = time()
    if(past_img_path is not None):
        blur_mech.blur(
            pastImg_path=past_img_path,
            actImg_path=img_path,
        )
        act_img_path = blur_mech.save_img(img_id=image_id)
    else:
        act_img_path = img_path
    print(">> RESULTS: enlapsed time (blur frame): {}".format(time()-s_t))
    # # --- ---- YOLO - TINY
    s_t = time()
    yolo_mech.process_img(imgfile=act_img_path, image_id=image_id, draw_box=True)
    yolo_mech.save_img(print_flag=False)
    print(">> RESULTS: enlapsed time (process_img): {}".format(time()-s_t))
    # s_t = time()
    # yolo_mech.print_img(out_path='/yolo_output.png')
    # print(">> RESULTS: enlapsed time (print_img): {}".format(time()-s_t))
    # s_t = time()
    # yolo_mech.segment_boxes_imgs()
    # print(">> RESULTS: enlapsed time (segment_boxes): {}".format(time()-s_t))
    # input("yolo finish, stop...")
    # # --- ---- object candidates
    # # utils_for_bodies.object_candidates(image_file=act_img_path, temp_folder=os.environ["TEMP_FOLDER"])
    # # input("object cantidates finish, stop...")
    # # --- ----
    # # --- make faces class
    # # --- get image
    # t_start = time()
    # face_mech.load_image_file(img_path_=act_img_path)
    # print("execution time to load_image: {}".format(time()-t_start))
    # # --- get faces
    # t_start = time()
    # face_mech.get_faces(upsample_factor=2)
    # print("execution time to get_faces (2): {}".format(time()-t_start))
    # # --- get face locations
    # t_start = time()
    # face_mech.face_locations_to_pixel_corners(print_flag = False, save_flag = True)
    # print("execution time to locate_face_pixels: {}".format(time()-t_start))
    # # --- get face landmarks
    # t_start = time()
    # face_mech.get_land_marks()
    # print("execution time to get_land_marks: {}".format(time()-t_start))
    # # --- final report
    # print("total execution time: {}".format(time()-t0_start))
    ## ENDING CYCLE
    new_people = len([1 for _ in yolo_mech.good_boxes if _['class'] == 1])
    new_faces = len([1 for _ in yolo_mech.good_boxes if _['class'] == 0])
    counting_mech.update_report(_data='person:#{}, atention:#{},'.format(new_people, new_faces))
    past_img_path = img_path
    idx_counter += 1
    # input("STOP #n!\n")
    dlib.hit_enter_to_continue()

assert False, 'laking try except'