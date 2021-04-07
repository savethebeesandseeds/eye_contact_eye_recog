
# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
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

import os
import sys
import argparse
import configparser
import logging
from datetime import datetime

# --- --- set the licence datetime
max_date = datetime(2021, 4, 15)
# --- ---
class ConfigLogger(object):
    def __init__(self, log):
        self.__log = log
    def __call__(self, config):
        self.__log.info("Config:")
        config.write(self)
    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)
# --------
# --- Configure Initial Variables
config_file = './defaultconfig.cfg'
# --- Configure Enviroment
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), config_file).replace("./","")
config = configparser.ConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_file))
# --- get file root path
ROOT = os.path.dirname(os.path.abspath(__file__))
# ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--temporal_folder","--temp",     type=lambda s:os.path.normpath(os.path.join(ROOT, str(s).replace("\"",""))),   default=config.get("DEFAULT","TEMP"),       help="Set path to temporal folder.")
    parser.add_argument("--report_folder","--report",     type=lambda s:os.path.normpath(os.path.join(ROOT, str(s).replace("\"",""))),   default=config.get("DEFAULT","REPORT"),       help="Set path for report folder.")
    parser.add_argument("--log_folder","--log",     type=lambda s:os.path.normpath(os.path.join(ROOT, str(s).replace("\"",""))),   default=config.get("DEFAULT","LOG"),       help="Set path for log folder.")
    
    parser.add_argument("--inflation_bodies",     type=float,   default=config.get("DEFAULT","INFLATION_BODIES"),       help="Set rate of inflation for bodies, for calibration porpouses; where INFLATION_BODIES==0.5 means to count just half of the detected bodies and INFLATION_BODIES==2.0 means to double the counting of bodies.")
    parser.add_argument("--inflation_faces",     type=float,   default=config.get("DEFAULT","INFLATION_FACES"),       help="Set rate of inflation for faces, for calibration porpouses; where INFLATION_FACES==0.5 means to count just half of the detected faces and INFLATION_FACES==2.0 means to double the counting of faces.")

    parser.add_argument("--source_path",     type=lambda s:os.path.normpath(os.path.join(ROOT, str(s).replace("\"",""))),   default=config.get("DEFAULT","SOURCE_PATH"), help="Set a path of video as a source, or to set hardware device type: -CAM-")
    
    parser.add_argument("--allow_temp_folder_size_overflow", "--temp_overflow",  type=lambda s: s.lower().strip() in ['-true-', 'true', 'si', 'yes', '1'],   default="false", help="Set -false- to reset temp fodler every 20 images, so that temp folder size is finite.")
    
    args = vars(parser.parse_args())
    utils_for_files.assert_folder(args['temporal_folder'])
    utils_for_files.assert_folder(args['report_folder'])
    utils_for_files.assert_folder(args['log_folder'])

    logfilename = args["log_folder"] + "/{}_logfile.log".format(datetime.utcnow().replace(tzinfo=None, microsecond=0).strftime('%m_%d_%Y'))
    utils_for_files.assert_file(logfilename)
    logging.basicConfig(filename=logfilename, level=logging.INFO)
    config_logger = ConfigLogger(logging)
    config_logger(config)

t0_start = time()
# Download paths for yolo models
# https://drive.google.com/file/d/1Vng1GL9mJf06z5gcahtSsinzrXajqmls/view
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
# --- get auxiliar paths
os.environ["TEMP_FOLDER"] = args['temporal_folder']
os.environ["REPORT_FOLDER"] = args['report_folder']
# --- system security check
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
print(f"{bcolors.HEADER}Source of Demo, valid until April 15/12021. {bcolors.ENDC}")
print(f"{bcolors.OKCYAN}\n\t.developed by:\n\t\t{bcolors.OKGREEN}contact@waajacu.com\n\t{bcolors.OKCYAN}.stable version:\n\t\t{bcolors.OKGREEN}1.0.{bcolors.ENDC}")
print(f"{bcolors.OKCYAN}\t.running file:\n\t\t{bcolors.OKGREEN}{__file__}\n\t{bcolors.OKCYAN}.temporal folder:\n\t\t{bcolors.OKGREEN}{os.path.normpath(args['temporal_folder'])}\n\t{bcolors.OKCYAN}.report folder:\n\t\t{bcolors.OKGREEN}{os.path.normpath(args['report_folder'])}\n\t{bcolors.OKCYAN}.configuration file:\n\t\t{bcolors.OKGREEN}{config_file}{bcolors.ENDC}\n")

cipher = utils_for_cipher.CIPHER_MECH(max_date=max_date)
if(not cipher.check_all_is_well(fast=False)):
    exit()
# reset folders and assert files
utils_for_files.reset_folder(os.environ["TEMP_FOLDER"], just_content=True)
# --- cnn path
# cnn_path = os.path.normpath(os.path.join(ROOT, '../dlib-19.21/download_from_source/mmod_human_face_detector.dat'))
# --- get file for landmark detection
# landmark_path = "../dlib-19.21/download_from_source/shape_predictor_68_face_landmarks.dat"
# landmark_path = os.path.normpath(os.path.join(ROOT, landmark_path))
# --- print paths
# print("Proyect ROOT path: {}".format(ROOT))
# print("Proyect TEMP folder: {}".format(os.environ["TEMP_FOLDER"]))
# print("SOURCE path: {}".format(source_path))
# --- ---- Config YOLO
yolo_shape = 'v3-tiny'
yolo_shape = 'v4-crodhuman-tiny'
yolo_shape = 'product'
regularize_path = (lambda s: os.path.normpath(os.path.join(ROOT, str(s))))
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
coco_names_file = regularize_path("yolo/models/coco.names")
if(yolo_shape=='v3-tiny'):
    yolo_weight_file = regularize_path("yolo/models/yolov3-tiny.weights")
    yolo_config_file = regularize_path("yolo/models/yolov3-tiny.cfg")
elif(yolo_shape=='v4-crodhuman-tiny'):
    yolo_weight_file = regularize_path("yolo/models/crowdhuman/yolov4-tiny-crowdhuman-416x416_best.weights")
    yolo_config_file = regularize_path("yolo/models/crowdhuman/yolov4-tiny-crowdhuman-416x416.cfg")
elif(yolo_shape=='product'):
    coco_names_file = regularize_path("src/network.names")
    yolo_weight_file = regularize_path("src/network.weights")
    yolo_config_file = regularize_path("src/network.cfg")
else:
    yolo_weight_file = regularize_path("yolo/models/yolov3.weights")
    yolo_config_file = regularize_path("yolo/models/yolov3.cfg")
# assert os.path.isfile(coco_names_file), 'No neural model parametrization coco file found .names'
# assert os.path.isfile(yolo_config_file), 'No neural model configuration file found .cfg'
# assert os.path.isfile(yolo_weight_file), 'No neural model hiperparameters file found .weights'
# --- ---- MECH'S
source_mech = utils_for_video.video_mech(
    source=args['source_path'], 
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
img_path = None
past_img_path = None
idx_counter = 0
reset_temp_folder_prob = 0.05
while(True):
    s_t = time()
    success = source_mech.get_frame(idx=idx_counter)
    img_path, image_id = source_mech.save_c_image()
    try:
        if(not success):
            print(f'>> {bcolors.WARNING}END: not more frames{bcolors.ENDC}')
            break
        print('---- ---- NEW IMG UUID:', image_id)
    except:
        print(f">> {bcolors.FAIL} RESULTS: source adquisition failed...{bcolors.ENDC}")
    print(f">> {bcolors.WARNING} RESULTS: {bcolors.ENDC}enlapsed time (get frame): {time()-s_t}{bcolors.ENDC}")
    s_t = time()
    if(past_img_path is not None):
        try:
            blur_mech.blur(
                pastImg_path=past_img_path,
                actImg_path=img_path,
            )
            act_img_path = blur_mech.save_img(img_id=image_id)
        except:
            print(f">> {bcolors.FAIL} RESULTS: blurring failed...{bcolors.ENDC}")
    else:
        act_img_path = img_path
    print(f">> {bcolors.WARNING} RESULTS: {bcolors.ENDC}enlapsed time (blur frame): {time()-s_t}{bcolors.ENDC}")
    # # --- ---- YOLO - TINY
    s_t = time()
    try:
        yolo_mech.process_img(imgfile=act_img_path, image_id=image_id, draw_box=True)
    except:
        print(f">> {bcolors.FAIL} RESULTS: Process image failed:{bcolors.ENDC}")
    yolo_mech.save_img(print_flag=False)
    print(f">> {bcolors.WARNING} RESULTS: {bcolors.ENDC}enlapsed time (process_img): {time()-s_t}{bcolors.ENDC}")
    # s_t = time()
    # yolo_mech.print_img(out_path='/yolo_output.png')
    # print(f">> {bcolors.WARNING} RESULTS: {bcolors.ENDC}enlapsed time (print_img): {time()-s_t}{bcolors.ENDC}")
    # s_t = time()
    # yolo_mech.segment_boxes_imgs()
    # print(f">> {bcolors.WARNING} RESULTS: {bcolors.ENDC}enlapsed time (segment_boxes): {time()-s_t}{bcolors.ENDC}")
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
    if(args['inflation_bodies'] <= 1.0):
        new_bodies = len([1 for _ in yolo_mech.good_boxes if _['class'] == 1 and random.randint(0,100) <= args['inflation_bodies']*100.0])
    else:
        new_bodies = len([1 for _ in yolo_mech.good_boxes if _['class'] == 1])
        new_bodies += len([1 for _ in yolo_mech.good_boxes if _['class'] == 1 and random.randint(0,100) <= (args['inflation_bodies']-1.0)*100.0])
    if(args['inflation_faces'] <= 1.0):
        new_faces = len([1 for _ in yolo_mech.good_boxes if _['class'] == 0 and random.randint(0,100) <= args['inflation_faces']*100.0])
    else:
        new_faces = len([1 for _ in yolo_mech.good_boxes if _['class'] == 0])
        new_faces += len([1 for _ in yolo_mech.good_boxes if _['class'] == 0 and random.randint(0,100) <= (args['inflation_faces']-1.0)*100.0])
    counting_mech.update_report(_data='bodies:#{}, faces:#{},'.format(new_bodies, new_faces))
    past_img_path = img_path
    idx_counter += 1
    # input("STOP #n!\n")
    # dlib.hit_enter_to_continue()
    if(not(args['allow_temp_folder_size_overflow']) and reset_temp_folder_prob is not None and reset_temp_folder_prob!= 0 and random.randint(0,100)<=reset_temp_folder_prob*100):
        utils_for_files.reset_folder(os.environ["TEMP_FOLDER"], just_content=True)

print(">> PROGRAM ENDED...")
