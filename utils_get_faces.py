import os
import uuid
from PIL import Image
import dlib

import utils_for_files

class DETECT_FACES:
    def __init__(self, temp_folder, img_path_=None, face_detector_method='hog', lm_predictor_path='shape_predictor_68_face_landmarks.dat', cnn_path='mmod_human_face_detector.dat', reset_env=False):
        if(img_path_ is not None):
            self.set_image_path(img_path_)
            self.load_image_file()
        self.temp_folder = temp_folder
        self.face_detector_method = face_detector_method
        self.lm_predictor_path = lm_predictor_path
        self.cnn_path = cnn_path
        if(reset_env):
            self.reset_env()
        
    def reset_env(self):
        self.win = dlib.image_window()
        self.win.clear_overlay()
        self.lm_predictor = dlib.shape_predictor(self.lm_predictor_path)
        if(self.face_detector_method.lower() == 'cnn'):
            self.face_detector = dlib.cnn_face_detection_model_v1(self.cnn_path)
        elif(self.face_detector_method.lower() == 'hog'):
            self.face_detector = dlib.get_frontal_face_detector()
        
    def set_image_path(self, path_):
        self.current_img_path = path_

    def load_image_file(self, img_path_):
        if(img_path_ is not None):
            self.set_image_path(path_=img_path_)
        if(utils_for_files.assert_file(self.current_img_path, assert_by_creation=False)):
            self.dlib_image = dlib.load_rgb_image(self.current_img_path)

    def get_faces(self, upsample_factor=1):
        # METHOD: 'HOG'/'cnn'
        # upsample_factor = int # The bigger the better, the bigger the slower
        if(self.face_detector_method.lower() == 'cnn'):
            self.face_det = self.face_detector(self.dlib_image, upsample_factor).rect
        elif(self.face_detector_method.lower() == 'hog'):
            self.face_det = self.face_detector(self.dlib_image, upsample_factor)
        print(">> Found #{} face(s) in this photograph.".format(len(self.face_det)))

    def face_locations_to_pixel_corners(self, print_flag=False, save_flag = False):
        self.faces_metadata = {}
        self.faces_metadata['faces_count'] = len(self.face_det)
        self.faces_metadata['faces'] = []
        # ---
        for idx, face_location in enumerate(self.face_det):
            self.faces_metadata['faces'].append({
                'id': str(uuid.uuid1()),
                'location': {
                    'top':face_location.top(), 
                    'right':face_location.right(), 
                    'bottom':face_location.bottom(), 
                    'left':face_location.left(),
                    }
                })
            # ---
            if(print_flag):
                print("A face was found located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(face_location.top(), face_location.left(), face_location.bottom(), face_location.right()))
            # ---
            self.current_pil_image = self.get_sub_images(
                top=face_location.top(), 
                right=face_location.right(), 
                bottom=face_location.bottom(), 
                left=face_location.left(),
                return_format = 'pil'
                )
            if(save_flag):
                save_img_path = self.temp_folder + '/' + self.faces_metadata['faces'][idx]['id']+'.jpg'
                self.save_pill_image(self.current_pil_image, path_ = save_img_path, print_flag = print_flag)
    
    def get_land_marks(self):
        self.win.clear_overlay()
        self.win.set_image(self.dlib_image)
        for idx, face_location in enumerate(self.face_det):
            # Get the landmarks/parts for the face in box d.
            shape = self.lm_predictor(self.dlib_image, face_location)
            print("IMG: {} :: Part 0: {}, Part 1: {} ...".format(
                idx, 
                shape.part(0),
                shape.part(1)))
            self.win.add_overlay(shape)
        self.win.add_overlay(self.face_det)

    def get_sub_images(self, top, right, bottom, left, return_format = 'array'):
        # ---
        face_image = self.dlib_image[top:bottom, left:right]
        # ---
        if(return_format == 'array'):
            return face_image
        elif(return_format == 'pil'):
            return Image.fromarray(face_image)

    def save_pill_image(self, pil_image, path_, print_flag = False):
        if(print_flag):
            print(">> saving image to file: {}".format(path_))
        pil_image.save(path_)
        