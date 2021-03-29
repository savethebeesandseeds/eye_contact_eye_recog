# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import sys
import cv2
import uuid
import copy
import numpy as np
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolo'))
import yolo_ocv_utils

# a box is: (x=b[0]-b[2]/2, y = b[1]-b[3]/2, w, h, extra, precistion, class)
class Yolo_mech:
    def __init__(
            self, 
            resize_size=512, 
            confidence=0.5, 
            threshold=0.5, 
            max_overlap=0.01,
            only_humans=False, 
            do_good_boxes=True, 
            do_new_boxes=True, 
            new_boxes_discriminant=0.1, 
            coco_names_file = "./models/coco.names",
            yolo_weight_file = "./models/yolov3-tiny.weights",
            yolo_config_file = "./models/yolov3-tiny.cfg",
            temp_folder='./temp/'
            ):
        self.resize_size = resize_size
        self.original_size = [resize_size, resize_size] # (width, height)
        self.only_humans = only_humans
        self.do_good_boxes = do_good_boxes
        self.do_new_boxes = do_new_boxes
        self.max_overlap = max_overlap
        self.new_boxes_discriminant = new_boxes_discriminant
        self.temp_folder = temp_folder
        self.past_boxes = None
        self.good_boxes = None
        self.past_box_aux = None
        # identification parameters
        self.set_paramters(confidence=confidence,threshold=threshold)
        # read coco object names
        self.LABELS = open(coco_names_file).read().strip().split("\n")
        # assign rondom colours to the corresponding class labels
        np.random.seed(45)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
        # read YOLO network model
        self.net = cv2.dnn.readNetFromDarknet(yolo_config_file, yolo_weight_file)
    def set_paramters(self, confidence, threshold):
        self.confidence = confidence
        self.threshold = threshold
    def process_img(self, imgfile, image_id=None, draw_box=False):
        # self.img_org = Image.open(imgfile).convert('RGB')
        # self.original_size[0], self.original_size[1] = self.img_org.size
        # self.img_resized = self.img_org.resize((self.resize_size, self.resize_size))
        # self.img_torch = yolo_utils.image2torch(self.img_resized)
        if(image_id is None):
            self.img_id = str(uuid.uuid1())
        else:
            self.img_id = image_id
        if(self.good_boxes is not None):
            self.past_boxes = copy.deepcopy(self.good_boxes)
            if(self.past_box_aux is not None):
                for pbx in self.past_box_aux:
                    self.past_boxes.append(pbx)
        # # s_t = time()
        # # output_all = self.model(self.img_torch)
        # # print("enlapsed time (output_all): {}".format(time()-s_t))
        self.detected_image, self.detected_boxes = yolo_ocv_utils.yolo_object_detection(imgfile, self.net, self.confidence, self.threshold, self.LABELS, self.COLORS)
        if(self.do_good_boxes):
            self.good_boxes = yolo_ocv_utils.get_good_boxes(all_data=self.detected_boxes, only_humans=self.only_humans, min_precision=self.threshold, max_overlap=self.max_overlap)
        else:
            self.good_boxes = self.detected_boxes
        if(self.do_new_boxes and self.past_boxes is not None):
            self.good_boxes, self.past_box_aux = yolo_ocv_utils.get_only_new_boxes(all_data=self.good_boxes, all_data_past=self.past_boxes, discriminant_factor=self.new_boxes_discriminant)
        # print("self.detected_boxes:", self.detected_boxes)
        # print("self.good_boxes:", self.good_boxes)
        # Draw boxes on the image
        if(draw_box):
            if(len(self.good_boxes)!=0):
                self.detected_image, text_list = yolo_ocv_utils.draw_boxes(self.detected_image, detected_boxes=self.good_boxes, labels=self.LABELS, colors=self.COLORS)
        else:
            text_list = ''
        # print("{} : {}".format(imgfile, text_list), flush=True)
        
    def sub_image(self, aux_box):
        # x, y = (b[0]-b[2]/2, b[1]-b[3]/2)
        area = (
            (aux_box[0] - aux_box[2]/2) * self.original_size[0], 
            (aux_box[1] - aux_box[3]/2) * self.original_size[1], 
            (aux_box[0] + aux_box[2]/2) * self.original_size[0], 
            (aux_box[1] + aux_box[3]/2) * self.original_size[1],
            )
        return self.img_org.crop(area)

    def segment_boxes_imgs(self):
        self.segments = []
        for idx, gb in enumerate(self.good_boxes):
            aux_name = self.temp_folder+'/'+self.img_id+'__{}.png'.format(idx)
            # self.sub_image(self.resize_box_to_original_box(gb)).save(aux_name)
            # self.sub_image(gb).save(aux_name)
            self.segments.append(self.sub_image(gb))
          
    def save_img(self, print_flag=False):
        out_path =  "{}/{}_full.jpg".format(self.temp_folder,self.img_id)
        if(print_flag):
            cv2.imshow("{}".format(out_path, self.detected_image), self.detected_image)
            cv2.waitKey(0)
        cv2.imwrite(out_path, self.detected_image)
