import torch
import numpy as np
from PIL import Image

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolo'))

import yolo_utils
from yolo_layer import *
from yolov3_tiny import *

from time import time

class Yolo_mech:
    def __init__(
            self, 
            isTiny=True, 
            size=512, 
            min_precision=0.0, 
            max_overlap=1.0, 
            only_humans=False, 
            state_dict='./models/yolov3_tiny_coco_01.h5'
            ):
        assert isTiny, 'regular yolo is not implemted, use tiny instead'
        if(isTiny):
            self.model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
            self.model.load_state_dict(torch.load(state_dict))
        _ = self.model.eval() # .cuda()
        self.isTiny = isTiny
        self.size = size
        self.min_precision = min_precision
        self.max_overlap = max_overlap
        self.only_humans = only_humans
    
    def process_img(self, imgfile):
        self.img_org = Image.open(imgfile).convert('RGB')
        self.img_resized = self.img_org.resize((self.size, self.size))
        self.img_torch = yolo_utils.image2torch(self.img_resized)
        # s_t = time()
        # output_all = self.model(self.img_torch)
        # print("enlapsed time (output_all): {}".format(time()-s_t))
        s_t = time()
        self.all_boxes = self.model.predict_img(self.img_torch)[0]
        print(">> RESULTS: enlapsed time (predict_img): {}".format(time()-s_t))
        s_t = time()
        self.good_boxes = yolo_utils.get_good_boxes(self.all_boxes, only_humans=self.only_humans, min_precision=self.min_precision, max_overlap=self.max_overlap)
        print(">> RESULTS: enlapsed time (filter_boxes): {}".format(time()-s_t))
    def print_img(self, out_path):
        yolo_utils.plot_img_detections(img=self.img_resized, out_path=out_path, result_boxes=self.good_boxes)

if __name__=='__main__':
    imgfile = "../DATA/img_people/people_and_dogs.jpg"
    size = 512
    min_precision = 0.9
    max_overlap = 0.01
    only_humans = True

    yolo_model = Yolo_mech(
        isTiny=True, 
        size=size, 
        min_precision=min_precision, 
        max_overlap=max_overlap, 
        only_humans=only_humans, 
        state_dict='./yolo/models/yolov3_tiny_coco_01.h5'
    )
    yolo_model.process_img(imgfile)
    yolo_model.print_img(out_path='mygraph.png')
