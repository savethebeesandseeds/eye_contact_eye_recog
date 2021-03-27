import os
import sys
import uuid
import copy
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolo'))

import yolo_utils
from yolo_layer import *
from yolov3_tiny import *

from time import time

# a box is: (x=b[0]-b[2]/2, y = b[1]-b[3]/2, w, h, extra, precistion, class)
class Yolo_mech:
    def __init__(
            self, 
            isTiny=True, 
            size=512, 
            min_precision=0.0, 
            max_overlap=1.0, 
            only_humans=False, 
            state_dict='./models/yolov3_tiny_coco_01.h5',
            temp_folder='./temp/'
            ):
        assert isTiny, 'regular yolo is not implemted, use tiny instead'
        if(isTiny):
            self.model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
            self.model.load_state_dict(torch.load(state_dict))
        _ = self.model.eval() # .cuda()
        self.isTiny = isTiny
        self.resize_size = size
        self.original_size = [size, size] # (width, height)
        self.min_precision = min_precision
        self.max_overlap = max_overlap
        self.only_humans = only_humans
        self.temp_folder = temp_folder
    
    def process_img(self, imgfile):
        self.img_org = Image.open(imgfile).convert('RGB')
        self.original_size[0], self.original_size[1] = self.img_org.size
        self.img_resized = self.img_org.resize((self.resize_size, self.resize_size))
        self.img_torch = yolo_utils.image2torch(self.img_resized)
        self.img_id = str(uuid.uuid1())
        # s_t = time()
        # output_all = self.model(self.img_torch)
        # print("enlapsed time (output_all): {}".format(time()-s_t))
        self.all_boxes = self.model.predict_img(self.img_torch)[0]
        self.good_boxes = yolo_utils.get_good_boxes(self.all_boxes, only_humans=self.only_humans, min_precision=self.min_precision, max_overlap=self.max_overlap)
    
    def resize_box_to_original_box(self, box):
        # print("original size: {}".format(self.original_size))
        # wh = self.original_size[0]/self.resize_size
        # hw = self.original_size[1]/self.resize_size
        # print("(2) wh:{}, hw:{}".format(wh,hw))
        wh = 1.0
        hw = 1.0
        aux_box = copy.deepcopy(box)
        aux_box[0] = aux_box[0] * wh
        aux_box[3] = aux_box[3] * wh
        aux_box[1] = aux_box[1] * hw
        aux_box[4] = aux_box[4] * hw
        return aux_box
    
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
            
    def print_img(self, out_path=None):
        if(out_path is None):
            yolo_utils.plot_img_detections(img=self.img_resized, out_path=self.temp_folder+'/'+self.img_id+'.png', result_boxes=self.good_boxes)
        else:
            yolo_utils.plot_img_detections(img=self.img_resized, out_path=self.temp_folder+out_path, result_boxes=self.good_boxes)

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
