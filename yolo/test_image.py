import torch
import numpy as np
from PIL import Image

import yolo_utils
from yolo_layer import *
from yolov3_tiny import *

from time import time

def test_tiny(imgfile):
    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(torch.load('./models/yolov3_tiny_coco_01.h5'))
    _ = model.eval() # .cuda()

    size = 512
    min_precision = 0.9
    max_overlap = 0.01
    only_humans = False

    img_org = Image.open(imgfile).convert('RGB')
    img_resized = img_org.resize((size, size))
    img_torch = yolo_utils.image2torch(img_resized)

    # s_t = time()
    # output_all = model(img_torch)
    # print("enlapsed time (output_all): {}".format(time()-s_t))
    s_t = time()
    all_boxes = model.predict_img(img_torch)[0]
    print(">> RESULTS: enlapsed time (predict_img): {}".format(time()-s_t))
    s_t = time()
    good_boxes = yolo_utils.get_good_boxes(all_boxes, only_humans=only_humans, min_precision=min_precision, max_overlap=max_overlap)
    print(">> RESULTS: enlapsed time (filter_boxes): {}".format(time()-s_t))
    aux = yolo_utils.plot_img_detections(img=img_resized, result_boxes=good_boxes)

if __name__=='__main__':
    imgfile = "../../DATA/img_people/people_and_dogs.jpg"
    test_tiny(imgfile=imgfile)
