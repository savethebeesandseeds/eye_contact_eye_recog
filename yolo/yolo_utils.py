import PIL
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import time
import math
import torch
import numpy as np
from matplotlib import pyplot as plt, rcParams, animation, patches, patheffects



def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    confs = [(1-b[4]) for b in boxes]
    sorted_idx = np.argsort(confs)
    out_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sorted_idx[i]]
        if confs[i] > -1:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                if confs[j] > -1:
                    box_j = boxes[sorted_idx[j]]
                    if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        confs[j] = -1
    return out_boxes


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    model.eval()
    img = image2torch(img)
    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    all_boxes = model.predict_img(img)[0]
    boxes = nms(all_boxes, nms_thresh)
    return boxes


def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea/uarea)


def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea


###################################################################
## Plotting helpers

# e.g. plot_multi_detections(img_tensor, model.predict_img(img_tensor))
def plot_multi_detections(imgs, results, figsize=None, **kwargs):
    if not figsize:
        figsize = (12, min(math.ceil(len(imgs)/3)*4, 30))
    _, axes = plt.subplots(math.ceil(len(imgs)/3), 3, figsize=figsize)
    
    if type(imgs) == np.ndarray and len(imgs.shape) == 4:
        imgs = [imgs]
    
    classes = []
    boxes = []
    extras = []
    for r in results:
        res = np.array([[float(b) for b in arr] for arr in r])
        if len(res) > 0:
            cla = res[:, -1].astype(int)
            b = res[:, 0:4]
            e = ["{:.2f} ({:.2f})".format(float(y[4]), float(y[5])) for y in res]
        else:
            cla, b, e = [], [], []
        classes.append(cla)
        boxes.append(b)
        extras.append(e)

    for j, ax in enumerate(axes.flat):
        if j >= len(imgs):
            #break
            plt.delaxes(ax)
        else:
            plot_img_boxes(imgs[j], boxes[j], classes[j], extras[j], plt_ax=ax, **kwargs)
        
    plt.tight_layout()


def plot_img_detections(img, out_path, result_boxes, **kwargs):
    b = np.array(result_boxes)
    if len(b) > 0:
        classes = b[:, -1].astype(int)
        boxes = b[:, 0:4]
    else:
        classes, boxes = [], []
    extras = ["{:.2f} ({:.2f})".format(b[4], b[5]) for b in result_boxes]
    return plot_img_boxes(img, boxes, classes, out_path=out_path, extras=extras, **kwargs)


def plot_img_data(x, y, rows=2, figsize=(12, 8), **kwargs):
    _, axes = plt.subplots(rows, 3, figsize=figsize)

    for j, ax in enumerate(axes.flat):
        if j >= len(y):
            break
        targets = y[j]
        if isinstance(targets, torch.Tensor):
            targets = targets.clone().reshape(-1,5)
            classes = targets[:, 0].cpu().numpy().astype(int)
        else:
            classes = targets[:, 0].astype(int)
        plot_img_boxes(x[j], targets[:, 1:], classes, plt_ax=ax, **kwargs)
        
    plt.tight_layout()


def plot_img_boxes(img, boxes, classes, out_path=None, extras=None, plt_ax=None, figsize=None, class_names=None, real_pixels=False, box_centered=True):
    if not plt_ax:
        _, plt_ax = plt.subplots(figsize=figsize)
    colors = np.array([[0.5,0.5,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])

    if type(img) == PIL.Image.Image:
        width = img.width
        height = img.height
    elif type(img) in [torch.Tensor, np.ndarray]:
        # if len(img.shape)>3: img = img[0]
        if type(img) == torch.Tensor:
            img = img.clone().cpu().numpy()
        width = img.shape[2]
        height = img.shape[1]
        img = img.transpose(1,2,0)
        if (img < 1.01).all() and (img >= 0).all():
            img = img.clip(0, 1) # avoid "Clipping input data to the valid range" warning after tensor roundings
    else:
        raise(f"Unkown type for image: {type(img)}")

    if len(boxes) > 0 and not real_pixels:
        boxes[:, 0] *= width; boxes[:, 2] *= width
        boxes[:, 1] *= height; boxes[:, 3] *= height

    for i in range(len(boxes)):
        b, class_id = boxes[i], classes[i]
        if b[0] == 0:
            break

        color = colors[class_id%len(colors)]

        if box_centered:
            x, y = (b[0]-b[2]/2, b[1]-b[3]/2)
            w, h = (b[2], b[3])
        else:
            x, y = b[0], b[1]
            w, h = b[2], b[3]

        patch = plt_ax.add_patch(patches.Rectangle([x, y], w, h, linewidth=0.1, fill=False, edgecolor=color, lw=2))
        patch.set_path_effects([patheffects.Stroke(linewidth=0.1, foreground='black', alpha=0.5), patheffects.Normal()])

        s = class_names[class_id] if class_names else str(class_id)
        if extras:
            s += "\n"+str(extras[i])
        patch = plt_ax.text(x+2, y, s, verticalalignment='top', color=color, fontsize=6, weight=100)
        patch.set_path_effects([patheffects.Stroke(linewidth=0.1, foreground='black', alpha=0.5), patheffects.Normal()])

    if(out_path is not None):
        _ = plt_ax.imshow(img)
        plt.savefig(out_path)
    else:
        _ = plt_ax.imshow(img)

# BY WAAJACU
def get_box_coords(box, box_centered=True):
    # (x,y) is the upper left point of the rectangle (w,h)
    if box_centered:
        x, y = (box[0]-box[2]/2, box[1]-box[3]/2)
        w, h = (box[2], box[3])
    else:
        x, y = box[0], box[1]
        w, h = box[2], box[3]
    return {'x':round(x,2),'y':round(y,2),'w':round(w,2),'h':round(h,2)}
def calculate_overlap(box0, box1):
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]
    def argmin(iterable):
        return min(enumerate(iterable), key=lambda x: x[1])[0]
    assert type(box0) == dict and type(box1) == dict, 'wrong boxes definition, use get_box_coords before'
    arg_min_x = argmin([box0['x'], box1['x']])
    arg_min_y = argmin([box0['y'], box1['y']])
    if(arg_min_x == 0):
        if(box0['x'] + box0['w'] <= box1['x']):return 0
    else:
        if(box1['x'] + box1['w'] <= box0['x']):return 0
    if(arg_min_y == 0):
        if(box0['y'] + box0['h'] <= box1['y']):return 0
    else:
        if(box1['y'] + box1['h'] <= box0['y']):return 0
    if(arg_min_x == 0):
        x_overlap = (box0['x'] - box1['x'] + box0['w'])
    else:
        x_overlap = (box1['x'] - box0['x'] + box1['w'])
    if(arg_min_y == 0):
        y_overlap = (box0['y'] - box1['y'] + box0['h'])
    else:
        y_overlap = (box1['y'] - box0['y'] + box1['h'])
    assert x_overlap > 0 and y_overlap > 0
    return x_overlap * y_overlap
def get_good_boxes(all_boxes, only_humans=False, min_precision=0.0, max_overlap=1.0):
    good_boxes = []
    actual_boxes = []
    for box in sorted(all_boxes, key=lambda x: x[5], reverse=True):
        # print("x:{}, y:{}, w:{}, h:{}".format(x*size,y*size,w*size,h*size))
        if(box[5] >= min_precision and (not only_humans or box[-1]==0)): # evaluate precision
            if(len(actual_boxes)!=0):
                cb = get_box_coords(box)
                for ab in actual_boxes:
                    # print(">>> \n\t{}, \n\t{} \n\t :: overlap:{}".format(cb, ab, calculate_overlap(cb, ab)))
                    if(calculate_overlap(cb, ab) <= max_overlap):
                        ab_flag = True
                    else:
                        ab_flag = False
                        break
                if(ab_flag):
                    actual_boxes.append(cb)
                    good_boxes.append(box)
            else:
                actual_boxes.append(get_box_coords(box))
                good_boxes.append(box)
    return good_boxes