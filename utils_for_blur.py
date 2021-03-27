import cv2
import numpy as np
from time import time

class BLUR_MECH:
    def __init__(self, blur_factor=(50,50), temp_folder='/temp/'):
        self.blur_factor = blur_factor
        self.temp_folder = temp_folder
    def blur(self, pastImg_path, actImg_path):
        self.img1 = cv2.imread(pastImg_path,0)
        self.img2 = cv2.imread(actImg_path,0)
        self.img2_color = cv2.imread(actImg_path)
        img3 = cv2.subtract(self.img1, self.img2)
        # Subtract the 2 image to get the difference region
        # Make it smaller to speed up everything and easier to cluster
        small_img = cv2.resize(img3,(0,0),fx = 0.25, fy = 0.25)
        # Morphological close process to cluster nearby objects
        fat_img = cv2.dilate(small_img, None,iterations = 3)
        fat_img = cv2.erode(fat_img, None,iterations = 3)
        fat_img = cv2.dilate(fat_img, None,iterations = 3)
        fat_img = cv2.erode(fat_img, None,iterations = 3)
        # Threshold strong signals
        _, bin_img = cv2.threshold(fat_img,20,255,cv2.THRESH_BINARY)
        # Analyse connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
        # Cluster all the intersected bounding box together
        rsmall, csmall = np.shape(small_img)
        new_img1 = np.zeros((rsmall, csmall), dtype=np.uint8)
        self.fill_rects(new_img1,stats)
        # Analyse New connected components to get final regions
        num_labels_new, labels_new, stats_new, centroids_new = cv2.connectedComponentsWithStats(new_img1)
        self.diff_shapes = np.uint8(200*labels/np.max(labels))
        self.diff_rectangles = np.uint8(200*labels_new/np.max(labels_new))
        # binarize
        self.diff_shapes[self.diff_shapes > 0] = 255
        self.diff_rectangles[self.diff_rectangles > 0] = 255
        self.mask = cv2.resize(self.diff_rectangles, (0,0), fx = 4.0, fy = 4.0)
        self.blurImg = cv2.blur(self.img2_color, self.blur_factor)
        aux_a = cv2.bitwise_and(self.img2_color, self.img2_color, mask=self.mask)
        aux_b = cv2.bitwise_and(self.blurImg, self.blurImg, mask=cv2.bitwise_not(self.mask))
        self.outputImg = cv2.add(aux_a, aux_b)
    # Function to fill all the bounding box
    def fill_rects(self, image, stats):
        for i,stat in enumerate(stats):
            if i > 0:
                p1 = (stat[0],stat[1])
                p2 = (stat[0] + stat[2],stat[1] + stat[3])
                cv2.rectangle(image,p1,p2,255,-1)
    def save_img(self, img_id):
        # cv2.imwrite('mask image.png',self.mask)
        aux_name = self.temp_folder+'/'+str(img_id)+'__blur.jpg'
        cv2.imwrite(aux_name,self.outputImg)
        return aux_name
    def print_imgs(self):
        cv2.imshow('mask image',self.mask)
        cv2.imshow('blurred image',self.blurImg)
        cv2.imshow("outputImg:",self.outputImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=='__main__':
    blur_mech = BLUR_MECH(
        blur_factor=(100,100),
        temp_folder='./',
    )
    blur_mech.blur(
        pastImg_path='img_1.jpg',
        actImg_path='img_2.jpg',
    )
    blur_mech.save_img(img_id='outputImg')
    # blur_mech.print_imgs()

