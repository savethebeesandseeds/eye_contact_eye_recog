# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import cv2
import numpy as np
from time import time

class BLUR_MECH:
    def __init__(self, blur_factor=(50,50), updown_scale=0.05, temp_folder='/temp/'):
        self.blur_factor = blur_factor
        self.updown_scale = updown_scale
        self.temp_folder = temp_folder
    def blur(self, pastImg_path, actImg_path):
        self.img1 = cv2.imread(pastImg_path,0)
        self.img2 = cv2.imread(actImg_path,0)
        self.img2_color = cv2.imread(actImg_path)
        img3 = cv2.subtract(self.img1, self.img2)
        # Subtract the 2 image to get the difference region
        # Make it smaller to speed up everything and easier to cluster
        small_img = cv2.resize(img3,(0,0),fx = self.updown_scale, fy = self.updown_scale)
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
        self.mask = cv2.resize(self.diff_rectangles, (0,0), fx = 1/self.updown_scale, fy = 1/self.updown_scale)
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


    # # # # USAGE
    # # # # python image_diff.py --first images/original_01.png --second images/modified_01.png

    # # # # import the necessary packages
    # # # from skimage.measure import compare_ssim
    
    # # # # construct the argument parse and parse the arguments
    # # # ap = argparse.ArgumentParser()
    # # # ap.add_argument("-f", "--first", required=True,
    # # #     help="first input image")
    # # # ap.add_argument("-s", "--second", required=True,
    # # #     help="second")
    # # # args = vars(ap.parse_args())

    # # # # load the two input images
    # # # imageA = cv2.imread(args["first"])
    # # # imageB = cv2.imread(args["second"])

    # # # # convert the images to grayscale
    # # # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # # # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # # # # compute the Structural Similarity Index (SSIM) between the two
    # # # # images, ensuring that the difference image is returned
    # # # (score, diff) = compare_ssim(grayA, grayB, full=True)
    # # # diff = (diff * 255).astype("uint8")
    # # # print("SSIM: {}".format(score))

    # # # # threshold the difference image, followed by finding contours to
    # # # # obtain the regions of the two input images that differ
    # # # thresh = cv2.threshold(diff, 0, 255,
    # # #     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # # # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # # #     cv2.CHAIN_APPROX_SIMPLE)
    # # # cnts = cnts[0] #if imutils.is_cv2() else cnts[1]

    # # # # loop over the contours
    # # # for c in cnts:
    # # #     # compute the bounding box of the contour and then draw the
    # # #     # bounding box on both input images to represent where the two
    # # #     # images differ
    # # #     (x, y, w, h) = cv2.boundingRect(c)
    # # #     cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # # #     cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # # # # show the output images
    # # # cv2.imshow("Original", imageA)
    # # # cv2.imshow("Modified", imageB)
    # # # cv2.imshow("Diff", diff)
    # # # cv2.imshow("Thresh", thresh)
    # # # cv2.waitKey(0)