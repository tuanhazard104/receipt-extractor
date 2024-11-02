""" File detector.py
    The functions for pre-processing, detect form, detect table, detect line
"""
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import datetime
import time

from arshot.services.table_detector.detector import detect_form
from arshot.utility.imageproc.image import four_point_transform
from arshot.services.table_detector.line_utils import *

PAD = 50
def grid_histogram_binary(img, overlap_ratio=0.1):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img = img

    bin_img = np.zeros_like(gray_img)
    img_H, img_W = gray_img.shape[:2]

    num_height_parts = 30
    num_width_parts = 40
    height_part = int(img_H / num_height_parts)
    width_part = int(img_W / num_width_parts)
    height_overlap = int(height_part * overlap_ratio)
    width_overlap = int(width_part * overlap_ratio)

    currect_height_idx = 0

    while (currect_height_idx < img_H):
        currect_width_idx = 0
        start_y = currect_height_idx
        end_y = currect_height_idx + height_part

        while (currect_width_idx < img_W):
            start_x = currect_width_idx
            end_x = currect_width_idx + width_part

            roi = gray_img[start_y:end_y, start_x:end_x]
            roi_out = bin_img[start_y:end_y, start_x:end_x]

            hist_full = cv2.calcHist([roi],[0],None,[100], [0,256])

            center_background = int(np.argmax(hist_full) * 255.0 / 100)

            roi_out[roi >= center_background - 3] = 255

            currect_width_idx += width_part - width_overlap

        currect_height_idx += height_part - height_overlap

    return (255 - bin_img).astype(np.uint8)
def binary_image_hist(image, pad=50):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_padded = cv2.copyMakeBorder(image_gray, 0, 0, 0, pad, cv2.BORDER_CONSTANT, None, value = 0)
    image_gray_padded[:, -pad:] = image_gray[:, :pad]

    # Histogram-based Threshold
    # bin_grid_hist_img = line_detection.utils.grid_histogram_binary(image_gray_padded)
    bin_grid_hist_img = grid_histogram_binary(image_gray_padded)
    ### try connected components
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_grid_hist_img, 4, cv2.CV_32S)
    # bin_grid_hist_img = labels

    # _, bin_grid_hist_img = cv2.threshold(bin_grid_hist_img,20,255,0)
    
    # # Adapt method
    # image_inv = line_detection.utils.invert_img(image_gray)
    # th, bin_img = line_detection.utils.cvt2binary(image_inv, low_th=128, up_th=255)

    return bin_grid_hist_img

def get_table(image_bgr):
    ### get_edge_image
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    
    kernel1 = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ], np.uint8)
    
    img_edge = cv2.dilate(img_edge, kernel1, iterations=2)
    img_edge = cv2.erode(img_edge, kernel1, iterations=2)
    
    ### find max contours
    contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    max_contour = contours[-1] if len(contours) else []
    
    if len(max_contour):
        return cv2.boundingRect(max_contour)
    return []
        

def remove_text(bin_grid_hist_img, width_range=(10,100), w_per_h=0.5):
    h,w = bin_grid_hist_img.shape[:2]
    # img_max_contour = np.zeros((h,w,3))
    img_max_contour = bin_grid_hist_img.copy()
    contours, hierarchy = cv2.findContours(bin_grid_hist_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    # print('Len contours:', len(contours), bin_grid_hist_img.shape)

    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        # if float(area) > float(400):
        if float(w) > width_range[1] or float(w) < width_range[0]:
        # if not (w_per_h < w/h < 1 + w_per_h and 12 < float(h) < 80):
            # cv2.drawContours(img_max_contour, [contour], -1, color=(255,255,255), thickness=cv2.FILLED)
            continue
        else:
            # continue
            # print(x,y,w,h)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # color = tuple(np.random.randint(100, 250, [3]).astype(int).tolist())
            cv2.drawContours(img_max_contour, [box], -1, color=(0,0,0), thickness=cv2.FILLED)
    return img_max_contour

class LineDetection2(object):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(0)
        
    def detect(self, img):
        ### 1.  pre-process
        ## `1.1 detect form
        rect = detect_form(img)
        img_warped = four_point_transform(img,rect) # remove background and warped
        self.img_org = img_warped.copy()
        H, W = self.img_org.shape[:2]
        topbot0 = [0, H]
        ##  1.2 binary image 
        img_bin = binary_image_hist(img_warped) # get binary image

        # print(">>>", img_bin.shape, img_warped.shape)
        ### 2.  extract table
        ##  2.1 find max contour
        table = get_table(img_warped)
        if table:
            x, y, w, h = table
            img_bin_zeros = np.zeros_like(img_bin)
            img_bin_zeros[y:y+h, x:x+w] = img_bin[y:y+h, x:x+w]
            img_bin = img_bin_zeros.copy()
            topbot0 = [y, y+h]
        ##  2.2 remove text
        img_bin = remove_text(img_bin) # remove text noise (issue: '1' cannot removed)
        ### try to erode
        img_bin = cv2.dilate(img_bin, (1,1), iterations=2)
        
        ### 3. Vertical line detection
        lines = self.lsd.detect(img_bin)[0] # Position 0 of the returned tuple are the detected lines
        # lines = cv2.HoughLinesP(img_bin, 1, np.pi/180, 30, np.array([]), 30, 2)
        vlines = get_line_by_angle(lines, 90) # get vertical lines
        
        ### 4. line grouping
        group_lines = get_groupping_lines(vlines, thresh=6, vertical=True)

        ### 5. remove noises
        groups_correct, groups_noise, gr_lengths  = remove_noise(group_lines.copy())

        ### 6. determine top an bottom of the table
        topbot, topbot2 = get_top_bot(groups_correct)
        for i in range(len(topbot)):
            topbot[i] = topbot[i] if topbot[i] is not None else topbot0[i]
        for i in range(len(topbot2)):
            topbot2[i] = topbot2[i] if topbot2[i] is not None else topbot0[i]
            
        ### revert correct group from noises here
        # groups_revterted = revert_group_from_noise(groups_noise=groups_noise, topbot=topbot)
        # groups_correct += groups_revterted
        groups_revterted = []
        
        ### 7. Final remove noise and correct, refine line from group
        groups_correct, _ = remove_noise2(groups_correct.copy(), topbot=topbot)
        # print('#05-remove noise&topbot:', time.time()-t1)        
        vlines = get_correct_lines_from_group(groups_correct, topbot=topbot, vertical=True)
        vlines = remove_duplicate_line(vlines, thresh=5)
        # print('#06-get_correct_lines_from_group:', time.time()-t1)
        return group_lines, groups_correct, groups_revterted, topbot2, vlines, gr_lengths, img_bin ## debug
        # return vlines
        
            

    
if __name__ == "__main__":
    line_detector = LineDetection2()
    image_paths = ['test_imageset_1/vertical/2023-08-24-15-48-59.jpg']
    for image_path in tqdm(image_paths):
        image_name = image_path.split('/')[-1]
        image_type = image_path.split('/')[-2]
        image = cv2.imread(image_path)
        vlines_corrected = line_detector.detect(image)
        
        
        # group_lines, group_lines_after_remove_noise, groups_revterted, topbot, vlines, gr_lengths, img_bin = line_detector.detect(image)
        
