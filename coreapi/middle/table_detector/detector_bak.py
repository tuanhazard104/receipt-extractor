""" File detector.py
    The functions for pre-processing, detect form, detect table, detect line
"""
import os
import sys
import cv2
import numpy as np
import math
from sympy import Point, Line
from sympy import Point as sympyPoint
from shapely.geometry import LineString
from shapely.geometry import Point as geometryPoint
from collections import Counter
from tqdm import tqdm
import time

sys.path.append('/hdd/tuannca/ocr/app-business-form-recognition_namtc/src')
from arshot.utility.imageproc.image import order_points, four_point_transform, find_four_corners
from arshot.services.table_extractor.line_utils import get_line_by_angle, get_groupping_lines, get_correct_lines_from_group, get_lines_by_distance, remove_duplicate_line, vLines_refined

def add_padding(img, rect, pad_size):
    h, w = img.shape[:2]
    
    box_x, box_y, box_w, box_h = rect
    xmin = np.clip(box_x - pad_size, 0, w)
    ymin = np.clip(box_y - pad_size, 0, h)
    xmax = np.clip(box_x + box_w + pad_size, 0, w)
    ymax = np.clip(box_y + box_h + pad_size, 0, h)
    return (xmin, ymin, xmax, ymax)

def detect_form(img0, thresh_value=30):
    img_copy = img0.copy()
    """ Detect the business form and do perspective transformation
    """
    # Binary image
    scale_ratio = 4
    img_resize = cv2.resize(img0, None, fx=1/scale_ratio, fy=1/scale_ratio)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    #thresh_value = np.mean(img_gray)*3/2
    _, img_bin = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
    
    kernel_size=(3,3)
    kernel = np.ones(kernel_size, np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)   # Opening the image

    # Find contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return img_copy, np.array([])

    # Find 4 corners
    max_contour = max(contours, key=cv2.contourArea)
    box = np.float32(find_four_corners(size = img_gray.shape, contour=max_contour))

    # rect = cv2.minAreaRect(max_contour)
    # box = cv2.boxPoints(rect)

    # Apply perspective transform
    box = np.int0(box) * scale_ratio
    rect = order_points(pts=box)
    img_warped = four_point_transform(img0, rect)

    return img_warped,  np.int0(rect)

def detect_table( img0):
    """ Detect the maximum table in the form
    """
    h, w = img0.shape[:2]
    img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    box = []
    i = 0
    while True:
        hull = cv2.convexHull(cnts[i])
        if cv2.contourArea(hull) / (h * w) > 0.9:
            i += 1
            continue
        peri = cv2.arcLength(hull, True)
        box = cv2.approxPolyDP(cnts[i], 0.12 * peri, True)
        break

    if len(box):
        rect = cv2.boundingRect(box) # format of each rect: x, y, w, h
        xmin, ymin, xmax, ymax = add_padding(img0, rect, pad_size=20)
        tbl_img = img0[ymin:ymax, xmin:xmax]
    else:
        tbl_img = img0.copy()

    return tbl_img

class LineDetection():
    def __init__(self):
        pass

    def get_edge_image(self, img0):
        img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img_edge = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        # img_edge = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -3)
        # kernel = np.ones((3,3),np.uint8)
        # img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        kernel1 = np.array([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ], np.uint8)
        img_edge = cv2.dilate(img_edge, kernel1, iterations=3)
        img_edge = cv2.erode(img_edge, kernel1, iterations=3)
        return img_edge

    def detect_lines(self, img_edge, vertical:bool):
        H, W = img_edge.shape[:2]
        # print(H,W) # 2160,3840
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W//400)) if W//400 > 0 else cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W//768, 1)) if W//768 > 0 else cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        if vertical:
            remove_text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W//110)) if W//110>0 else cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        else:
            remove_text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

        img_proc = img_edge.copy()
        if vertical:
            # connect thick line is detected by 2 thin lines due to canny detection
            img_proc = cv2.dilate(img_proc, hor_kernel, iterations=2)
            # img_proc = cv2.erode(img_proc, hor_kernel, iterations=5)
            
            # remove some ---- line
            img_proc = cv2.erode(img_proc, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=2)
            img_proc = cv2.dilate(img_proc, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=2)
            
            # # dilate the canny image to connect some break line (ex: ----)
            img_proc = cv2.dilate(img_proc, ver_kernel, iterations=5)
            img_proc = cv2.erode(img_proc, ver_kernel, iterations=5)

            img_proc = cv2.dilate(img_proc, hor_kernel, iterations=2)

        else:
            # # dilate the canny image to connect some break line
            img_proc = cv2.dilate(img_proc, ver_kernel, iterations=1)
            # then erode the dilated image to remove the vertical line
            img_proc = cv2.erode(img_proc, hor_kernel, iterations=2)
            img_proc = cv2.dilate(img_proc, hor_kernel, iterations=2)
            
            img_proc = cv2.dilate(img_proc, ver_kernel, iterations=1)    

        # remove text
        iterations = 1 if vertical else 3
        img_proc = cv2.erode(img_proc, remove_text_kernel, iterations=iterations)
        img_proc = cv2.dilate(img_proc, remove_text_kernel, iterations=iterations)
        
        # action to improve result
        # img_proc = cv2.dilate(img_proc, hor_kernel, iterations=iterations)
        img_proc = cv2.erode(img_proc, hor_kernel, iterations=iterations)   
        # find text contours to remove
        contours, hierarchy = cv2.findContours(img_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i,cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
            perimeter = cv2.arcLength(cnt,True)
            if vertical:
                if perimeter < (H/3):
                    img_proc = cv2.drawContours(img_proc,[cnt], 0, (0), -1) 
            else:
                if perimeter < (W/7.5):
                    img_proc = cv2.drawContours(img_proc,[cnt], 0, (0), -1)  

        # line detection
        lines = cv2.HoughLinesP(image=img_proc, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=15)
        
        # for line in lines:
        #     x1, y1, x2, y2 = line[0].astype(int)
        #     cv2.line(img_proc, (x1, y1), (x2, y2), (1), 2)
            
        ### <---debug 110923--->
        #cv2.imwrite(f"debug_{im_name}_imgproc.jpg", img_proc)

        angle = 90 if vertical else 0
        lines = get_line_by_angle(lines, angle)
        return lines

    def get_correct_lines(self, lines, thresh=4, vertical=False):
        if lines == [] or lines is None:
            return lines
        group_lines = get_groupping_lines(lines=lines, thresh=thresh, vertical=vertical)
        correct_lines = get_correct_lines_from_group(groups=group_lines, vertical=vertical)
        # correct_lines1 = correct_lines.copy()
        correct_lines = get_lines_by_distance(correct_lines)
        correct_lines2 = correct_lines.copy()
        correct_lines = remove_duplicate_line(correct_lines)
        correct_lines3 = correct_lines.copy()
        if vertical:
            correct_lines = vLines_refined(correct_lines, vertical=vertical)
        return correct_lines
    
    def one_shot(self, image, vertical=True):
        img_edge = self.get_edge_image(image)
        lines = self.detect_lines(img_edge=img_edge, vertical=vertical)
        v_lines = self.get_correct_lines(lines=lines, thresh=1, vertical=vertical)
        return v_lines
        

class TableDetection():
    def __init__(self):
        self.line_detector = LineDetection()

    def get_table_point_limit(self, h_lines, v_lines):
        h_thresh = 15
        v_thresh = 62

        def get_groups(lines, thresh, vertical:bool, start:bool):
            if vertical:
                # sap xep theo thu tu y1<y2
                for line in lines:
                    if line[1]>line[3]:
                        line = [line[2], line[3], line[0], line[1]]
            else:
                # sap xep theo thu tu x1<x2
                for line in lines:
                    if line[0]>line[2]:
                        line = [line[2], line[3], line[0], line[1]]

            if vertical and start:
                lines = sorted(lines, key=lambda x: x[1])
            if vertical and (not start):
                lines = sorted(lines, key=lambda x: x[3])
            if (not vertical) and start:
                lines = sorted(lines, key=lambda x: x[0])
            if (not vertical) and ( not start):
                lines = sorted(lines, key=lambda x: x[2])

            groups = []

            for i in range(len(lines)):
                startX, startY, endX, endY = lines[i]
                if vertical and start:
                    query = startY
                elif vertical and (not start):
                    query = endY
                elif (not vertical) and start:
                    query = startX
                else:
                    query = endX
                if i==0:
                    group = [query]
                    continue
                if query - group[-1] < thresh:
                    group.append(query)
                else:
                    groups.append(group)
                    group = [query]
            groups.append(group)
      
            return groups
            
        startX_gr = get_groups(lines=h_lines, thresh=h_thresh, vertical=False, start=True)
        endX_gr = get_groups(lines=h_lines, thresh=h_thresh, vertical=False, start=False)
        startY_gr = get_groups(lines=v_lines, thresh=v_thresh, vertical=True, start=True)
        endY_gr = get_groups(lines=v_lines, thresh=v_thresh, vertical=True, start=False)
        
        startX = min(max(startX_gr, key=lambda x:len(x)))
        endX = max(max(endX_gr, key=lambda x:len(x)))
        startY = min(max(startY_gr, key=lambda x:len(x)))
        endY = max(max(endY_gr, key=lambda x:len(x)))

        return (startX, startY, endX, endY)
    
    def get_2_limit_h_lines(self, h_lines, startY, endY):
        top_h_line = min(h_lines, key=lambda x:abs(startY-x[1]))
        bottom_h_line = min(h_lines, key=lambda x:abs(endY-x[3]))
        return [top_h_line, bottom_h_line]
    
    def get_2_limit_v_lines(self, v_lines, top_h_line, thresh=50):
        left_v_line = min(v_lines, key=lambda x: abs(top_h_line[0]-x[0]))
        right_v_line = min(v_lines, key=lambda x:abs(top_h_line[2]-x[0]))
        
        if abs(left_v_line[0] - top_h_line[0]) > thresh:
            left_v_line = [top_h_line[0], 0, top_h_line[0], 9999] 
            
        if abs(right_v_line[0] - top_h_line[2]) > thresh:
            right_v_line = [top_h_line[2], 0, top_h_line[2], 9999] 
        return [left_v_line, right_v_line]
    
    def limit_line_refined(self, limit_h_lines, limit_v_lines):   
        top_h_line, bottom_h_line = limit_h_lines
        left_v_line, right_v_line = limit_v_lines
        top_h_line = Line(sympyPoint(top_h_line[0], top_h_line[1]), sympyPoint(top_h_line[2], top_h_line[3]))
        bottom_h_line = Line(sympyPoint(bottom_h_line[0], bottom_h_line[1]), sympyPoint(bottom_h_line[2], bottom_h_line[3]))
        left_v_line = Line(sympyPoint(left_v_line[0], left_v_line[1]), sympyPoint(left_v_line[2], left_v_line[3]))
        right_v_line = Line(sympyPoint(right_v_line[0], right_v_line[1]), sympyPoint(right_v_line[2], right_v_line[3]))
        
        top_left_point = top_h_line.intersection(left_v_line)[0].coordinates
        top_right_point = top_h_line.intersection(right_v_line)[0].coordinates
        bottom_left_point = bottom_h_line.intersection(left_v_line)[0].coordinates
        bottom_right_point = bottom_h_line.intersection(right_v_line)[0].coordinates
        
        top_left_point = [int(point) for point in top_left_point]
        top_right_point = [int(point) for point in top_right_point]
        bottom_left_point = [int(point) for point in bottom_left_point]
        bottom_right_point = [int(point) for point in bottom_right_point]
        
        top_h_line = [top_left_point[0], top_left_point[1], top_right_point[0], top_right_point[1]]
        bottom_h_line = [bottom_left_point[0], bottom_left_point[1], bottom_right_point[0], bottom_right_point[1]]
        left_v_line = [top_left_point[0], top_left_point[1], bottom_left_point[0], bottom_left_point[1]]
        right_v_line = [top_right_point[0], top_right_point[1], bottom_right_point[0], bottom_right_point[1]]
        
        limit_h_lines = [top_h_line, bottom_h_line]
        limit_v_lines = [left_v_line, right_v_line]
        return (limit_h_lines, limit_v_lines)
    
    def table_refined(self, h_lines, v_lines, limit_h_lines, limit_v_lines, h_ratio_thresh=0.5, v_pixel_thresh=200):
        def get_new_line(line, limit_lines):
            line = Line(sympyPoint(line[0], line[1]), sympyPoint(line[2], line[3]))
            new_points = []
            for limit_line in limit_lines:
                limit_line = Line(sympyPoint(limit_line[0], limit_line[1]), sympyPoint(limit_line[2], limit_line[3]))
                new_point = line.intersection(limit_line)[0].coordinates
                new_point = [int(p) for p in new_point]
                new_points.extend(new_point)
            return new_points
        def remove_duplicate_line(lines, thresh=5):
            new_lines = []
            for i in range(len(lines)-1):
                line = lines[i]
                next_line = lines[i+1]
                
                x1A, y1A, x2A, y2A = line
                x1B, y1B, x2B, y2B = next_line

                # ptdt: ax+by+c=0
                a = y1B - y2B
                b = x2B - x1B
                c = y2B*(x1B-x2B) - x2B*(y1B-y2B)
                
                distAB = abs(a*x1A+b*y1A+c)/math.sqrt(a**2+b**2)
                
                if distAB > thresh:
                    new_lines.append(line)
            # them line cuoi cung
            new_lines.append(lines[-1])
            return new_lines

        h_lines_refined = []
        standard_width = abs(limit_h_lines[0][2] - limit_h_lines[0][0])
        for h_line in h_lines:
            width = abs(h_line[0]-h_line[2])
            if (width/standard_width > h_ratio_thresh):
                new_h_line = get_new_line(h_line, limit_v_lines)
                if (limit_h_lines[0][1] <= new_h_line[1] <= limit_h_lines[1][1]):
                    h_lines_refined.append(new_h_line)
        
        v_lines_refined = []
        standard_height = abs(limit_v_lines[0][1]-limit_v_lines[0][3])
        for v_line in v_lines:
            # height = abs(v_line[3]-v_line[1])
            if abs(min(v_line[1],v_line[3]) - limit_v_lines[0][1]) < v_pixel_thresh:
                new_v_line = get_new_line(v_line, limit_h_lines)
                if limit_v_lines[0][0] <= new_v_line[0] <= limit_v_lines[1][0]:
                    v_lines_refined.append(new_v_line)
                    
            if abs(max(v_line[1],v_line[3]) - limit_v_lines[0][3]) < 50:
                line = LineString([(v_line[0], v_line[1]), (v_line[2], v_line[3])])
                lim_h_line = LineString([(limit_h_lines[0][0], limit_h_lines[0][1]), (limit_h_lines[0][2], limit_h_lines[0][3])])
                x = line.intersection(lim_h_line)
               # if type(x.coords) != list:
                if not x.is_empty:
                    new_v_line = get_new_line(v_line, limit_h_lines)
                    v_lines_refined.append(new_v_line)
                    
        # add the limit lines because of missing limit line on some image
        h_lines_refined.extend(limit_h_lines)
        v_lines_refined.extend(limit_v_lines)
        
        h_lines_refined = sorted(h_lines_refined, key=lambda x:x[1])
        v_lines_refined = sorted(v_lines_refined, key=lambda x:x[0])
        
        # final refine
        h_lines_refined = remove_duplicate_line(h_lines_refined)
        v_lines_refined = remove_duplicate_line(v_lines_refined)   
        
        return h_lines_refined, v_lines_refined

    
    def detect(self, img0):
        img = img0.copy()
        img_edge = self.line_detector.get_edge_image(img)        
        v_lines = self.line_detector.detect_lines(img_edge, vertical=True)
        h_lines = self.line_detector.detect_lines(img_edge, vertical=False)
        v_correct_lines = self.line_detector.get_correct_lines(v_lines, thresh=4, vertical=True)
        h_correct_lines = self.line_detector.get_correct_lines(h_lines, thresh=1, vertical=False)
        
        # <-- debug 260723 -->
        # for v_line in v_correct_lines:
        #     x1, y1, x2, y2 = v_line
        #     cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        # # for h_line in h_correct_lines:
        # #     x1, y1, x2, y2 = h_line
        # #     cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
        # cv2.imwrite('debug0.jpg', img)
        # <-- end debug --->
            
        # get the top_left and bottom_right of the table
        startX, startY, endX, endY = self.get_table_point_limit(h_lines=h_correct_lines,
                                                                v_lines=v_correct_lines)
        
        # # <-- debug 260723 -->
        # cv2.rectangle(img, (startX, startY), (endX, endY), (255,0,0), 2)
        # cv2.imwrite('debug0.jpg', img)

        # get 4 limit lines of the table
        limit_horizontal_lines = self.get_2_limit_h_lines(h_correct_lines, startY, endY)
        limit_vertical_lines = self.get_2_limit_v_lines(v_correct_lines, limit_horizontal_lines[0], thresh=50)
        
        # # <-- debug 260723 -->
        # for limit_line in limit_horizontal_lines:
        #     x1, y1, x2, y2 = limit_line
        #     cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
        #     cv2.imwrite('debug0.jpg', img)

        
        # refined these limit lines
        limit_horizontal_lines, limit_vertical_lines = self.limit_line_refined(limit_horizontal_lines, limit_vertical_lines)
        
        # get the correct lines of the table
        h_lines_refined, v_lines_refined =  self.table_refined(h_correct_lines, v_correct_lines, limit_horizontal_lines, limit_vertical_lines)

        # for v_line in v_lines_refined:
        #     x1, y1, x2, y2 = v_line
        #     cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        # for h_line in h_lines_refined:
        #     x1, y1, x2, y2 = h_line
        #     cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
        # cv2.imwrite('debug0.jpg', img)

        return [h_lines_refined, v_lines_refined]

if __name__ == "__main__":
    import glob
    output_folder = 'debug_line'
    os.makedirs(output_folder, exist_ok=True)
    line_detector = LineDetection()
    # hLines = line_detector.one_shot(img, vertical=False)
    # table_detector = TableDetection()
    imgs_path = glob.glob('/hdd/tuannca/ocr/app-business-form-recognition_namtc/src/test_imageset_1/horizon/*.jpg')
    for img_path in tqdm(imgs_path):
        # im_name = img_path.split('/')[-2]
        global im_name
        im_name = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path)
        img, _ = detect_form(img)
        vLines = line_detector.one_shot(img, True)
        hLines = []
        for vline in vLines:
            x1, y1, x2, y2 = vline
            cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
        for hline in hLines:
            x1, y1, x2, y2 = hline
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imwrite(f'debug_{im_name}.jpg', img)
