""" File detector.py
    The functions for pre-processing, detect form, detect table, detect line
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # arshot root directory
sys.path.append(str(ROOT))

from arshot import ObjectDetector
from arshot.utility.imageproc.image import order_points, four_point_transform, get_box_area, find_four_corners_np
from arshot.services.table_detector.config import yolov5_param, col_lines_cfg
import time
def detect_form(img0, thresh_value=35):
    """ Detect the business form and do perspective transformation
    """
    # Binary image
    scale_ratio = 4
    img_resize = cv2.resize(img0, None, fx=1/scale_ratio, fy=1/scale_ratio)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    #thresh_value = np.mean(img_gray)*3/2
    #_, img_bin = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
    thres_otsu, img_bin = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if thres_otsu < thresh_value:
        return np.array([])
    
    kernel_size=(3,3)
    kernel = np.ones(kernel_size, np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)   # Opening the image

    # Find contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return np.array([])

    # Find 4 corners
    max_contour = max(contours, key=cv2.contourArea)
    box = np.float32(find_four_corners_np(size = img_gray.shape, contour=max_contour))

    # rect = cv2.minAreaRect(max_contour)
    # box = cv2.boxPoints(rect)

    # Apply perspective transform
    box = np.int0(box) * scale_ratio
    rect = order_points(pts=box)

    return rect

class TableDetection():
    config_param = yolov5_param

    def __init__(self):
        self.detector = ObjectDetector(self.config_param)
    
    def detect(self, image):
        """ Detect table objects printed in the business form

        Args:
            image (np.ndarray): the input image
        Returns:
            results: list of detected table. Each table has format (label, confidence, box) 
        """
        res = self.detector.detect(image)
        return res

class TableLineDetection():
    def __init__(self):
        pass

    def get_union_length(self, segments):
        n = len(segments)
        points = [None] * (n*2)
        for i in range(n):
            points[i*2] = (segments[i][0], False)
            points[i*2+1] = (segments[i][1], True)
        points = sorted(points, key=lambda x: x[0])

        result = 0
        counter = 0
        corrupt_dist = 0
        for i in range(n*2):
            if i > 0 and (points[i][0] > points[i-1][0]) and (counter > 0):
                result += points[i][0] - points[i-1][0]
        
            if points[i][1]:
                counter -= 1
            else:
                if i > 0 and counter == 0: # cal corrupt dist 
                    # print('cal dist', points[i], points[i-1])
                    corrupt_dist += points[i][0] - points[i-1][0]
                counter += 1

        return result, corrupt_dist
    
    def detect_col_lines(self, image):
        img_h, img_w = image.shape[:2]
        # Binary image
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
        thresh = cv2.bitwise_not(thresh)

        # Detect vertical lines by applying HoughLineP
        vertical = thresh.copy()
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, (1, 1), iterations=8)
        vertical = cv2.erode(vertical, (1, 1), iterations=7)

        ver_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 30, np.array([]), 30, 2)
        if ver_lines is None:
            return None, None

        temp_line = []
        for line in ver_lines:
            for x1, y1, x2, y2 in line:
                if y1 >= y2:
                    temp_line.append([x2, y2, x1, y1])
                else:
                    temp_line.append([x1, y1, x2, y2])
        ver_lines = sorted(temp_line, key=lambda x: x[0])

        # Get best lines for vertical lines
        lastx1 = -111111
        lines_y1 = []
        lines_y2 = []
        lines_x = []
        ver1_lines = []
        count = 0
        for x1, y1, x2, y2 in ver_lines:
            if x1 <= lastx1 + col_lines_cfg["max_distance_segments"]:
                lines_y1.append(y1)
                lines_y2.append(y2)
                lines_x.append(x1)
            else:
                if (count != 0 and len(lines_y1) != 0):
                    segments = [(i, j) for i, j in zip(lines_y1, lines_y2)]
                    l, corrupt_dist = self.get_union_length(segments)
                    if l >= img_h * col_lines_cfg["min_union_segments_over_height"] and corrupt_dist <= l * col_lines_cfg["max_corrupt_segments_over_line"]:
                        x = sum(lines_x) // len(lines_x)
                        ymin = min(lines_y1)
                        ymax = max(lines_y2)
                        ver1_lines.append([x, ymin, x, ymax])

                lastx1 = x1
                lines_y1 = []
                lines_y2 = []
                lines_x = []
                lines_y1.append(y1)
                lines_y2.append(y2)
                lines_x.append(x1)
                count += 1
        
        # x = sum(lines_x) // len(lines_x)
        # ymin = min(lines_y1)
        # ymax = max(lines_y2)

        segments = [(i, j) for i, j in zip(lines_y1, lines_y2)]
        l, corrupt_dist = self.get_union_length(segments)
        if l >= img_h * col_lines_cfg["min_union_segments_over_height"] and corrupt_dist <= l * col_lines_cfg["max_corrupt_segments_over_line"]:
            x = sum(lines_x) // len(lines_x)
            ymin = min(lines_y1)
            ymax = max(lines_y2)
            ver1_lines.append([x, ymin, x, ymax])

      

        # Merge group lines too close each other
        lastx = -11111
        ver2_lines = []
        curr_line = []
        curr_l = 0
        for x1, y1, x2, y2 in ver1_lines:
            if lastx < 0:
                curr_line = [x1, y1, x2, y2]
                lastx = x1

            l = y2 - y1
            if x1 - lastx <= col_lines_cfg["max_distance_line"]: # too close
                # continue
                if l > curr_l:
                    curr_l = l 
                    curr_line = [x1, y1, x2, y2]
                    lastx = x1
            else:
                if len(curr_line) > 0:
                    ver2_lines.append(curr_line)
                else:
                    ver2_lines.append([x1, y1, x2, y2])
                lastx = x1
                curr_line = [x1, y1, x2, y2]
                curr_l = l

        # last line
        if len(curr_line):
            ver2_lines.append(curr_line)

        return ver2_lines, vertical

if __name__ == "__main__":    
    table_detector = TableDetection()
    col_lines_detector = TableLineDetection()

    img_dir = '/hdd/namtp/app-business-form-recognition/src/siameses/tabnet/table_data/images/'
    paths = os.listdir(img_dir)

    out_dir = f'output'
    os.makedirs(out_dir, exist_ok=True)
    COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for path in tqdm(paths[:100]):
        img = cv2.imread(f'{img_dir}/{path}')
        img_name = os.path.basename(path)

        # Detect table objects
        ret = table_detector.detect(img)

        if len(ret) == 0:
            cv2.imwrite(f'{out_dir}/{img_name}', img)
            continue

        # Get the largest table
        largest_box = max(ret, key=lambda x: get_box_area(x[2]))[2]
        xmin, ymin, xmax, ymax = map(int, largest_box)

        # Get the table image
        table_body = img[ymin: ymax, max(0, xmin - 20): xmax + 20]
        col_lines, thres_img = col_lines_detector.detect_col_lines(table_body)

        ### Visualize
        thres_img = np.dstack((thres_img, thres_img, thres_img))
        img_rst = table_body.copy()
        for i, (x1, y1, x2, y2) in enumerate(col_lines):
            cv2.line(img_rst, (x1, y1), (x2, y2), COLORS[i % len(COLORS)], 5)

        img_vis = np.vstack((table_body, thres_img, img_rst))
        cv2.imwrite(f'{out_dir}/{img_name}', img_vis)
