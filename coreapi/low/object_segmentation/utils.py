import cv2
import numpy as np

def find_four_corners_np(size, contour):
    height, width = size
    borders = np.array([[0,0], [0,height], [width, 0], [width, height]])
    repeat_contour = np.repeat(contour, 4, axis=1)
    dist = np.sum(np.absolute((repeat_contour - borders)), axis=2)
    indicates = np.argmin(dist, axis=0)
    corners = contour[indicates]
    return np.squeeze(corners)

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detect_paper(mask):

    scale_ratio = 1
    img_bin = cv2.resize(mask, None, fx=1/scale_ratio, fy=1/scale_ratio)
    kernel_size=(3,3)
    kernel = np.ones(kernel_size, np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)   # Opening the image

    # Find contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return np.array([])

    # Find 4 corners
    max_contour = max(contours, key=cv2.contourArea)
    box = np.float32(find_four_corners_np(size = img_bin.shape, contour=max_contour))

    # rect = cv2.minAreaRect(max_contour)
    # box = cv2.boxPoints(rect)

    # Apply perspective transform
    box = np.int0(box) * scale_ratio
    rect = order_points(pts=box)

    return rect

def probs2mask(probs, is_pad=True):
    imgH, imgW = probs.shape
    padH, padW = 0, 0
    if is_pad:
        padH, padW = imgH//2, imgW//2
    mask = np.zeros((imgH + padH, imgW + padW), dtype=probs.dtype)
    mask[padH : padH + imgH, padW : padW + imgW] = probs * 255
    return mask
