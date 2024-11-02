""" file: image.py:
    Implementation of image processing algorithms

"""

import cv2
import numpy as np
import math
from skimage import io
from PIL import Image, ImageEnhance

class BackgroundSubtractor:
    """
    The class to implement the background subtraction method.

    Args:
        learning (boolean): Update the background if learning is True.
        var_threshold (int): Threshold on the squared distance between the pixel and the sample to decide
                    whether a pixel is close to that sample. This parameter does not affect the background update
        detect_shadows (boolean): If true, the algorithm will detect shadows and mark them.
    """
    def __init__(self, learning=True, var_threshold=16, detect_shadows=True, name="MOG2 Background Subtractor"):
        self.learning = learning
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.name = name
        self.bgsub = cv2.createBackgroundSubtractorMOG2(
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows)

    def get_background_subtract_image(self, image):
        """ Apply background subtraction and get the mask image (binary image)

        Args:
            image (array): The input image

        Returns:
            The mask image is outputed
        """
        prep_img = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        if self.learning:
            bgsub_img = self.bgsub.apply(prep_img, learningRate=0)
        else:
            bgsub_img = self.bgsub.apply(prep_img)
        _, mask_img = cv2.threshold(bgsub_img, 127, 255, cv2.THRESH_BINARY)
        # kernel_size = np.ones((3,3), np.uint8)
        # mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel_size)

        return mask_img, bgsub_img

def padding_image(image, size, central=True, pad_color=0):
    """ Padding image to the specified size

    Args:
        image (array): Raw image to be padded
        size (int): The size of image (h, w) after padding
        central (boolean): Padding around the image or padding to bottom-right of origin image
        pad_color (int): The color of region will be padded to the raw image. Default is black color.

    Returns:
        the image after padding:    [None] if the target size is not larger than original size
                                    [image] if padding successfuly
    """
    img_h, img_w = image.shape[:2]
    target_h, target_w = size

    # Set the padding color
    if len(image.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3

    if img_h > target_h or img_w > target_w:
        print("Can not padding")
        return None

    # Padding central
    if central:
        pad_lef = (target_w - img_w) // 2
        pad_top = (target_h - img_h) // 2
    else:
        pad_lef = 0
        pad_top = 0
    
    pad_rig = target_w - img_w - pad_lef
    pad_bot = target_h - img_h - pad_top

    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_lef, pad_rig,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)
    padded_dim = (pad_lef, pad_top, pad_rig,  pad_bot) 

    return {"image": padded_image, "pad_info": padded_dim}

def convert_coordinate(box, crop_coord, image_width, image_height):
    """ Convert the box's coordinate from cropping image to origin image.

    Args:
        box (list): The coordinate of box in cropping image [xmin, ymin, xmax, ymax]
        crop_coord (list): The coordinate of cropping image in the origin image [(xmin, ymin), (xmax, ymax)]
        image_width (int), image height (int): The size of the origin image
    Returns:
        The coordinate of box in the origin image
    """

    crop_width = crop_coord[1][0] - crop_coord[0][0]
    crop_height = crop_coord[1][1] - crop_coord[0][1]

    xmin = int(box[0] * (crop_width / image_width)) + crop_coord[0][0]
    ymin = int(box[1] * (crop_height / image_height)) + crop_coord[0][1]
    xmax = int(box[2] * (crop_width / image_width)) + crop_coord[0][0]
    ymax = int(box[3] * (crop_height / image_height)) + crop_coord[0][1]

    return [xmin, ymin, xmax, ymax]

def crop_image(image,ratio=0.8):
    """ Crop a part of image based on the ratio parameter.

    Args:
        image (array): The raw input image for cropping.
        ratio (float): Ratio value < 1. This parameter determines the size of new image after cropping.

    Returns:
        The cropped image
    """

    height, width = image.shape[:2]
    xmin = int(width*(1- ratio)/2)
    ymin = int(height*(1 - ratio)/2)
    xmax = int(width * (1 - (1 - ratio)/2))
    ymax = int(height * (1 - (1 - ratio)/2))
    crop_img = image[ymin:ymax, xmin:xmax]
    crop_h, crop_w = crop_img.shape[:2]
    if crop_h > 0 and crop_w > 0:
        crop_img = cv2.resize(crop_img, (width, height))
        coordinate_crop = [(xmin, ymin), (xmax, ymax)]
    else:
        crop_img = image
        coordinate_crop = [(0, 0), (width, height)]
    return crop_img, coordinate_crop

def adjust_gamma(img, gamma=0.8):
    """Correct the brightness of an image by using a non-linear transformation
    between the input values and the mapped output values

    Args:
        img (array): The raw input image
        gamma (float): Gamma values < 1 will shift the image towards the darker end of the spectrum
                        while gamma values > 1 will make the image appear lighter

    Returns:
        The image after gamma correction
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def find_contour(img):
    """Apply finding contour function to search all the continuous points (along the boundary),
        having the same color or intensity

    Args:
        img (array): The raw input image

    Returns:
        list: The list of all the contours in the image.
            Each individual contour is a Numpy array of (x, y) coordinates of boundary points of the object.
    """
    if cv2.__version__.split('.')[0] == '4':
        cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, cnts, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def check_glossy(img, detect_box, light_th=200):
    """Check whether the glossy is in the region of image

    Args:
        img (array): The raw input image
        detect_box (tuple): the corrdinate (xmin, ymin, xmax, ymax) of the region to be checked
        light_th (int): the threshold of lighting

    Returns:
        bool: The return value. True for glossy checked, False otherwise.
    """

    img_h, img_w = img.shape[:2]
    xmin, ymin, xmax, ymax = detect_box
    w = xmax - xmin
    h = ymax - ymin

    # Padding
    ext_ratio = 0.5
    xmin = max(0, int(xmin - w*ext_ratio))
    xmax = min(img_w, int(xmax + w*ext_ratio))
    ymin = max(0, int(ymin - h*ext_ratio))
    ymax = min(img_h, int(ymax + h*ext_ratio))
    im = img[ymin:ymax, xmin:xmax]

    # Gamma correction and convert to LAB space
    im = adjust_gamma(im, gamma=0.2)
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(im_lab)

    # Binarize on lightness channel
    l_bin = np.where(l < light_th, 0, 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    l_bin = cv2.erode(l_bin, kernel, iterations=1)
    l_bin = cv2.dilate(l_bin, kernel, iterations=2)

    # Verify glossy region
    area_th = 0.2 * w * h
    cnts = find_contour(l_bin)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > area_th]
    glare_check = False
    if len(cnts) > 0:
        glare_check = True
    return glare_check

def square_center_crop(image) -> np.ndarray:
    """Crop the center of the image so the new image has equal width and height, and return it

    Args:
        image (np.ndarray): Image

    Returns:
        image (np.ndarray): Square Image
    """

    img_h, img_w = image.shape[:2]
    if img_h < img_w:
        start_w = (img_w - img_h) // 2
        image = image[:, start_w:start_w + img_h].copy()
    elif img_h > img_w:
        start_h = (img_h - img_w) // 2
        image = image[start_h:start_h + img_w, :].copy()
    return image


def expand_convex_polygon(polygon, scale_x=1., scale_y=1.):
    """Process the convex pologon
        1.find the center of the contour (moments() or getBoundingRect())
        2.subtract it from each point in the contour
        3.multiply contour points x,y by a scale factor
        4.add the center again to each point

    Args:
        polygon (array):
        scale_x (float):
        scale_y (float):

    Returns:
        array: the polygon after processing
    """

    polygon = polygon.astype(int)
    x, y, w, h = cv2.boundingRect(polygon)
    cx, cy = x + w // 2, y + h // 2
    polygon -= np.array([cx, cy])
    polygon = polygon.astype(float)
    polygon[:, 0] *= scale_x
    polygon[:, 1] *= scale_y
    polygon += np.array([cx, cy])
    polygon = polygon.astype(int)
    return polygon

def transform(cnt):
    """Find the corners of the object and the dimensions of the object

    Args:
        cnt (list): the contours of objects to be checked

    Returns:
        int, int: the width and height of the object
        list: the coordinates of the object
    """

    pts = []
    n = len(cnt)
    for i in range(n):
        pts.append(list(cnt[i]))

    sums = []
    diffs = []
    for i in pts:
        x = i[0]
        y = i[1]
        sum_xy = x + y
        diff = y - x
        sums.append(sum_xy)
        diffs.append(diff)

    sums_order = np.argsort(sums)
    diffs_order = np.argsort(diffs)
    n = len(sums)
    rect = [pts[sums_order[0]], pts[diffs_order[0]],
            pts[diffs_order[n - 1]], pts[sums_order[n - 1]]]

    h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 +
                 (rect[0][1] - rect[2][1]) ** 2)  # height of left side
    h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 +
                 (rect[1][1] - rect[3][1]) ** 2)  # height of right side
    h = max(h1, h2)

    w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 +
                 (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
    w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 +
                 (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
    w = max(w1, w2)
    return int(w), int(h), rect

def warp(image, polygon):
    """warp perspective transformation

    Args:
        image (array): the input image
        polygon (list): the coordinate of object

    Returns:
        The image after warp perspective transformation
    """
    polygon = np.array(polygon)
    polygon = expand_convex_polygon(polygon, 1, 1)  # need expand box???
    polygon[:, 0] = np.clip(polygon[:, 0], 0, image.shape[1])
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image.shape[0])

    w, h, rect = transform(polygon)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts1 = np.float32(rect)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    warp_img = cv2.warpPerspective(image, matrix, (w, h))
    return warp_img

def filter_object(cnts, width, height, min_area_ratio=0.005, ref_point_id=0):
    '''
    T.B.D
    '''
    box = []
    cnt = []
    if len(cnts) == 0:
        return box, cnt

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    ref_area = cv2.contourArea(cnts[0]) / 2
    min_area = width * height * min_area_ratio

    min_dist = math.sqrt(width ** 2 + height ** 2)
    ref_point_list = [(0, 0), (width, 0), (width, height), (0, height), (width // 2, height //2)]
    ref_point = ref_point_list[ref_point_id]

    for i in range(len(cnts)):

        # Ignore small objects
        if cv2.contourArea(cnts[i]) < min_area:
            continue

        if cv2.contourArea(cnts[i]) < ref_area:
            continue

        rect = cv2.boundingRect(cnts[i])
        xmin, ymin = rect[0], rect[1],
        xmax, ymax = rect[0] + rect[2], rect[1] + rect[3]

        # if (rect[3] < width / 28) and (rect[3] < rect[2] / 20): # Ignore horizontal line
        #     continue

        if ymax < height / 2: # Ignore the far objects near the top border
            continue

        # Find the object close to center of frame
        center_obj_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        dist_obj = calc_distance_2points(ref_point, center_obj_point)
        if dist_obj < min_dist:
            min_dist = dist_obj
            box = (xmin, ymin, xmax, ymax)
            cnt = cnts[i]

    return box, cnt

def calc_distance_2points(p1, p2):
    """
    Calculate the distance between 2 points.
    """
    p1x, p1y = p1
    p2x, p2y = p2
    dx = p2x - p1x
    dy = p2y - p1y
    return math.sqrt(dx * dx + dy * dy)

def get_box_area(box):
    """
    Calculate the area of bounding box

    Args:
        box(list): The bouning box [xmin, ymin, xmax, ymax]
    """
    return (box[3] - box[1])*(box[2] - box[0])

def load_image(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def enhance_sharpness(img, factor:int):
    ''' Enhance the sharpeness and contrast of input image
    Args:
		img (nd.array): input image
		factor (int): factor of Sharpen and Contrast enhancement
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

def find_four_corners(size, contour):
    height, width = size
    borders = [[0,0], [0,height], [width, 0], [width, height]]
    max_dis = math.sqrt(height**2 + width**2)
    corners = [None, None, None, None]
    distance = [max_dis, max_dis, max_dis, max_dis]
    for point in contour:
        x,y = point[0]
        for index, border in enumerate(borders):
            # Using eclude distance
            #temp_dis = math.sqrt((x-border[0])**2 + (y-border[1])**2)
            # Using mahattan distance
            temp_dis = math.sqrt(abs(x-border[0]) + abs(y-border[1]))
            if distance[index] > temp_dis:
                distance[index] = temp_dis
                corners[index] = [x,y]
    return corners

def find_four_corners_np(size, contour):
    height, width = size
    borders = np.array([[0,0], [0,height], [width, 0], [width, height]])
    repeat_contour = np.repeat(contour, 4, axis=1)
    dist = np.sum(np.absolute((repeat_contour - borders)), axis=2)
    indicates = np.argmin(dist, axis=0)
    corners = contour[indicates]
    return np.squeeze(corners)