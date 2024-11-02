from shapely.geometry import Polygon, Point, LineString
import numpy as np
import cv2
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta_time = round(float(end - start), 8)
        print("- Function {} run in {}'s".format(func.__name__, delta_time))
        return result

    return wrapper

def isPolygonOverlap(poly1, poly2, iou_th=0.2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2)
    ### polygon or linestring?
    isOverlap = False
    if isinstance(intersection, Polygon):
        intersection_per_poly1 = intersection.area / min(poly1.area, poly2.area)
        if intersection_per_poly1 >= iou_th:
            isOverlap = True
    return isOverlap

def isPolyContainBox(box, polygon):
    """
    box: [x1, y1, x2, y2]
    polygon : [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    """
    # print('box:\n', box)
    # print('poly:', polygon)
    polygon = Polygon(polygon)
    box_center = Point(box[[0, 1]].sum() / 2, box[[2, 3]].sum() / 2)
    return polygon.contains(box_center)

def is_box_same_line(bbox1, bbox2, overlapTh=0.3):
    # box, query_box
    bbox1_ymin, bbox1_ymax = sorted([bbox1[2],bbox1[3]])
    bbox2_ymin, bbox2_ymax = sorted([bbox2[2],bbox2[3]])

    if bbox1_ymax <= bbox2_ymin or bbox2_ymax < bbox1_ymin:
        return False
    
    if bbox1_ymin >= bbox2_ymin and bbox1_ymax <= bbox2_ymax:
        return True

    y = [bbox1_ymin, bbox1_ymax, bbox2_ymin, bbox2_ymax]
    y.sort()
    ratio = (y[2] - y[1]) / (y[3] - y[0])
    ret = False
    if ratio > overlapTh:
        ret = True
    return ret


def get_boxes_inside_areas(box_list, config_area):
    """ Filter and return the list of boxes inside specified areas
    Args:
        box_list (list): the list of boxes [x1, x2, y1, y2]
        config_area : [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    """
    boxes_inside = []
    boxes_outside = []
    area_polygon = Polygon(config_area)
    for box in box_list:
        if np.array(box).ndim == 2:
            box_center = Point(np.array(box)[:,0].mean(), np.array(box)[:,1].mean())
        else:
            x1, x2, y1, y2 = box
            box_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        if area_polygon.contains(box_center):
            boxes_inside.append(box)
        else:
            boxes_outside.append(box)
    return boxes_inside, boxes_outside


def get_boxes_inside_areas2(box_list, config_area):
    """ Filter and return the list of boxes inside specified areas
    Args:
    intersects
        box_list (list): the list of boxes [x1, x2, y1, y2]
        config_area : [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    """
    min_IOU = 0.3
    max_IOU = 0.6
    boxes_inside = []
    boxes_outside = []
    area_polygon = Polygon(config_area)
    for box in box_list:
        if not len(box):
            continue
        elif np.array(box).ndim == 2: # poly
            box_polygon = Polygon(box)
        else:
            box_polygon = Polygon(rect2poly(box))
        intersection = box_polygon.intersection(area_polygon)
        if not intersection.is_empty:
            ### polygon or linestring?
            if isinstance(intersection, Polygon):
                inter_per_textBox = intersection.area / box_polygon.area
                if min_IOU <= inter_per_textBox < max_IOU:
                    box_inside = intersection.exterior.coords[:]
                    boxes_inside.append(poly2rect(box_inside, xxyy=True))
                elif inter_per_textBox >= max_IOU:
                    box_inside = box
                    boxes_inside.append(box_inside)

            difference = box_polygon.difference(area_polygon)
            if isinstance(difference, Polygon):
                boxes_outside.append(poly2rect(difference.exterior.coords[:], xxyy=True))
            else: # multi polygon
                for difference_poly in difference.geoms:
                    boxes_outside.append(poly2rect(difference_poly.exterior.coords[:], xxyy=True))

    return boxes_inside, boxes_outside

def get_single_digits_bbox(img_gray, bbox, iteration=3):
    img_h, img_w = img_gray.shape[:2]
    xmin, xmax, ymin, ymax = bbox
    box_h = abs(ymax - ymin)
    expand_ratio = 0.1
    ymin = max(0, int(ymin - expand_ratio * box_h))
    ymax = min(img_h, int(ymax + expand_ratio * box_h))
    temp_img = img_gray[ymin:ymax, xmin:xmax]
    _, thresh = cv2.threshold(temp_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=iteration)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[1] < box_h / 2) and (cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] > box_h / 2)]
    
    if len(contours):
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        x1 = max(0, int(x - 0.15 * h)) + xmin
        y1 = max(0, int(y - 0.15 * h)) + ymin
        x2 = min(img_w, int(x + w + 0.15 * h)) + xmin
        y2 = min(img_h, int(y + h + 0.15 * h)) + ymin
    else:
        x1, x2, y1, y2 = bbox # keep as before

    return x1, x2, y1, y2

def recover_boxes_single_digits(img_gray, horizontal_list, recover_config_area, refer_config_area):
    """ Recover for single characters based on the configuration
    Args:
        horizontal_list (list): the list of boxes [x1, x2, y1, y2]
        recover_config_area : [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        refer_config_area : [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    """
    recover_list, _ = get_boxes_inside_areas(horizontal_list, recover_config_area)
    refer_list, _ = get_boxes_inside_areas(horizontal_list, refer_config_area)
    
    if len(recover_list):
        minX = np.array(recover_list)[:,0].min()
        maxX = np.array(recover_list)[:,1].max()
    else:
        minX, maxX = 0, 0

    minXConfigArea = np.array(recover_config_area)[:,0].min() + 3
    maxXConfigArea = np.array(recover_config_area)[:,0].max() - 3

    # Recover based on the element of refer column
    recover_col_list = []
    for refbox in refer_list:
        x1, x2, y1, y2 = refbox
        recbox = [minXConfigArea, maxXConfigArea, y1, y2]
        recover_col_list.append(recbox)

    # Correct the coordinates inside column
    for cell_bbox in recover_col_list:
        cell_x1, cell_x2, cell_y1, cell_y2 = cell_bbox
        cell_polygon = Polygon([(cell_x1, cell_y1), (cell_x2, cell_y1), (cell_x2, cell_y2), (cell_x1, cell_y2)])

        adj_x1, adj_x2 = cell_x1, cell_x2
        adj_y1, adj_y2 = cell_y1, cell_y2
        if minX > 0 and maxX > 0: # At least 1 text box detected by model
            adj_x1, adj_x2 = minX, maxX

        det_box_id_list = []
        cell_y1_list = []
        cell_y2_list = []
        for i, box in enumerate(horizontal_list):
            x1, x2, y1, y2 = box
            box_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
            # if cell_polygon.contains(box_center):
            if isPolygonOverlap(rect2poly(cell_bbox), rect2poly(box), iou_th=0.2):
                cell_y1_list.append(y1)
                cell_y2_list.append(y2)
                det_box_id_list.append(i)

        if len(det_box_id_list) > 0:
            # horizontal_list[det_box_id_list[0]] = [adj_x1, adj_x2, min(cell_y1_list), max(cell_y2_list)]
            # horizontal_list = [box for idx, box in enumerate(horizontal_list) if idx not in det_box_id_list[1:]]
            pass
        else:
            # if adj_x1 == cell_x1 and adj_x2 == cell_x2:  # need to narrow down
            #     adj_x1, adj_x2, adj_y1, adj_y2 = get_single_digits_bbox(img_gray, (adj_x1, adj_x2, adj_y1, adj_y2))
            horizontal_list.append([adj_x1, adj_x2, adj_y1, adj_y2])

    return horizontal_list
    
def get_boxes_each_table_row(box_list, h_line1, h_line2):
    row_polygon = [(h_line1[0], h_line1[1]), (h_line1[2], h_line1[3]), (h_line2[2], h_line2[3]), (h_line2[0], h_line2[1])]
    row_boxes, _ = get_boxes_inside_areas(box_list, row_polygon)
    new_box_list = [box for box in box_list if box not in row_boxes]
    return row_boxes, new_box_list

def get_boxes_for_all_row(box_list, table):
    h_lines, v_lines = table
    new_box_list = box_list.copy()
    
    table_boxes_sorted = []
    for i in range(len(h_lines)-1):
        h_line_boxes = []
        h_line = h_lines[i]
        next_h_line = h_lines[i+1]
        boxes_each_row, new_box_list = get_boxes_each_table_row(new_box_list, h_line, next_h_line)
        table_boxes_sorted.append(boxes_each_row)
    return table_boxes_sorted        

def get_boxes_with_lines(box_list):
    box_list = sorted(box_list, key=lambda x: x[2])
    line_marked = [None] * len(box_list)
    line_id = 0
    for i in range(len(box_list)):
        if line_marked[i] is None:
            line_marked[i] = line_id
            for j in range(i+1, len(box_list)):
                if is_box_same_line(box_list[i], box_list[j]):
                    line_marked[j] = line_id
            line_id += 1

    boxes_with_lines = []
    line_marked = np.array(line_marked)
    
    for i in range(max(line_marked) + 1):
        indices = np.where(line_marked == i)[0]
        same_line_boxes = [box_list[j] for j in indices]
        sorted_same_line_boxes = sorted(same_line_boxes, key=lambda x: x[0])
        boxes_with_lines.append(sorted_same_line_boxes)

    return boxes_with_lines

def get_boxes_with_lines_new(box_list):
    box_list = sorted(box_list, key=lambda x: x[2])
    rows = []
    row = []
    for i in range(len(box_list)):
        box = box_list[i]
        if i==0:
            row = [box]
            continue
        # compare with the box which have best y2.
        query_box = max(row, key=lambda x: x[3])
        if is_box_same_line(box, query_box):
            row.append(box)
        else:
            # stop groupping, init new group
            rows.append(row)
            row = [box]
    # add last group, which has not added before.
    rows.append(row)
    return rows
            

def get_boxes_for_table_colum(table_boxes_sorted, config_areas, table):
    """
    get columns version 1, using vlines
    """
    h_lines, v_lines = table
    table_boxes_final = []
    for row in table_boxes_sorted: 
        boxes_each_row = []
        for i in range(len(v_lines)-1):
            boxes_each_colum = []
            v_line = v_lines[i]
            next_v_line = v_lines[i+1]
            colum_polygon = [(v_line[0], v_line[1]), (next_v_line[0], next_v_line[1]), (next_v_line[2], next_v_line[3]), (v_line[2], v_line[3])]
            boxes_each_colum, _ = get_boxes_inside_areas(row, colum_polygon)
            boxes_each_row.append(boxes_each_colum)
        table_boxes_final.append(boxes_each_row)
    return table_boxes_final

def getCol(rows, config_areas):
    """
    get column version 2, using config
    """
    newRows = []
    for i, row in enumerate(rows):
        col = []
        for field_name, field_coord in config_areas.items():
            boxesIn, boxesOut = get_boxes_inside_areas2(row, rect2poly(field_coord, xxyy=False))
            # print(f'row{i}' ,field_name, boxesIn)
            # print('boxOut:', boxesOut)
            # print('-'*10)
            col.append(boxesIn)
        newRows.append(col)
    return newRows


def seperate(table_boxes_sorted, table, pixel=10, x_per_y_thresh=0.2):
    # seperating
    h_lines, v_lines = table
    new_table_boxes_sorted = []
    for i in range(len(table_boxes_sorted)):
        new_row_boxes = []

        # get the list of vertical lines which intersected with the box
        for j in range(len(table_boxes_sorted[i])):
            # check intersection
            box = table_boxes_sorted[i][j]
            x1,x2,y1,y2 = box
            box_polygon = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
            
            # get intersection line
            intersection_lines = []
            for v_line in v_lines:
                line = LineString([(v_line[0], v_line[1]), (v_line[2], v_line[3])])
                intersection_line = box_polygon.intersection(line)
                if not intersection_line.is_empty: # neu co cat
                    intersection_lines.append(intersection_line.coords[:])
            
            # seperate the box by intersection lines.
            if len(intersection_lines) > 0:
                new_boxes = []
                for i_line_index in range(len(intersection_lines)):
                    if i_line_index == 0:
                        # handle with first box.
                        new_box_x1 = x1
                    else:
                        new_box_x1 = int(intersection_lines[i_line_index-1][0][0]) + pixel
                        
                    new_box_x2 = int(intersection_lines[i_line_index][0][0]) - pixel
                    new_box_y1, new_box_y2 = y1, y2
                    new_box = [new_box_x1, new_box_x2, new_box_y1, new_box_y2]
                    if (new_box_x2-new_box_x1)/abs(new_box_y2-new_box_y1) > x_per_y_thresh:
                        new_boxes.append(new_box)
                        
                    # handle with last box
                    if i_line_index == len(intersection_lines)-1:
                        last_new_box_x1 = int(intersection_lines[i_line_index][0][0]) + pixel
                        last_new_box_x2 = x2
                        last_new_box = [last_new_box_x1, last_new_box_x2, new_box_y1, new_box_y2]
                        if (last_new_box_x2-last_new_box_x1)/abs(new_box_y2-new_box_y1) > x_per_y_thresh:
                            new_boxes.append(last_new_box)
                new_row_boxes.extend(new_boxes)
            # if don't have intersection line then keep the original box.
            else:
                new_row_boxes.append(box)
        new_table_boxes_sorted.append(new_row_boxes)
    return new_table_boxes_sorted
    
def merge(table_boxes_splitted):
    # merging
    for i in range(len(table_boxes_splitted)):
        for j in range(len(table_boxes_splitted[i])):
            colum = np.array(table_boxes_splitted[i][j])
            if len(colum) > 1:
                # check is same line again but increase the threshold to merge 2 boxes same line. 
                colum = sorted(colum, key=lambda x:x[2])
                for k in range(len(colum)-1):
                    if is_box_same_line(colum[k], colum[k+1], 0.5):
                        x1, x2, y1, y2 = colum[k]
                        next_x1, next_x2, next_y1, next_y2 = colum[k+1]
                        colum[k] = [0, 0, 0, 0]
                        colum[k+1] = [min(x1,next_x1), max(x2,next_x2), min(y1,y2), max(next_y1,next_y2)]    
                    else:
                        continue 
                
                # after merging, get the box which have maximum y1.
                table_boxes_splitted[i][j] = max(colum, key=lambda x:x[2])
                
                ## old version, merge all boxes of the cell.
                # X1 = colum[:,0]
                # Y1 = colum[:,2]
                # X2 = colum[:,1]
                # Y2 = colum[:,3]
                # startX, endX = min(np.concatenate((X1,X2))), max(np.concatenate((X1,X2)))
                # startY, endY = min(np.concatenate((Y1,Y2))), max(np.concatenate((Y1,Y2)))
                # table_boxes_splitted[i][j] = [startX, endX, startY, endY]
            # if a cell have one box, just get this one.
            elif len(colum) == 1:
                table_boxes_splitted[i][j] = table_boxes_splitted[i][j][0]
    return table_boxes_splitted

def seperate_and_merge(box_list, config_areas, table):
    # table_boxes_sorted = get_boxes_for_all_row(box_list, table)
    ### group text boxes for horizontal 
    table_boxes_sorted = get_boxes_with_lines_new(box_list)
    ### seperate text boxes by vlines
    seperated_table_boxes = seperate(table_boxes_sorted, table)

    ### group seperated boxes into 2 consecutive vlines
    # table_boxes_splitted = get_boxes_for_table_colum(seperated_table_boxes, config_areas, table)
    table_boxes_splitted = getCol(seperated_table_boxes, config_areas)

    ### merge group boxes to 1 box
    final_table_boxes = merge(table_boxes_splitted)
    
    # convert to the original boxes stucture to recognize them. 
    # get the address list of each text box, which is very helpful.
    original_boxes = []
    address_list = []
    for i, row in enumerate(final_table_boxes):
        for j, colum in enumerate(row):
            if len(colum) > 0:
                original_boxes.append(colum)
                address = [i,j]
                address_list.append(address)
    return original_boxes, address_list, table_boxes_splitted

def getTableBoxes(box_list, config_areas, table):
    ### get text boxes which inside the configure area
    box_list_filtered = []
    boxesOut = box_list
    for field_name, field_coord in config_areas.items():
        boxesIn, boxesOut = get_boxes_inside_areas2(box_list, rect2poly(field_coord, xxyy=False))
        box_list_filtered.extend(boxesIn)
    ### group text boxes for horizontal, don't need vlines or config
    rows = get_boxes_with_lines_new(box_list_filtered)
    ### seperate text boxes by vlines
    rows = seperate(rows, table)
    ### group seperated boxes into configure area
    cols = getCol(rows, config_areas)
    ### merge
    cols = merge2(cols, firstBox=True)
    ### get address and 1-d boxes list
    box_list = []
    address_list = []
    for i in range(len(cols)):
        for j in range(len(cols[i])):
            if len(cols[i][j]):
                box_list.append(cols[i][j])
                address = [i,j]
                address_list.append(address)
    return box_list, address_list, cols

def getTableBoxesDict(box_list, config_areas, vlines):
    ### 1. Get boxes which inside configure area
    fieldBoxes = dict()
    for field_name, field_coord in config_areas.items():
        boxesIn, boxesOut = get_boxes_inside_areas2(box_list, rect2poly(field_coord, xxyy=False))
        fieldBoxes[field_name] = boxesIn
    ###
    pass

def merge2(cols, firstBox=False):
    for i in range(len(cols)):
        for j in range(len(cols[i])):
            if len(cols[i][j]) != 0:
                if not firstBox:
                    cell = np.array(cols[i][j])
                    xmin, xmax = cell[:, (0,1)].min(), cell[:, (0,1)].max()
                    ymin, ymax = cell[:, (2,3)].min(), cell[:, (2,3)].max()
                    cols[i][j] = [xmin, xmax, ymin, ymax]
                else:
                    cols[i][j] = cols[i][j][0]
    return cols

def mergeBox(boxes):
    """
    box: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    while None in boxes:
        boxes.pop(boxes.index(None))
    boxes = np.array(boxes)
    boxes = boxes.reshape(-1, 2)
    x1 = min(boxes[:, 0])
    y1 = min(boxes[:, 1])
    x2 = max(boxes[:, 0])
    y2 = max(boxes[:, 1])
    return [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]

def poly2rect(poly, xxyy=True):
    """
    poly: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    rect: [x1, x2, y1, y2]
    """
    poly_arr = np.array(poly).astype(np.int32)
    if len(poly_arr.shape) == 1:
        return poly
    # x1, y1, x2, y2 = poly[0][0], poly[0][1], poly[2][0], poly[2][1]
    # print(poly_arr)
    xmin = poly_arr[:, 0].min()
    xmax = poly_arr[:, 0].max()
    ymin = poly_arr[:, 1].min()
    ymax = poly_arr[:, 1].max()

    if xxyy:
        return [xmin, xmax, ymin, ymax]
    else:
        return [xmin, ymin, xmax, ymax]

def rect2poly(rect, xxyy=True):
    """
    poly: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    rect: [x1, y1, x2, y2] / [x1, x2, y1, y2]
    """
    rect_arr = np.array(rect)
    if len(rect_arr.shape) == 2:
        return rect
    # x1, y1, x2, y2 = rect
    # poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if xxyy:
        xmin, xmax = rect_arr[[0, 1]].min(), rect_arr[[0, 1]].max()
        ymin, ymax = rect_arr[[2, 3]].min(), rect_arr[[2, 3]].max()
    else:
        xmin, xmax = rect_arr[[0, 2]].min(), rect_arr[[0, 2]].max()
        ymin, ymax = rect_arr[[1, 3]].min(), rect_arr[[1, 3]].max()
    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return poly

def isPolygonOverlap(poly1, poly2, iou_th=0.2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2)
    ### polygon or linestring?
    isOverlap = False
    if isinstance(intersection, Polygon):
        intersection_per_poly1 = intersection.area / min(poly1.area, poly2.area)
        if intersection_per_poly1 >= iou_th:
            isOverlap = True
    return isOverlap

def boxesOverlap(bbox1, bbox2):
    """
    y - coord overlap
    """
    bbox1_ymin, bbox1_ymax = bbox1[2], bbox1[3]
    bbox2_ymin, bbox2_ymax = bbox2[2], bbox2[3]
    overlap = max(0, min(bbox1_ymax, bbox2_ymax) - max(bbox1_ymin, bbox2_ymin))
    ratio = overlap / min(bbox1_ymax - bbox1_ymin, bbox2_ymax - bbox2_ymin)
    return ratio

def isBoxesOverlap(bbox1, bbox2, ratio_th=0.4):
    ratio = boxesOverlap(bbox1, bbox2)
    return ratio > ratio_th

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def boxesDistY(boxes):
    """
    box: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    boxes = np.array(boxes)
    boxes = sorted(boxes, key=lambda x: x[0][1]) # sort for y coord
    total_distY = 0
    for i in range(len(boxes)):
        if i==0:
            continue
        # distY = (min(boxes[i][:, 1]) + max(boxes[i][:, 1])) / 2 - \
        #         (min(boxes[i-1][:, 1]) + max(boxes[i-1][:, 1])) / 2
        distY = abs(min(boxes[i][:, 1]) - max(boxes[i-1][:, 1]))
        total_distY += distY
    return total_distY

