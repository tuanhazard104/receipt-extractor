""" File line_utils.py
    The functions for processing the line
"""
import numpy as np
import math
import cv2

from sympy import Point, Line
from sympy import Point as sympyPoint
from shapely.geometry import LineString
from shapely.geometry import Point as geometryPoint

def get_line_by_angle(lines, angle, limit = 5):
    new_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            spl = math.degrees(math.atan2((y1 - y2), (x1 - x2))) % 360
            if (angle - limit < spl < angle + limit) or (angle + 180 - limit < spl < angle + 180 + limit):
                new_lines.append(line)
    else:
        return None
    return new_lines

def dist_between_2_group(group1, group2):
    ### gia su ca 2 da dc sap xep theo x
    group1_maxX = np.array(group1)[:,0,2].max()
    group2_minX = np.array(group2)[:,0,0].min()
    return group2_minX-group1_maxX

def get_groupping_lines(lines, thresh, vertical:bool):
    groups = []
    group = [[]]
    if lines is None:
        return groups
    if vertical:
        lines = group_sort(lines, for_y=False)
    else:
        lines = group_sort(lines, for_y=True)

    # groupping process
    for i in range(len(lines)):
        if i==0:
            # init first group
            group = [lines[i]]
            continue
        if not vertical:
            dist = lines[i][0][1] - max(np.array(group)[:,0,3]) # distY
        else:
            dist = min(lines[i][0][0], lines[i][0][2]) - max(np.append(np.array(group)[:,0,2], np.array(group)[:,0,0])) # distX

        if dist <= thresh:
            # add current line to the current group
            group.append(lines[i])
        else:
            if not vertical:
                # tiep tuc check line tiep sau no, neu van k t/m thi moi tao group moi
                if i==len(lines)-1:
                    next_distY = dist
                else:
                    next_distY = lines[i+1][0][1] - max(np.array(group)[:,0,3]) # distY
                if next_distY > thresh:
                    # init new group
                    groups.append(group)
                    group = [lines[i]]
            else:
                # # check lan nua cho chac cu'
                # if len(groups)>0:
                #     dist_gr = dist_between_2_group(groups[-1], group) 
                #     if dist_gr <= thresh:
                #         groups[-1].extend(group)
                #     else:
                #         groups.append(group)
                #         group = [lines[i]]
                # else:
                groups.append(group)
                group = [lines[i]]
                ### debug here   
    # add last group, which have not added before.
    groups.append(group)
    return groups

def get_correct_lines_from_group(groups, topbot, vertical:bool):
    # get correct lines
    correct_lines = []
    for i in range(len(groups)):
        (startX, startY, endX, endY) = get_line_from_group(groups[i], vertical=vertical, topbot=topbot, cluster=True)
        correct_lines.append([startX, startY, endX, endY])
    return correct_lines

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

        # distAB = abs(a*x1A+b*y1A+c)/math.sqrt(a**2+b**2)
        distAB = min(x1B, x2B) - max(x1A, x2A)

        if distAB > thresh:
            new_lines.append(line)            
            
    # add last line.
    if not len(lines):
        return []
    new_lines.append(lines[-1])
    return new_lines

def get_lines_by_distance(lines, low_ratio=0.6, high_ratio=3):
    length_list = []
    for line in lines:
        x1A, y1A, x2A, y2A = line
        length_line = math.sqrt((x2A - x1A) ** 2 + (y2A - y1A) ** 2)
        length_list.append(length_line)
    '''
    org_length_list = length_list.copy()
    length_list.sort()
    list1 = np.array(length_list[1:])
    list2 = np.array(length_list[:-1])
    derivative =  np.subtract(list1, list2)

    first_max_derivate = max(derivative)
    second_max_derivate = max([a for i,a in enumerate(derivative) if a < first_max_derivate])
    index1 = derivative.tolist().index(first_max_derivate)
    index2 = derivative.tolist().index(second_max_derivate)
    length_selected = length_list[1:][min(index1, index2):max(index1, index2)]

    new_lines = []
    for i, line in enumerate(lines):
        if org_length_list[i] in length_selected:
            new_lines.append(line)
    '''
    length_mean = np.mean(length_list)
    new_lines = []
    for i, line in enumerate(lines):
        length_ratio = length_list[i] / length_mean
        if low_ratio < length_ratio < high_ratio:
            new_lines.append(line)
  
    return new_lines

def vLines_refined(vLines, thresh=50, vertical = True):
    # print("in vlines_refined function, len(vlines)=", len(vLines))
    # sorted, key = y1
    if vertical:
        vLines_sorted = sorted(vLines, key=lambda x: x[1])
    else: # horizontal line
        vLines_sorted = sorted(vLines, key=lambda x: x[0])
    groups = []
    group = []
    for i in range(len(vLines_sorted)):
        startY = vLines_sorted[i][1]
        prev_startY = vLines_sorted[i-1][1]
        if i==0:
            group = [vLines_sorted[i]]
            continue
        # add current line to the current group.
        if startY - prev_startY < thresh:
            group.append(vLines_sorted[i])
        # stop groupping, init new group
        else:
            groups.append(group)
            group = [vLines_sorted[i]]
    # add last group
    groups.append(group)

    # # choose the group which have 2 lines or more.
    # new_groups = [group for group in groups if len(group)>=3]
    
    # choose the group which have maximum lines.
    best_group = max(groups, key=lambda x: len(x))
    if len(best_group) == 0:
        return vLines
    if len(groups) > 0:
        # check min y1 in best group
        if vertical:
            min_startY = best_group[0][1]
            # min_startY = min_startY - 10
            # vLine refine
            for j in range(len(vLines)):
                startX, startY, endX, endY = vLines[j]
                vLines[j] = [endX, min_startY, endX, endY]
        else:
            min_startX = best_group[0][0]
            # min_startX = min_startX - 10
            for j in range(len(vLines)):
                startX, startY, endX, endY = vLines[j]
                vLines[j] = [min_startX, startY, endX, endY]
    return vLines

def get_group_length(group, vertical):
    # preprocess
    for line in group:
        if vertical:
            # sap xep lai theo thu tu y1<y2
            if line[0][1] > line[0][3]:
                line[0] = [line[0][2], line[0][3], line[0][0], line[0][1]]
            # sap xep cac lines theo thu tu y1 tang dan
            group = sorted(group, key=lambda x: x[0][1])
        else:
            # sap xep lai theo thu tu x1<x2
            if line[0][0] > line[0][2]:
                line[0] = [line[0][2], line[0][3], line[0][0], line[0][1]]
            # sap xep cac lines theo thu tu x1 tang dan
            group = sorted(group, key=lambda x: x[0][0])
    # get coordinates
    if vertical:
        start_index = np.argmin(np.array(group)[:,0,1])
        end_index = np.argmax(np.array(group)[:,0,3])
    else:
        start_index = np.argmin(np.array(group)[:,0,0])
        end_index = np.argmax(np.array(group)[:,0,2])
    startX, startY = group[start_index][0][0], group[start_index][0][1]
    endX, endY = group[end_index][0][2], group[end_index][0][3]
    return (startX, startY, endX, endY)

def get_line_from_group(group, vertical, topbot=None, cluster=None):
    # preprocess
    for line in group:
        if vertical:
            # sap xep lai theo thu tu y1<y2
            if line[0][1] > line[0][3]:
                line[0] = [line[0][2], line[0][3], line[0][0], line[0][1]]
            # sap xep cac lines theo thu tu y1 tang dan
            group = sorted(group, key=lambda x: x[0][1])
        else:
            # sap xep lai theo thu tu x1<x2
            if line[0][0] > line[0][2]:
                line[0] = [line[0][2], line[0][3], line[0][0], line[0][1]]
            # sap xep cac lines theo thu tu x1 tang dan
            group = sorted(group, key=lambda x: x[0][0])
    # if cluster is None:
    x1_min, x1_max = np.min(np.array(group)[:,0,0]), np.max(np.array(group)[:,0,0])
    x2_min, x2_max = np.min(np.array(group)[:,0,2]), np.max(np.array(group)[:,0,2])
    y1_min, y1_max = np.min(np.array(group)[:,0,1]), np.max(np.array(group)[:,0,1])
    y2_min, y2_max = np.min(np.array(group)[:,0,3]), np.max(np.array(group)[:,0,3])
    x1_avg = int((x1_min + x1_max) / 2)
    x2_avg = int((x2_min + x2_max) / 2)
    y1_avg = int((y1_min + y1_max) / 2)
    y2_avg = int((y2_min + y2_max) / 2)
    # y_avg = int((y1_avg + y2_avg) / 2)
    # hor_line = LineString([(0, y_avg), (9999, y_avg)])
    # avg_line = LineString([(0, y_avg), (9999, y_avg)])
    if cluster:
        dbs = DBSCAN1D(eps=2, min_samples=10)
        X1 = np.array(group)[:,0,0]
        X2 = np.array(group)[:,0,2]
        # X1
        points, noises = cluster_points(dbs, X1)
        if len(points):
            points_density = max(points, key=lambda x: len(x))
            x1_avg = min(points_density)
        # X2
        points, noises = cluster_points(dbs, X2)
        if len(points):
            points_density = max(points, key=lambda x: len(x))
            x2_avg = max(points_density)
        
    if topbot:
        top, bot = topbot
        top_hor_line = LineString([(0, top), (9999, top)])
        bot_hor_line = LineString([(0, bot), (9999, bot)])
        old_line = LineString([(x1_avg, y1_avg), (x2_avg, y2_avg)])
        y1_intersec = top_hor_line.intersection(old_line)
        if not y1_intersec.is_empty:
            x1, y1 = y1_intersec.x, y1_intersec.y
        else:
            x1, y1 = x1_avg, top
        y2_intersec = bot_hor_line.intersection(old_line)
        if not y2_intersec.is_empty:
            x2, y2 = y2_intersec.x, y2_intersec.y
        else:
            x2, y2 = x2_avg, bot 
        
    return(x1, y1, x2, y2)

# def get_line_from_group2(group, topbot=None):
#     X1 = np.array(group)[:,0,0]
#     X2 = np.array(group)[:,0,2]
#     dbs = DBSCAN1D(eps=2, min_samples=10)
#     # X1
#     points, noises = cluster_points(dbs, X1)
#     if len(points):
#         x1 = 
    

### need to improve this function
def is_group_noise(group, max_length, length_thresh_ratio=0.2, num_of_group_thresh=3, topbot=None):
    group = np.array(group)
    # group = sorted(group, key=lambda x: x[0][0])
    
    # # first we need to cluster group in group
    # groups_in_group = get_groupping_lines(group, thresh=4, vertical=False)
    # if len(groups_in_group) > num_of_group_thresh and get_length(group) / max_length <= length_thresh_ratio:
    group_length = get_length(group)

    if group_length / max_length <= length_thresh_ratio:
        # if get_length(max(groups_in_group, key=lambda x:get_length(x))) / max_length > 0.1:
        #     is_noise = False
        # else:
        #     is_noise = True
        is_noise = True
    else:
        is_noise = False
    
    return is_noise
    

    # group_length = get_length(group)
    # if group_length < thresh:
    #     return True
    # return False   
    
def is_group_noise2(group, ratio_thresh=0.8, width_thresh=10, topbot=None):
    group = group_sort(group, for_y=True)
    group_length = get_length(group)
    group_width = get_width(group)
    score = 0
    count = 0
    if topbot is not None:
        thresh = -50
        group = np.array(group)
        top, bot = topbot
        y1_min = group[:,0,1].min()
        y2_max = group[:,0,3].max()
        if y1_min - bot > thresh or top - y2_max > thresh:
            is_noise = True
            ratio = -1
            return is_noise, ratio
    
    for line_i in range(len(group)):
        if line_i==0:
            continue
        dist = group[line_i][0][1] - group[line_i-1][0][3]
        dist = max(0, dist)
        score += dist
        if group[line_i][0][1] - group[line_i-1][0][3] > 0:
            count+=1
    # print('-'*10, len(group))
    # groups_in_group = get_groupping_lines(group, thresh=2, vertical=False)
    ratio = score / group_length
    if ratio < ratio_thresh:
        # if group_width > width_thresh:
        #     is_noise = True
        # else:
        is_noise = False
    else:
        is_noise = True
    return is_noise, ratio

# def refine_group(group):
    
    
    
def group_sort(group, for_y=True):
    if for_y:
        group = [[line[0][[2,3,0,1]]] if line[0][3] < line[0][1] else line for line in group]
        group = sorted(group, key=lambda x: x[0][1])
    else:
        group = [[line[0][[2,3,0,1]]] if line[0][2] < line[0][0] else line for line in group]
        group = sorted(group, key=lambda x: x[0][0])
    
    group = np.array(group)
    return group

def groups_sort(groups, for_y=True):
    for group in groups:
        group = np.array(group)
        for i in range(len(group)):
            x1, y1, x2, y2 = group[i][0]
            if for_y:
                if y1 > y2:
                    group[i] = [[x2, y2, x1, y1]]
            else: # for x
                if x1 > x2:
                    group[i] = [[x2, y2, x1, y1]]
    return groups

def get_length(group):
    group = np.array(group)
    group_length = 0
    for line in group:
        x1, y1, x2, y2 = line[0]
        line_length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        group_length += line_length
    return group_length
    # minY = np.min(group[:, :, 1])
    # maxY = np.max(group[:, :, 3])
    # return abs(maxY-minY)
    
def get_width(group):
    group = np.array(group)
    group = group_sort(group, for_y=False)
    minX = min(group[:, 0, 0])
    maxX = max(group[:, 0, 2])
    return maxX - minX

def get_avg_length(groups):
    groups = np.array(groups)
    groups_length = np.array([get_length(group) for group in groups])
    avg_length = groups_length.mean()
    return avg_length

def get_max_length(groups):
    groups = np.array(groups)
    groups_length = np.array([get_length(group) for group in groups])
    max_length_index = np.argmax(groups_length)
    max_length = groups_length[max_length_index]
    return max_length_index, max_length

def remove_noise(groups, length_thresh=None, topbot=None):
    groups = groups_sort(groups)
    avg_length = get_avg_length(groups)
    _, max_length = get_max_length(groups)
    length_thresh = avg_length / 2 if length_thresh is None else length_thresh

    group_lengths = []
    groups_noise = []
    groups_correct = []
    for i, group in enumerate(groups):
        group_lengths.append(get_length(group))
        if is_group_noise(group, max_length, topbot=topbot):
            groups_noise.append(group)
            # groups[i] = None
        else:
            groups_correct.append(group)
            
    # groups = [group for group in groups if group is not None]
    return groups_correct, groups_noise, group_lengths
    # return groupss
    
def remove_noise2(groups, topbot=None):
    gr_lengths = []
    for i, group in enumerate(groups):
        is_noise, gr_length =  is_group_noise2(group, topbot=topbot)
        if is_noise:
            groups[i] = None
        else:
            gr_lengths.append(gr_length)
            
    groups = [group for group in groups if group is not None]
    return groups, gr_lengths

def revert_group_from_noise(groups_noise, topbot, groups_correct=None, y1_thresh=20, y2_thresh=20):
    groups_noise_revert = []
    ### assump that topbot was found perfectly
    for group in groups_noise:
        groups_in_group = get_groupping_lines(group, thresh=2, vertical=False)
        max_density_group = max(groups_in_group, key=lambda x:len(x))
        max_density_group = group_sort(np.array(max_density_group), for_y=True)
        max_density_group_length = max(max_density_group[:, 0, 3]) - min(max_density_group[:, 0, 1])
        # group_last = groups_in_group[-1]
        # group_last = group_sort(group_last, for_y=True)
        # if len(group_last) > 3 and abs(Y2_max-topbot[1])<20:
        #     groups_noise_revert.append(group_last)
        group = group_sort(group, for_y=True)
        group_dist_Y2 = np.array([(topbot[1] - line[0][3]) for line in group])
        group_dist_Y1 = np.array([(line[0][1] - topbot[0]) for line in group])
        
        min_distY1_index = np.argmin(np.absolute(group_dist_Y1))
        min_distY2_index = np.argmin(np.absolute(group_dist_Y2))
        
        cond_top = abs(group_dist_Y1[min_distY1_index]) < y1_thresh and group[min_distY2_index][0][3] - topbot[0] > y1_thresh*6
        cond_bot = abs(group_dist_Y2[min_distY2_index]) < y2_thresh and topbot[1] - group[min_distY1_index][0][1] > y2_thresh*6
        
        if (cond_top or cond_bot) and max_density_group_length > 100:
            # length = group_dist_Y2[min_distY2_index] - topbot[0]
            # x1, y1, x2, y2 = group[min_distY1_index][0]
            # img_warped3 = cv2.putText(img_warped3, f'{length:.2f}', (x1, int(y1)-50), 
            #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1, cv2.LINE_AA)
            groups_noise_revert.append(group)

    # print(len(groups_noise_revert))
    return groups_noise_revert
    

import matplotlib.pyplot as plt
from dbscan1d.core import DBSCAN1D
def get_top_bot(groups):
    Y1_min, Y2_max, Y1_max, Y2_min = None, None, None, None
    # preprocess
    Y1s = []
    Y2s = []
    for group in groups:
        group = np.array(group)
        for i in range(len(group)):
            x1, y1, x2, y2 = group[i][0]
            if y1 > y2:
                group[i] = [[x2, y2, x1, y1]]
            minY1 = min(group[:,0,1])
            maxY2 = max(group[:,0,3])
        Y1s.append(minY1)
        Y2s.append(maxY2)
    Y1s = np.array(Y1s)
    Y2s = np.array(Y2s)
        
    dbs = DBSCAN1D(eps=10, min_samples=5)
    # Y1
    point_groups, noises = cluster_points(dbs, Y1s)
    point_groupsY1 = sorted(point_groups, key=lambda x: len(x))
    if len(point_groupsY1):
        point_group_max = np.array(point_groupsY1[-1])
        Y1_min = point_group_max.min()
        Y1_max = point_group_max.max()
    # Y2
    point_groups, noises = cluster_points(dbs, Y2s)
    point_groupsY2 = sorted(point_groups, key=lambda x: len(x))
    if len(point_groupsY2):
        point_group_max = np.array(point_groupsY2[-1])
        Y2_max = point_group_max.max()
        Y2_min = point_group_max.min()
    return [Y1_min, Y2_max], [Y1_max, Y2_min]
    
def cluster_points(dbs, points):
    # init dbscan object
    
    # get labels for each point
    labels = dbs.fit_predict(points)
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    # show core point indices
    dbs.core_sample_indices_

    # get values of core points
    dbs.components_

    unique_labels = list(set(labels))
    
    noises = [point for point, label in zip(points, labels) if label==-1]

    point_groups = []
    for unique_label in unique_labels:
        point_group = [point for point, label in zip(points, labels) if label==unique_label and label!=-1]
        if len(point_group) > 0: 
            point_groups.append(point_group)
        
    return point_groups, noises

def line_group_cluster(lines_group):
    '''
    cluster lines in 1 group
    '''
    
def get_groupping_lines2(lines, thresh, min_sample=10, vertical=True):
    groups_correct_index = []
    groups_noise_index = []
    lines = group_sort(lines, for_y=False)
    lines_outside = lines
    for line_index, line in enumerate(lines):
        if line_index in [item for sublist in groups_correct_index for item in sublist]:
            continue
        lines_inside_index, lines_outside_index = find_line_inside_thresh(line, lines, thresh)
        if len(lines_inside_index) > min_sample:
            groups_correct_index.append(lines_inside_index)
        else:
            groups_noise_index.append(lines_outside_index)
    return groups_correct_index, groups_noise_index

def find_line_inside_thresh(line0, lines, thresh):
    # line_inside_thresh_index = []
    lines_inside_index = []
    lines_outside_index = []
    x_range = [line0[0][0] - thresh, line0[0][2] + thresh]
    for line_i, line in enumerate(lines):
        if x_range[0] < line[0][0] and line[0][1] < x_range[1]:
            lines_inside_index.append(line_i)
        else:
            lines_outside_index.append(line_i)
    return lines_inside_index, lines_outside_index

def check_line_in_group(line, group):
    is_in = np.any(np.all(line == group, axis=1))

class LineGroupping:
    def __init__(self) -> None:
        pass
    
    def find_line_inside(self, line0, lines, lines_inside_index, thresh):
        # lines_inside_index = []
        lines_inside = []
        lines_outside = []
        x_range = [line0[0][0] - thresh, line0[0][2] + thresh]
        for line_i, linee in enumerate(lines):
            if len(lines[line_i][0]) > 0:
                if x_range[0] <= linee[0][0] and linee[0][1] <= x_range[1]:
                    lines_inside.append(linee)
                    lines_inside_index.append(line_i)
                    lines[line_i] = [[]]
                else:
                    lines_outside.append(linee)
        # lines = np.array([line for line in lines if len(line[0])>0])
        return lines_inside, lines_inside_index, lines
    
    def get_groups(self, lines, thresh, min_sample):
        groups_correct = []
        groups_noise = []
        lines_inside_index = []
        
        lines_outside0 = lines.copy()
        lines = group_sort(lines, for_y=False)
        c=0
        for ii, line in enumerate(lines):
            # if line in [item for sublist in groups_correct for item in sublist] or line in [item for sublist in groups_noise for item in sublist]:
            #     continue
            if ii in lines_inside_index:
                c+=1
                continue
            else:
                lines_inside, lines_inside_index, lines_outside0 = self.find_line_inside(line, lines_outside0, lines_inside_index, thresh)
                if len(lines_inside) >= min_sample:
                    groups_correct.append(lines_inside)
                else:
                    groups_noise.append(lines_inside)

        return groups_correct, groups_noise   
        