from collections import OrderedDict

import numpy as np

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

EPS = 1e-6

def find_true_2d(x):
    """
    find value in 2d, return in 2 axis
    """
    out = []
    _, m = x.shape
    x_list = x.ravel()
    idxs = [idx for idx, value in enumerate(x_list) if value == 1]
    for idx in idxs:
        i, j = idx//m, idx % m
        out.append([i, j])
    return out

def create_overlap_mat(min_y, max_y):
    """
    create overlap mat
    """

    n_boxes = len(min_y)
    overlap_mat = np.zeros((n_boxes, n_boxes))
    for i in range(n_boxes):
        for j in range(i+1, n_boxes):

            overlap_rate = -(min_y[i]-max_y[j]) / (max_y[i] - min_y[j] + EPS)
            if overlap_rate < 0:
                overlap_rate = 0
            elif overlap_rate > 1:
                overlap_rate = 1./overlap_rate

            # if max_y[i] < max_y[j]:
            #     high = j
            #     low = i

            # else:
            #     high = i
            #     low = j
            # overlap_rate = (max_y[high] - min_y[low]) / (max_y[low] - min_y[high])
            # if overlap_rate < 0:
            #     overlap_rate = 0
            overlap_mat[i, j] = round(overlap_rate, 2)
    # print(overlap_mat)
    return overlap_mat


def create_dist_mat(min_x, min_y, max_x, max_y, wide_ratio_thresh=1.5):
    """
    Create dist mat
    """
    # dist rate over wide or height???!
    n_boxes = len(min_x)
    dist_mat = np.zeros((n_boxes, n_boxes))
    for i in range(n_boxes):
        for j in range(i+1, n_boxes):
            dist = min(abs(max_x[i] - min_x[j]), abs(max_x[j] - min_x[i]))
            overlap = np.sign((max_x[i] - min_x[j])*(min_x[i] - max_x[j]))
            if overlap == -1:
                # print('='*50)
                dist = 0
            dist = dist*overlap
            wide_box_i = (max_y[i] - min_y[i])
            #wide_box_i = ()
            wide_box_j = (max_y[j] - min_y[j])
            wide_ratio = wide_box_i / wide_box_j
            # 2 box must have similar size, ratio not too big or too small
            if wide_ratio > wide_ratio_thresh or wide_ratio < 1/wide_ratio_thresh:
                #dist_rate = -1
                dist_rate = dist / (wide_box_i + wide_box_j)
            else:
                dist_rate = dist / (wide_box_i + wide_box_j)  # case dist==0

            dist_mat[i, j] = round(dist_rate, 2) + EPS
    # print(dist_mat)
    return dist_mat

def find_group(boxes, overlap_thresh=0.3, dist_thresh=0.9):
    """
    Find group of boxes
    """
    assert boxes.shape[1:] == (4, 2)
    min_xy = np.min(boxes, axis=1)
    max_xy = np.max(boxes, axis=1)
    min_y = min_xy[:, 1]
    min_x = min_xy[:, 0]
    max_y = max_xy[:, 1]
    max_x = max_xy[:, 0]

    overlap_mat = create_overlap_mat(min_y, max_y)
    dist_mat = create_dist_mat(min_x, min_y, max_x, max_y)
    # print(overlap_mat)
    #print(dist_mat)
    #print(boxes[:, 0, 0])
    # for i in range(boxes.shape[0]):
    #    print(i, boxes[i, 0, 0], boxes[i, 1, 0])
    # print(boxes)
    #boxes = sorted(boxes, key=lambda x: x[0,0])
    #print(boxes[:, 0, 0])

    # overlap_mat[overlap_mat>=overlap_thresh] = 1
    # overlap_mat[overlap_mat<overlap_thresh] = 0

    # dist_mat[dist_mat<=dist_thresh] = 1 #??? not work
    # dist_mat[dist_mat>dist_thresh] = 0

    #print(overlap_mat[:5, :5])
    #print(dist_mat[:5, :5])
    overlap_mat = np.where(overlap_mat >= overlap_thresh,
                           1, 0)  # condition overlap
    dist_mat = np.where((dist_mat > 0) & (
        dist_mat <= dist_thresh), 1, 0)  # condition distance

    group_mat = overlap_mat * dist_mat
    # print(overlap_mat)
    # print(dist_mat)
    # print(group_mat)

    couples = find_true_2d(group_mat)  # list of couple box can group
    groups = []  # divide to group

    #print('Couple: ', couples)

    del_idx = []
    for i, _ in enumerate(couples):
        found = False
        group_found = -1
        # print(len(groups))
        for j, _ in enumerate(groups):
            if couples[i][0] in groups[j] or couples[i][1] in groups[j]:
                if not found:
                    groups[j].extend(couples[i])
                    group_found = j
                    found = True
                else:
                    groups[group_found].extend(groups[j])
                    del_idx.append(j)

                # break
        if not found:
            groups.append(couples[i])
    # del_idx = []
    # for i in range(len(groups)-1):
    #     for j in range(i, len(groups)):
    #         for idx in groups[j]:
    #             if idx in groups[i]:
    #                 groups[i].extend(groups[j])
    #                 del_idx.append(j)
    groups_tmp = []
    for idx, _ in enumerate(groups):
        if idx not in del_idx:
            groups_tmp.append(groups[idx])
    groups = groups_tmp.copy()

    groups = [list(set(group)) for group in groups]
    #print('Group: ', groups)
    return groups

def order_in_group(boxes, idxs):
    """
    find order of boxes in group
    order: left -> right
    order by tl x
    """
    assert boxes.shape[1:] == (4, 2)
    idxs = np.array(idxs)
    group = boxes[idxs]
    tl_x = group[:, 0, 0]  # get value of tl x
    order = np.argsort(tl_x)
    idxs = idxs[order]
    return idxs


def group_box(boxes, idxs):
    """
    merge boxes in group to only one polygon
    """
    assert boxes.shape[1:] == (4, 2)
    # print(idxs)
    idxs = np.array(idxs)
    group = boxes[idxs]
    #min_xy = np.min(group, axis=1)
    #max_xy = np.max(group, axis=1)
    #_, __, rect = transform(group)
    top_left = group[:, 0, :]
    top_right = group[:, 1, :]
    bottom_right = group[:, 2, :]
    bottom_left = group[:, 3, :]

    tl_most = np.min(top_left, axis=0)
    tr_most = [np.max(top_right, axis=0)[0], np.min(top_right, axis=0)[1]]
    br_most = np.max(bottom_right, axis=0)
    bl_most = [np.min(bottom_left, axis=0)[0], np.max(bottom_left, axis=0)[1]]

    tr_most = np.array([max(br_most[0], tr_most[0]),
                        min(tl_most[1], tr_most[1])])
    bl_most = np.array([min(tl_most[0], bl_most[0]),
                        max(br_most[1], bl_most[1])])
    rect = [tl_most, tr_most, br_most, bl_most]
    rect = np.array(rect)
    #print('Shape of rect', rect.shape)
    # print(group)
    # print(rect)
    # input()
    return rect


def bb_intersection_over_union(box1, box2):
    """
    Calculate iou between 2 boxes
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_1 = max(box1[0], box2[0])
    y_1 = max(box1[1], box2[1])
    x_2 = min(box1[2], box2[2])
    y_2 = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_2 - x_1 + 1) * max(0, y_2 - y_1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    # return the intersection over union value
    return iou
