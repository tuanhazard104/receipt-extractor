# Import system libs
import os

from scipy.signal import find_peaks
import cv2
import numpy as np

# Import project packages
from arshot.aideploy.text_detection.CRAFT import craft_utils
from arshot.aideploy.text_detection.CRAFT import imgproc
from .craft import CRAFT

DEBUG = False

class CharacterSegmentation():
    def __init__(self):
        pass

    def cut_vertical(self, dst_img, dst_score_text):

        list_char_box = []

        h, w = dst_img.shape[0:2]
        if len(dst_score_text)!=0:
            # max_th3 = np.max(dst_score_text, 0)
            pj_th3 = np.sum(dst_score_text, 0)
        else:
            return []
        # pj_th3 = np.max(dst_score_text, 0)

        # local_mins = np.r_[True, pj_th3[1:] < pj_th3[:-1]] & np.r_[pj_th3[:-1] < pj_th3[1:], True]

        peaks, _ = find_peaks(pj_th3, prominence=0.001)

        dst_img_temp = dst_img.copy()

        # plt.plot(range(dst_img.shape[1]),pj_th3)
        # for idx in peaks:
        #     plt.axvline(x=idx)
        # plt.show()

        # cv2.imshow('',np.vstack((cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY),(dst_score_text*255.0).astype(np.uint8))))
        # cv2.waitKey()
        x1 = np.where(pj_th3 != 0)[0][0]
        i = 0
        # list_character = []
        if len(peaks) > 0:
            peak1 = peaks[i]
        while (i < len(peaks) - 1):
            peak2 = peaks[i + 1]
            if peak2 - peak1 < h/1.25 * 0.005:
                i = i + 1
                continue
            x2 = np.argmin(pj_th3[peak1:peak2]) + peak1

            if peak1 - x1 > 2 * (x2 - peak1):
                cut_x1 = peak1 - (x2 - peak1)
                cut_x2 = x2
            elif 2 * (peak1 - x1) < x2 - peak1:
                cut_x1 = x1
                cut_x2 = peak1 + (peak1 - x1)
            else:
                cut_x1 = x1
                cut_x2 = x2

            # if cut_x2 - cut_x1 > h/1.25*0.06:
            if cut_x2 - cut_x1 > 0:

                cut_x1 = np.clip(int(cut_x1 - 0.08 * h), 0, cut_x1)
                sub_crop = dst_img_temp[0:dst_img.shape[0], cut_x1:cut_x2].copy()
                list_char_box.append(sub_crop)

            x1 = x2
            peak1 = peak2
            i = i + 1


        cut_x1 = np.clip(int(x1 - 0.1 * h), 0, x1)
        cut_x2 = np.where(pj_th3 != 0)[0][-1]

        sub_crop = dst_img_temp[0:dst_img.shape[0], cut_x1:cut_x2].copy()
        list_char_box.append(sub_crop)
        # cv2.imshow('sub_crop', sub_crop)
        # cv2.waitKey()

        return list_char_box


    def segment(self, detected_box):
        """
        Cut Vertical
        :param detected_box: [box_id, poly_coords, warped_box, warped_score_text, warped_score_link]
        :return:
        """

        # We only get the warped_box and warped_score_text
        # detected_box[2] and detected_box[3]

        box_id, poly_coords, warped_box, warped_score_text= detected_box
        list_box = self.cut_vertical(warped_box, warped_score_text)

        return list_box
