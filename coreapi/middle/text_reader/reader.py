""" File reader.py
    The service is to read all texts
"""
import os, sys
import cv2
import numpy as np
import re

from ...low.text_detection import TextDetector
# from ...low.text_recognition import TextRecognizer
# from ...low.strhub import TextRecognizer
from ...low.viet_ocr import TextRecognizer
from ...low.text_recognition.recognizer import get_image_list
from ...utility.utils import reformat_input, reformat_input_batched
from ...utility.imageproc.image import warp
from .box_utils import seperate_and_merge, get_boxes_inside_areas, get_boxes_inside_areas2, recover_boxes_single_digits, getTableBoxes

EXPECTED_H = 32

class TextReader(object):
    def __init__(self, lang_list, gpu=True, 
                 model_storage_directory=None, user_network_directory=None, 
                 detect_network="craft", recog_network='standard', 
                 quantize=True, cudnn_benchmark=False,
                 recog_model='standard'):
        """Create an Text Reader

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.
            gpu (bool): Enable GPU support (default)
            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            user_network_directory (string): Path to directory for custom network architecture.
            If not specified, it is as defined by the environment variable
        """
        self.detector = TextDetector(
                                    gpu=gpu, 
                                    model_storage_directory=model_storage_directory,
                                    detect_network="craft",
                                    quantize=quantize, cudnn_benchmark=cudnn_benchmark
                                    )
        self.recog_model = recog_model
        # self.recognizer = TextRecognizer(
        #                                 lang_list=lang_list,
        #                                 gpu=gpu,
        #                                 model_storage_directory=model_storage_directory,
        #                                 user_network_directory=user_network_directory,
        #                                 recog_network=recog_network,
        #                                 quantize=quantize,
        #                                 cudnn_benchmark=cudnn_benchmark
        #                                 )
        self.recognizer = TextRecognizer()
	        
    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        # # # horizontal_list, free_list = horizontal_list[0], free_list[0] 
        # result = self.recognizer.recognize(img_cv_grey, horizontal_list, free_list,\
        #                         decoder, beamWidth, batch_size,\
        #                         workers, allowlist, blocklist, detail, rotation_info,\
        #                         paragraph, contrast_ths, adjust_contrast,\
        #                         filter_ths, y_ths, x_ths, False, output_format)
        result = self.recognizer.recognize(img_cv_grey, horizontal_list, free_list)
        return result
    
    def readtextlang(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognizer.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)
       
        char = []
        directory = 'characters/'
        for i in range(len(result)):
            char.append(result[i][1])
        
        def search(arr,x):
            g = False
            for i in range(len(arr)):
                if arr[i]==x:
                    g = True
                    return 1
            if g == False:
                return -1
        def tupleadd(i):
            a = result[i]
            b = a + (filename[0:2],)
            return b
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open ('characters/'+ filename,'rt',encoding="utf8") as myfile:  
                    chartrs = str(myfile.read().splitlines()).replace('\n','') 
                    for i in range(len(char)):
                        res = search(chartrs,char[i])
                        if res != -1:
                            if filename[0:2]=="en" or filename[0:2]=="ch":
                                print(tupleadd(i))

    def readtext_batched(self, image, n_width=None, n_height=None,\
                         decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                         workers = 0, allowlist = None, blocklist = None, detail = 1,\
                         rotation_info = None, paragraph = False, min_size = 20,\
                         contrast_ths = 0.1, adjust_contrast = 0.5, filter_ths = 0.003,\
                         text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                         canvas_size = 2560, mag_ratio = 1.,\
                         slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                         width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                         threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                         output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        When sending a list of images, they all must of the same size,
        the following parameters will automatically resize if they are not None
        n_width: int, new width
        n_height: int, new height
        '''
        img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

        horizontal_list_agg, free_list_agg = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        result_agg = []
        # put img_cv_grey in a list if its a single img
        img_cv_grey = [img_cv_grey] if len(img_cv_grey.shape) == 2 else img_cv_grey
        for grey_img, horizontal_list, free_list in zip(img_cv_grey, horizontal_list_agg, free_list_agg):
            result_agg.append(self.recognizer.recognize(grey_img, horizontal_list, free_list,\
                                            decoder, beamWidth, batch_size,\
                                            workers, allowlist, blocklist, detail, rotation_info,\
                                            paragraph, contrast_ths, adjust_contrast,\
                                            filter_ths, y_ths, x_ths, False, output_format))

        return result_agg

    def readtext_custom(self, image, table_lines=None, config_areas=None, roi_inside_recog=None, recover_col=None,
                 decoder = 'greedy', beamWidth= 5, batch_size = 1, \
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.9,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        # ### debug
        # image_debug = img.copy()
        # for box in horizontal_list:
        #     box = np.absolute(np.array(box))
        #     x1, x2, y1, y2 = box
        #     cv2.rectangle(image_debug, (x1,y1), (x2,y2), (50,120,50), 2)
        # cv2.imwrite(f'debug_boxes.jpg', image_debug)

        hlines, vlines = table_lines
        if len(hlines) + len(vlines) > 0:
            print('if len(hlines) + len(vlines) > 0:', len(hlines), len(vlines))
        # if table_lines:
            ## Seperate and merge text boxes based on column lines
            # horizontal_list_refined, address_list, table_boxes_splitted = seperate_and_merge(horizontal_list, config_areas, table_lines)
            horizontal_list_refined, address_list, table_boxes_splitted = getTableBoxes(horizontal_list, config_areas, table_lines)

            # seperated_table_boxes = table_boxes_splitted
            # for i in range(len(seperated_table_boxes)):
            #     color = tuple(np.random.choice((range(256)), size=3).astype(np.int64).tolist())
            #     # seperated_table_boxes[i] = np.array(seperated_table_boxes[i])
            #     for j in range(len(seperated_table_boxes[i])):
            #         if len(seperated_table_boxes[i][j]):
            #             x1, x2, y1, y2 = seperated_table_boxes[i][j]
            #             cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)

        else:
            horizontal_list_refined = horizontal_list.copy()
            address_list = []

        # image_debug = img.copy()

        # for box in horizontal_list_refined:
        #     box = np.absolute(np.array(box))
        #     x1, x2, y1, y2 = box
        #     cv2.rectangle(image_debug, (x1,y1), (x2,y2), (50,120,50), 2)
        # cv2.imwrite(f'debug_boxes_mergeBoxUse.jpg', image_debug)
        
        if roi_inside_recog and config_areas:
            horizontal_list_filtered = []
            free_list_filtered = []
            for x, box in config_areas.items():
                x1, y1, x2, y2 = box
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                horizontal_list_filtered.extend(get_boxes_inside_areas2(horizontal_list_refined, polygon)[0])
                free_list_filtered.extend(get_boxes_inside_areas2(free_list, polygon)[0])
        else:
            horizontal_list_filtered = horizontal_list_refined.copy()
            free_list_filtered = free_list.copy()

        # image_debug = img.copy()
        # for box in horizontal_list_filtered:
        #     box = np.absolute(np.array(box))
        #     x1, x2, y1, y2 = box
        #     cv2.rectangle(image_debug, (x1,y1), (x2,y2), (50,120,50), 2)
        # cv2.imwrite(f'debug_boxes_obscureUse.jpg', image_debug)

        if recover_col and config_areas:
            x1, y1, x2, y2 = config_areas[recover_col["target"]]
            recover_config_area = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            x1, y1, x2, y2  = config_areas[recover_col["refer"]]
            refer_config_area = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            horizontal_list_filtered = recover_boxes_single_digits(img_cv_grey, horizontal_list_filtered, recover_config_area, refer_config_area)

        # image_debug = img.copy()
        # # print(image_debug.shape)
        # for box in horizontal_list_filtered:
        #     box = np.absolute(np.array(box))
        #     x1, x2, y1, y2 = box
        #     cv2.rectangle(image_debug, (x1,y1), (x2,y2), (120,120,50), 2)
        # cv2.imwrite(f'debug_boxes_recover.jpg', image_debug)


        if self.recog_model == 'custom':
            imgs, bboxes = [], []
            org_h, org_w = img.shape[:2]
            for bbox in horizontal_list_filtered:
                xmin, xmax, ymin, ymax = bbox
                xmin = max(0, xmin)
                xmax = min(org_w, xmax)
                ymin = max(0, ymin)
                ymax = min(org_h, ymax)
                if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
                    continue
                bboxes.append([[xmin,ymin],[xmax,ymin], [xmax,ymax],[xmin,ymax]])
                imgs.append(img[ymin:ymax, xmin:xmax])

            input_patches = []
            for box_id, bbox in enumerate(free_list_filtered):
                warp_img = warp(img, bbox)
                warp_img_h, _ = warp_img.shape[:2]
                ratio = EXPECTED_H / warp_img_h
                warp_img = cv2.resize(warp_img, None, fx=ratio, fy=ratio)
                input_patches.append([box_id, warp_img.shape[1], warp_img])
            input_patches = sorted(input_patches, key=lambda x:x[1])
            warp_imgs = [patch[2] for patch in input_patches]
            imgs.extend(warp_imgs)

            result = []
            if len(imgs) > 0: # If there is text images to recognize
                res = self.recognizer.recognize(imgs)
                for bbox, ele in zip(bboxes, res):
                    text, conf = ele
                    result.append((bbox, text, conf))
        else:
            result = self.recognizer.recognize(img_cv_grey, horizontal_list_filtered, free_list_filtered,\
                                    decoder, beamWidth, batch_size,\
                                    workers, allowlist, blocklist, detail, rotation_info,\
                                    paragraph, contrast_ths, adjust_contrast,\
                                    filter_ths, y_ths, x_ths, False, output_format)

        return result, address_list
    
    def readtext_custom2(self, image, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard'):
        img, img_cv_grey = reformat_input(image)
        horizontal_list, free_list = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        image_list, max_width = get_image_list(horizontal_list[0], free_list[0], img_cv_grey)
        ### get image list
        imgs = []
        boxes = []
        for x in image_list:
            box, img_cropped = x
            imgs.append(img_cropped)
            boxes.append(box)
        ### get text
        results = []
        if len(imgs) > 0: # If there is text images to recognize
            res = self.recognizer.recognize(imgs)
            for bbox, ele in zip(boxes, res):
                text, conf = ele
                results.append((bbox, text, conf))
        return results
