"""
    (C) Copyright 2023 ARS Group (Advanced Research & Solutions)
    File orient_classifier.py
        Classify the orientation of business form paper among 4 classes: 0 Deg, 90 Deg, 180 Deg and 270 Deg
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models

from .utils import img_to_tensor

# Configure the model path
g_dir_path = os.path.dirname(os.path.realpath(__file__))
g_model_path = os.path.join(g_dir_path, 'models/form_orientation_20240304.pth')
CLASS_NAMES = ['0', '180', '270', '90']

class PaperOrientationClassifier:
    """ Classify the orienation of business form or invoice object
    """
    def __init__(self, gpu=True, model_path=g_model_path):
        
        if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        self.min_size = 224
        self.class_names = CLASS_NAMES

        print("[INFO] Loading Business Form Orienation Classifier Model", model_path)
        self.model = models.mnasnet1_0()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=len(self.class_names))
        )

        self.model.to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()


    def classify(self, img):
        """ This function classify the orientation of business form in the input image

        Args:
            img (ndarray): Input image

        Returns:
            class_name (str)
        """
        scores = {
            self.class_names[0]: 0,
            self.class_names[1]: 0,
            self.class_names[2]: 0,
            self.class_names[3]: 0
        }

        tensor1 = img_to_tensor(img, self.device, self.min_size)

        # Rotate 90
        rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        tensor2 = img_to_tensor(rot_img, self.device, self.min_size)

        # Rotate 180
        rot_img = cv2.rotate(img, cv2.ROTATE_180)
        tensor3 = img_to_tensor(rot_img, self.device, self.min_size)

        # Rotate 270
        rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        tensor4 = img_to_tensor(rot_img, self.device, self.min_size)

        input_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), 0)
        with torch.no_grad():
            out = self.model(input_tensor)
            predict = out.cpu().numpy()
        
        '''
        Score 
                    0    270   180   90  
        tensor1     0    270   180   90
        tensor2     270  180   90    0
        tensor3     180  90    0     270
        tensor4     90   0     270   180
        '''
        scores[self.class_names[0]] = predict[0][0] + predict[1][3] + predict[2][1] + predict[3][2]
        scores[self.class_names[1]] = predict[0][1] + predict[1][2] + predict[2][0] + predict[3][3]
        scores[self.class_names[2]] = predict[0][2] + predict[1][0] + predict[2][3] + predict[3][1]
        scores[self.class_names[3]] = predict[0][3] + predict[1][1] + predict[2][2] + predict[3][0]

        # Get max score
        max_score = 0
        rst = self.class_names[0]
        for key, value in scores.items():
            if value > max_score:
                max_score = value
                rst = str(key)

        return rst, scores

if __name__ == "__main__":
    import glob
    from tqdm import tqdm

    classifier = PaperOrientationClassifier()
    
    image_dir = '/hdd/tuannca/ocr/docs_orientation/data_val/resized/'
    image_paths = glob.glob(f'{image_dir}/*/*.jpg') + glob.glob(f'{image_dir}/*/*.png')

    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)

    no_fail_cases = 0
    with open(f'{output_dir}/fail_cases.txt', 'w') as f:
        for i, image_path in enumerate(tqdm(image_paths)):
            img = cv2.imread(image_path)

            gt_angle = image_path.split('/')[-2].split('deg')[-1]

            pred_angle, scores = classifier.classify(img)
            
            if pred_angle != gt_angle:
                img_name = f'val_true{gt_angle}_pred{pred_angle}_{i}.jpg'
                cv2.imwrite(os.path.join(output_dir, img_name), img)

                no_fail_cases += 1
                f.write(image_path + '\n')

                print
    
    print(f'Accuracy: {(len(image_paths) - no_fail_cases)} / {(len(image_paths))}')
