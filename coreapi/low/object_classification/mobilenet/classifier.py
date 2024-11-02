import os 
import cv2
import numpy as np
import json
import torch
import os
import glob

from .mobilenetv2 import get_model

dir_path = os.path.dirname(os.path.realpath(__file__))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_min(img, min_size=224):
    h, w = img.shape[:2]
    if h > w:
        ratio = min_size / w
    else:
        ratio = min_size / h
    return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

def img_to_tensor(image, device):
    img = image.copy()
    img = resize_min(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = (img - mean) / std
    inp = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(inp).float().to(device).unsqueeze(0)

with open(os.path.join(dir_path, 'model_info.json'), 'r') as f:
    model_info = json.load(f)[-1] # Latest
    model_path = model_info['model']
    class_names = model_info['classes']

class Classifier:

    def __init__(self, config):
        self.model_path = config.model_path
        self.conf_thres = config.conf_thres
        self.device = config.device
        self.class_names = config.class_names

        if config.model_path is None:
            weight = model_path
        weight_path = os.path.join(dir_path, weight)
        print("Loading classifier weights: ", weight_path)
        self.model = get_model(self.device, len(self.class_names), weight_path)
        
        self.model.eval()

    def classify(self, img):
        inputs = img_to_tensor(img, self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        
        score = torch.softmax(outputs, 1)
        confidence, preds = torch.max(score, 1)
        fruit_clsname = self.class_names[preds]         
        return fruit_clsname, confidence
