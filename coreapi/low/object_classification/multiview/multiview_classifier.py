
import os
import errno
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

from .resnet import resnet18

def load_model(device, num_classes, model_path):
    print("[INFO] Loadding classifier weights:", model_path)
    model = resnet18(num_classes=num_classes)
    model.to(device)

    if model_path:
        checkpoint = torch.load(model_path)
        #best_acc = checkpoint['best_acc']
        #start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    return model

class MultiviewClassifier(object):
    def __init__(self, config):

        self.class_names = config.class_names
        self.model_path = config.model_path
        self.confidence_threshold = config.confidence

        if config.use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        if os.path.exists(self.model_path):
            self.model = load_model(self.device, len(self.class_names), self.model_path)
            self.model.eval()
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), ])

    def classify(self, images):
        inputs = []
        for img in images:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze_(0)
            inputs.append(img)
        with torch.no_grad():
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda(self.device)
            outputs = self.model(inputs)
            sorted_result, index = torch.sort(outputs.data, 1, descending=True)

        # Normalize to probability domain
        sorted_result = torch.nn.Softmax(dim=1)(sorted_result)
        sorted_result, index = sorted_result.squeeze(), index.squeeze()
        sorted_result = (sorted_result.cpu().tolist())
        index = index.cpu().tolist()

        # Check confidence threshold
        if sorted_result[0] >= self.confidence_threshold:
            category = self.class_names[index[0]]
        else:
            category = None

        list_candidates = []
        for id, sr in zip(index, sorted_result):
            list_candidates.append((id, sr))

        result = {
            "value": category,
            "candidates": list_candidates
        }
        return result