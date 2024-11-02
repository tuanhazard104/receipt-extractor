import os, sys
import numpy as np
import cv2
import torch
from PIL import Image

from .utils import get_image_list
from .models.utils import load_from_checkpoint, parse_model_args, _get_config
from .data.module import SceneTextDataModule
from .models import PARSeq


class TextRecognizer:
    def __init__(self, model_name="parseq", gpu=True):
        if not gpu or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        config = _get_config(model_name)
        self.recognizer = PARSeq(**config)
        self.recognizer.load_state_dict(torch.load("coreapi/low/strhub/weights/parseq-bb5792a6.pt"))
        self.recognizer.eval().to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.recognizer.hparams.img_size)
    
    def recognize(self, image, horizontal_list=None, free_list=None, batch_size=1):
        image_list, max_width = get_image_list(horizontal_list, free_list, image, model_height = self.recognizer.hparams.img_size[0])
        words = []
        for polygon, image in image_list:
            image = Image.fromarray(image).convert('RGB')
            image = self.img_transform(image).unsqueeze(0).to(self.device)
            # p = self.recognizer(image).softmax(-1)
            # pred, p = self.recognizer.tokenizer.decode(p)
            logits = self.recognizer(image)
            probs = logits.softmax(-1)
            preds, probs = self.recognizer.tokenizer.decode(probs)
            confidence = 0
            for pred, prob in zip(preds, probs):
                confidence += prob.prod().item()
            word = [polygon, preds[0], confidence]
            words.append(word)
        return words
        

if __name__ == "__main__":
    recognizer = TextRecognizer()