import torch
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from .utils import get_image_list

class TextRecognizer:
    def __init__(self, model_name="vgg_transformer", gpu=True):
        if not gpu or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        config = Cfg.load_config_from_name(model_name)
        config['device'] = self.device
        self.recognizer = Predictor(config)
    
    def recognize(self, image, horizontal_list=None, free_list=None, batch_size=1):
        image_list, max_width = get_image_list(horizontal_list, free_list, image, model_height = self.recognizer.config['dataset']['image_height'])
        words = []
        for polygon, image in image_list:
            image = Image.fromarray(image).convert('RGB')
            text, confidence = self.recognizer.predict(image, return_prob=True)
            word = [polygon, text, confidence]
            words.append(word)
        return words

if __name__ == "__main__":
    recognizer = TextRecognizer()
    import cv2
    image = cv2.imread(r"E:\project\ARS\str\images\text_recognition\jp\2023-08-24-15-48-59_1_170-0003.jpg")
    words = recognizer.recognize(image)
    print(words)