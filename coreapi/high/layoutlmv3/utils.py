import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def quad_to_box(quad):
    quad = np.array(quad)
    if len(quad) == 8:
        Xs, Ys = quad[::2], quad[1::2]
    else:
        Xs, Ys = quad[:, 0], quad[:, 1]
    x1, x2 = max(0, min(Xs)), max(Xs)
    y1, y2 = max(0, min(Ys)), max(Ys)
    box = [x1, y1, x2, y2]
    return tuple(box)

def preprocess_intput(image, words):
    # image = Image.fromarray(image).convert("BGR")
    names=["O", "B-TOTAL_COST", "I-TOTAL_COST", "B-OTHER", "I-OTHER", "B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "B-TIMESTAMP", "I-TIMESTAMP"]
    words_layoutlm = []
    boxes = []
    ner_tags = []
    for word in words:
        bbox, text, conf = word
        words_layoutlm.append(text)
        boxes.append(normalize_bbox(quad_to_box(bbox), image.size))
    return image, words_layoutlm, boxes

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def visualize(image, outputs):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("application/views/resources/font/CharisSIL-Regular.ttf", 18)
    label2color = {"total_cost": "red", "seller": "green", "address": "purple", "timestamp": "blue", "other": None}
    # for prediction, box, word in zip(field_names, boxes, words):
    for box, word, predicted_label, conf_score in outputs:
        if predicted_label == "other":
            continue
        draw.rectangle(box, outline=label2color[predicted_label])
        visualize_text = f"{predicted_label}: {word}"
        draw.text((box[0]+10, box[1]-10), text=visualize_text, fill=label2color[predicted_label], font=font)
    return image