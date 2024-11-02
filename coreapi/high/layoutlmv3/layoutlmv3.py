import numpy as np
from PIL import Image, ImageFont, ImageDraw

from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from torch import nn

from ...middle import TextReader
from .utils import preprocess_intput, unnormalize_box, visualize, iob_to_label

class TokenClassifier(object):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "tuanhazard104/layoutlmv3-finetune-mcocr",
            apply_ocr=False,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            "tuanhazard104/layoutlmv3-finetune-mcocr",
        )

    def classify(self, image, ocr_words):
        image, words, boxes = preprocess_intput(image, ocr_words)
        # encode
        encoding = self.processor(image, words, boxes=boxes, truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoding.pop('offset_mapping')
        # forward pass
        outputs = self.model(**encoding)
        decoding = self.decode(encoding, offset_mapping, outputs.logits, image.size)
        return decoding
    
    def decode(self, encoding, offset_mapping, logits, size):
        outputs = []
        width, height = size
        predictions = logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()
        input_ids = encoding.input_ids.squeeze().tolist()
        offsets = np.array(offset_mapping.squeeze().tolist())
        ### mapping box to input_id
        box2inputId = dict()
        box_keys = ['-'.join([str(x) for x in box]) for box in np.unique(np.array(token_boxes), axis=0)]
        for box_key in box_keys:
            box2inputId[box_key] = []

        for box, input_id in zip(token_boxes, input_ids):
            box_key = '-'.join([str(x) for x in box])
            box2inputId[box_key].append(input_id)

        first_idx_word = np.where(offsets[:, 0] == 0)[0]
        
        for i in range(1, len(first_idx_word)-1): # ignore start & stop
            word_idx = offsets[first_idx_word[i]:first_idx_word[i+1]]
            word_logits = logits.squeeze()[first_idx_word[i]:first_idx_word[i+1]]
            word_probs = nn.functional.softmax(word_logits, dim=-1)
            word_probs = word_probs[0].detach().numpy()
            token_box = token_boxes[first_idx_word[i]]
            ### get input_id
            box_key = '-'.join([str(x) for x in token_box])
            input_id = box2inputId[box_key]
            # input_id = input_ids[first_idx_word[i]:first_idx_word[i+1]]
            token_word = self.processor.decode(input_id, skip_special_tokens=True)
            
            predicted_label_id = word_probs.argmax()
            predicted_label = self.model.config.id2label[predicted_label_id]
            predicted_label = iob_to_label(self.model.config.id2label[predicted_label_id]).lower()
            conf_score = word_probs[predicted_label_id]
            bbox = unnormalize_box(token_box, width, height)
            out = [bbox, token_word, predicted_label, conf_score]
            if predicted_label != "other":
                outputs.append(out)
        return outputs


class FieldExtractor(object):
    def __init__(self):
        self.reader = TextReader()
        self.classifier = TokenClassifier()
    
    def predict(self, image):
        ocr_words = self.reader.readtext(image)
        outputs = self.classifier.classify(image, ocr_words)
        image_rst = visualize(image, outputs)
        return image_rst
