import glob
import os
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from coreapi import FieldExtractor
from coreapi.low.object_segmentation.segmenter import Segmenter
import gradio as gr

fieldExtractor = FieldExtractor()
invoiceDetector = Segmenter()

def inference(image):
    image = invoiceDetector.predict(image)
    image_rst = fieldExtractor.predict(image)
    return image_rst

def main():
    title = "Interactive demo: LayoutLMv2"
    description = "Demo for Microsoft's LayoutLMv2, a Transformer for state-of-the-art document image understanding tasks. This particular model is fine-tuned on FUNSD, a dataset of manually annotated forms. It annotates the words into QUESTION/ANSWER/HEADER/OTHER. To use it, simply upload an image or use the example image below. Results will show up in a few seconds."
    article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2012.14740'>LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding</a> | <a href='https://github.com/microsoft/unilm'>Github Repo</a></p>"
    # examples =[['images/images_cropped/1729611894336.jpg']]
    examples = "images/demo"

    css = """.output_image, .input_image {height: 600px !important}"""

    iface = gr.Interface(fn=inference, 
                        inputs=gr.Image(type="pil"), 
                        outputs=gr.Image(type="pil", label="annotated image"),
                        title=title,
                        description=description,
                        article=article,
                        examples=examples,
                        css=css)
    iface.launch(debug=True)

if __name__ == "__main__":
    main()


    