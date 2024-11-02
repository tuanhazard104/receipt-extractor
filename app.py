import gradio as gr

from coreapi import FieldExtractor, Segmentor

fieldExtractor = FieldExtractor()
invoiceDetector = Segmentor()

def inference(image):
    image = invoiceDetector.predict(image)
    image_rst = fieldExtractor.predict(image)
    return image_rst

def main():
    title = "Demo: Vietnamese Receipt Extractor"
    description = "It extract to four key information from given receip image: **Seller** / **Address** / **Date** / **Total cost**. Results will show up in a few seconds."
    article = "<p style='text-align: center'><a href='https://github.com/tuanhazard104/receipt-extractor'>Github Repo</a></p>"
    examples = "demo/"

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


    