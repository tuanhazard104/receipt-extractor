# Visual Information Extraction from Vietnamese receipt

### Models

- Receipt segmentation: DeepLabV3
- Text detection: CRAFT
- Text recognition: VietOCR
- Visual information extraction: LayoutLMv3

### Pretrained model

- [CRAFT](https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file#test-instruction-using-pretrained-model)
- [DeepLabV3](https://github.com/spmallick/learnopencv/blob/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3/README.md#dataset-and-trained-model-download-links)
- [LayoutLMV3](https://huggingface.co/tuanhazard104/layoutlmv3-finetune-mcocr)


### Datasets

- [MC-OCR](https://www.rivf2021-mc-ocr.vietnlp.com/)
- [SROIE](https://rrc.cvc.uab.es/?ch=13)

### Evaluation  

TODO

### Command

- install dependencies

```bash
pip install -r requirements.txt
```

- Run the demo application

```bash
python app.py
```

### Preview

TODO


### Reference

- MC-OCR dataset: https://www.rivf2021-mc-ocr.vietnlp.com/
- DeepLabV3: https://github.com/spmallick/learnopencv/tree/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- VietOCR: https://github.com/pbcquoc/vietocr
- LayoutLMV3: https://github.com/microsoft/unilm/tree/master/layoutlmv3