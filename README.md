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

| Fields         | Precision | Recall | F1    | # instances |
| -------------- | --------- | ------ | ----- |-------------|
| Seller         | 0.92      | 0.97   | 0.94  | 211         |
| Address        | 0.94      | 0.98   | 0.96  | 439         |
| Date           | 0.91      | 0.87   | 0.89  | 172         |
| Total_cost     | 0.94      | 0.96   | 0.95  | 250         |
| **Overall**    | 0.93      | 0.95   | 0.94  |             |


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