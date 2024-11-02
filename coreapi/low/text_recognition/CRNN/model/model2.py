"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from arshot.coreapi.text_recognition_21.CRNN.modules.transformation import TPS_SpatialTransformerNetwork
from arshot.coreapi.text_recognition_21.CRNN.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from arshot.coreapi.text_recognition_21.CRNN.modules.sequence_modeling import BidirectionalLSTM
from arshot.coreapi.text_recognition_21.CRNN.modules.prediction import Attention

import cv2
class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class, num_fiducial=20, imgH=32, imgW=100):
        super(Model, self).__init__()
        print(input_channel, output_channel, hidden_size, num_class, num_fiducial, imgH, imgW)

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text):
        """ Transformation stage """
        # y = input[0].div_(2).add_(0.5).cpu()
        # y = y.permute(1, 2, 0).numpy() * 255
        # cv2.imwrite('ezOCRbeforeTPS.jpg', y)
        # input = self.Transformation(input)
        # y = input[0].div_(2).add_(0.5).cpu()
        # y = y.permute(1, 2, 0).numpy() * 255
        # cv2.imwrite('ezOCRafterTPS.jpg', y)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
