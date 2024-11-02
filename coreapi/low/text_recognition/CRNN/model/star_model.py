import torch.nn as nn
from .modules import ResNet_FeatureExtractor, BidirectionalLSTM, TPS_SpatialTransformerNetwork

class Model(nn.Module):

    def __init__(self, num_class, **kwargs):
        super(Model, self).__init__()
        self.network_params = kwargs
        input_channel, output_channel, hidden_size = kwargs['input_channel'], kwargs['output_channel'], kwargs['hidden_size']
        num_fiducial, imgH, imgW = kwargs['num_fiducial'], kwargs['imgH'], kwargs['imgW']

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text):
        if self.network_params.get('Transformation') == 'TPS' and self.network_params.get('use_transform'):
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
