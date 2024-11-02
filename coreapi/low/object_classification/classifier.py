
""" File classifier.py
    Class ObjectClassifier includes methods to inference the object classification model
"""
from ... import core
from ... import const

class ObjectClassifier:
    """ Provide the model selection and methods of object classification

    Args:
        config (dict): configuration of the selected model
    """
    def __init__(self, config):
        if isinstance(config, dict):
            config = core.ObjectView(config)

        if config.algo_name == const.CLASSIFIER_MULTIVIEW:
            from .multiview.multiview_classifier import MultiviewClassifier as Classifier
        elif config.algo_name == const.CLASSIFIER_MOBILENETV2:
            from .mobilenet.classifier import Classifier
        ### Reserve for other algorithms
        
        self.classifier = Classifier(core.ObjectView(config.algo_config))
    
    def classify(self, image):
        """ classify object in the inputed image

        Args:
            image (array): the input single image
                In case of multi-view classification, it is list of images view

        Returns:
            dict: the dictionary of classification result (class-name and confidence score)
        """
        result = self.classifier.classify(image)
        return result
    
    def classifyBatch(self, images):
        """ classify object in the multiple images (inference with batch of images)

        Args:
            images (array): the batch of input images

        Returns:
            list: List of classification result of input images.
            The result of each input image is the dictionary of classification result (class-name and confidence score)
        """
        pass