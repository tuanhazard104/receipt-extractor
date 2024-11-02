"""
    (C) Copyright 2023 ARS Group (Advanced Research & Solutions)
    File identifier.py
        Classify object by triplet
"""

import os
import cv2
import torch
import torch.nn.functional as F
from .utils import img_to_tensor
from .model import MobileNetId, VGG16Id

# Configure the model path
g_dir_path = os.path.dirname(os.path.realpath(__file__))
g_model_path = os.path.join(g_dir_path, 'models/vgg16_best.pth')

MIN_MATCHES = 25

class FormIdentifier:
    """ Classify business form or invoice object
    Args:
        device (int):
        images_dir (str):
    """
    def __init__(self, gpu=True, model_path=g_model_path):
        
        if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.min_size = 320

        print("[INFO] Loading Business Form Indentifier Model", model_path)
        self.model = VGG16Id()
        self.model.to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.database = {}

    def prepare_database(self, db_data):
        """ This function passes data loaded from DB
        """
        self.database["class_names"] = []
        self.database["image_paths"] = []

        for dat in db_data:
            (id, cls_name, image_path, embedding) = dat
            self.database["class_names"].append(cls_name)
            self.database["image_paths"].append(image_path)

            embedding = torch.from_numpy(embedding).to(self.device)
            if "embeddings" not in self.database:
                self.database["embeddings"] = embedding
            else:
                self.database["embeddings"] = torch.cat([self.database["embeddings"], embedding])

    def predict_img_class(self, img, threshold=0.5):
        """ This function detects the class of the input image

        Args:
            img (ndarray): Input image
            threshold (float): To specify whether image is in database or not

        Returns:
            class_name (str)
        """
        if len(self.database["embeddings"]) == 0: # Empty database
            return False, -1, ''

        input_tensor = img_to_tensor(img, self.device, self.min_size)
        with torch.no_grad():
            out = self.model(input_tensor)
            dist = F.relu((out - self.database["embeddings"]).pow(2).sum(1)).sqrt()
            min_dist, min_index = torch.min(dist, 0)
        
        if min_dist > threshold:
            return False, min_dist, self.database["class_names"][min_index]
        return True, min_dist, self.database["class_names"][min_index]

    def predict_img_top_result(self, img, topk=None, excluded_list=None):
        """ This function detects and lists top-K (candidates) classes in database that look similar to input image
        
        Args:
            img (ndarray): Input image
            topk (int): The number of candidates returned
            excluded_list (list): The list of class that will be excluded in the output
        
        Returns:
            result (list): The top-K (candidates) classes
        """
        class_names = self.database["class_names"]
        image_paths = self.database["image_paths"]
        if len(class_names) == 0:
            return []
        if topk is None:
            topk = len(class_names)

        input_tensor = img_to_tensor(img, self.device, self.min_size)
        with torch.no_grad():
            out = self.model(input_tensor)
            dist = F.relu((out - self.database["embeddings"]).pow(2).sum(1)).sqrt()
            sorted_rst = dist.argsort()[:topk]

        dist = dist.cpu().numpy()

        result = []
        uniq_anchors = []
        for i in sorted_rst:
            name = class_names[int(i)]
            image_path = image_paths[int(i)]
            if excluded_list is not None and name in excluded_list:
                continue

            if name not in uniq_anchors:
                uniq_anchors.append(name)
                result.append([name, image_path, round(dist[int(i)], 6)])
        
        return result
    
    def add_embedding(self, img, class_name, image_path):
        """ Add a pattern to database

        Returns:
            embedding: The vector embedding of input image
        """
        input_tensor = img_to_tensor(img, self.device, self.min_size)
        with torch.no_grad():
            embedding = self.model(input_tensor)
        if "embeddings" not in self.database:
            self.database["embeddings"] = embedding
        else:
            self.database["embeddings"] = torch.cat([self.database["embeddings"], embedding])
        self.database["class_names"].append(class_name)
        self.database["image_paths"].append(image_path)
        return embedding
    
    def update_embedding(self, img, class_name):
        """ Update embedding of a class
        """
        class_names = self.database["class_names"]
        class_index = class_names.index(class_name)
        input_tensor = img_to_tensor(img, self.device, self.min_size)
        with torch.no_grad():
            embedding = self.model(input_tensor)
            self.database["embeddings"][class_index] = embedding
    
    def delete_embedding(self, image_path):
        """ Delete an anchor image in database
        """
        class_names = self.database["class_names"]
        image_paths = self.database["image_paths"]
        embeddings = list(torch.chunk(self.database["embeddings"], len(class_names)))
        
        del_item_id = image_paths.index(image_path)

        del class_names[del_item_id]
        del image_paths[del_item_id]
        del embeddings[del_item_id]

        self.database["class_names"] = class_names
        self.database["image_paths"] = image_paths
        self.database["embeddings"] = torch.cat(embeddings)

    def delete_class(self, class_name):
        """ Delete a class from database
        """
        class_names = self.database["class_names"]
        image_paths = self.database["image_paths"]
        embeddings = list(torch.chunk(self.database['embeddings'], len(class_names)))

        del_item_ids = [i for i, x in enumerate(class_names) if x == class_name]
        for offset, del_item_id in enumerate(del_item_ids):
            del class_names[del_item_id - offset]
            del image_paths[del_item_id - offset]
            del embeddings[del_item_id - offset]
        
        self.database["class_names"] = class_names
        self.database["image_paths"] = image_paths
        if len(embeddings) > 0:
            self.database["embeddings"] = torch.cat(embeddings)
        else:
            del self.database["embeddings"] 
    
    def rename_class(self, class_name_old, class_name_new):
        """ Rename a class by new name
        """
        class_names = self.database["class_names"]
        image_paths = self.database["image_paths"]

        change_name_indexes = [i for i, x in enumerate(class_names) if x == class_name_old]
        for change_name_index in change_name_indexes:
            class_names[change_name_index] = class_name_new

            image_path_split = image_paths[change_name_index].split('/')
            image_path_split[-2] = class_name_new
            image_paths[change_name_index] = '/'.join(str(e) for e in image_path_split)
        
        self.database["class_names"] = class_names
        self.database["image_paths"] = image_paths

    def compute_embedding(self, image):
        """ Compute embedding
        """
        input_tensor = img_to_tensor(image, self.device, self.min_size)
        with torch.no_grad():
            out = self.model(input_tensor)
        return out

    def compute_distance(self, vector1, vector2):
        """ Compute distance
        """
        with torch.no_grad():
            distance = F.relu((vector1 - vector2).pow(2).sum(1)).sqrt()

        return distance
