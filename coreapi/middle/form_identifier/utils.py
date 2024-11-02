import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=224),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def img_to_tensor(image, device, size=None):
    """
    Convert image to tensor
    """
    img = image.copy()

    if size is not None:
        img = cv2.resize(img, (size, size))
    else:
        img = cv2.resize(img, (224, 224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = (img - mean) / std
    inp = np.transpose(inp, (2, 0, 1))
    return torch.from_numpy(inp).float().to(device).unsqueeze(0)

def mine_triplet_hard(outputs, labels):
    """
    Calculate triplet hard
    """
    ### I challenge you to understand this
    embeddings = outputs.detach().cpu()
    num = len(outputs)
    dist = torch.sum(embeddings ** 2, dim=1).view(-1, 1) * 2 - 2 * embeddings.matmul(embeddings.T)
    dist = F.relu(dist).sqrt()
    dist = dist.numpy()
    hori = labels.expand((num, num))
    verti = labels.view(-1, 1).expand((num, num))
    mask = (hori == verti).numpy().astype(np.int) # Same label = 1, different = 0
    positive_mask = np.logical_xor(mask, np.eye(num, dtype=np.int)).astype(np.int)
    positive_dist_with_mask = positive_mask * dist
    negative_dist_with_mask = dist + (mask * np.max(dist, axis=1, keepdims=True))
    anchor = np.arange(num)
    posi_index = np.argmax(positive_dist_with_mask, axis=1)
    nega_index = np.argmin(negative_dist_with_mask, axis=1)
    batch_hard = np.vstack([anchor, posi_index, nega_index]).T
    fail_batch = []

    for i in np.where(positive_mask.sum(axis=1) == 0)[0]:
        batch_hard[i][1] = batch_hard[i][0]
        fail_batch.append(i)
    #print(positive_mask.sum(axis=1))
    for i, trio in enumerate(batch_hard):
        #if i not in fail_batch:
        #    assert trio[0] != trio[1]
        assert trio[0] != trio[2]
        assert trio[1] != trio[2]
        assert labels[trio[0]] == labels[trio[1]]
        assert labels[trio[0]] != labels[trio[2]]
    return batch_hard
