import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

import torchvision.transforms.functional as F
import time
import os
import copy
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 255, 'constant')

def train_and_save():
    data_transforms = {
        'train': transforms.Compose([
            SquarePad(),
            transforms.Resize(size=(224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # Normalize the image using the mean and std of ImageNet, need to compute the mean and std of custom dataset?     
        ]),
        'val': transforms.Compose([
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    # Initalize dataset
    data_dir = 'data_orientation_104_480'
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print('Dataset size - Train', len(image_datasets['train']))
    print('Dataset size - Val', len(image_datasets['val']))
    
    # Load data by DataLoader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Train code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-'*10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc >= best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best epoch: ', best_epoch)
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    mnetv2 = models.mobilenet_v2(pretrained=True)
    for param in mnetv2.parameters():
        param.requires_grad = False

    mnetv2.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    mnetv2  = mnetv2.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.SGD(mnetv2.classifier.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if len(class_names) > 100:
        num_epochs = 25
    else:
        num_epochs = 15
    mnetv2 = train_model(mnetv2, criterion, optimizer,
                            exp_lr_scheduler, num_epochs=40)

    curr_time = datetime.now()
    model_name = f'{str(curr_time)[5:10]} {len(class_names)} classes-best with normalization.pth'
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(mnetv2.state_dict(), os.path.join(model_dir, model_name))
    with open('model_info.json', 'a+') as f:
        pass
    with open('model_info.json', 'r+') as f:
        try:
            model_info = json.load(f)
        except:
            model_info = []
        model_info.append({'time': str(datetime.now()), 'model': os.path.join(model_dir, model_name), 'classes': class_names})
        f.seek(0)
        json.dump(model_info, f, indent=4)
    print("Saving model successfully")

if __name__ == '__main__':
    train_and_save()

    
