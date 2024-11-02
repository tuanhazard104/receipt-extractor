from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
import importlib
from .utils import CTCLabelConverter, ListDataset, AlignCollate
from .utils import custom_mean
import math
import os

def recognizer_predict(model, converter, test_loader, batch_max_length,\
                       ignore_idx, char_group_idx, decoder = 'greedy', beamWidth= 5, device = 'cpu'):
    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)

            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            ######## filter ignore_char, rebalance
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            preds_prob[:,:,ignore_idx] = 0.
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            if decoder == 'greedy':
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
            elif decoder == 'beamsearch':
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_beamsearch(k, beamWidth=beamWidth)
            elif decoder == 'wordbeamsearch':
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_wordbeamsearch(k, beamWidth=beamWidth)

            preds_prob = preds_prob.cpu().detach().numpy()
            values = preds_prob.max(axis=2)
            indices = preds_prob.argmax(axis=2)
            preds_max_prob = []
            for v,i in zip(values, indices):
                max_probs = v[i!=0]
                if len(max_probs)>0:
                    preds_max_prob.append(max_probs)
                else:
                    preds_max_prob.append(np.array([0]))

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = custom_mean(pred_max_prob)
                result.append([pred, confidence_score])

    return result

def get_recognizer(recog_network, network_params, character,\
                   separator_list, dict_list, model_path,\
                   device = 'cpu', quantize = True):

    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)

    dirname = os.path.dirname(os.path.relpath(__file__)).replace(os.path.sep, ".")
    # dirname = "coreapi.low.text_recognition.CRNN"
    if recog_network == 'generation1':
        model_pkg = importlib.import_module(".".join([dirname, "model.model"]))
    elif recog_network == 'generation2':
        model_pkg = importlib.import_module(".".join([dirname, "model.vgg_model"]))
    elif recog_network == 'starnet':
        model_pkg = importlib.import_module(".".join([dirname, "model.star_model"]))
    else:
        model_pkg = importlib.import_module(recog_network)
    model = model_pkg.Model(num_class=num_class, **network_params)

    if device == 'cpu':
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model, converter

def get_text(character, imgH, imgW, recognizer, converter, image_list,\
             ignore_char = '',decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, workers = 1, device = 'cpu'):
    batch_max_length = int(imgW/10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True)

    # predict first round
    result1 = recognizer_predict(recognizer, converter, test_loader,batch_max_length,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    # predict second round
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False,
                        num_workers=int(workers), collate_fn=AlignCollate_contrast, pin_memory=True)
        result2 = recognizer_predict(recognizer, converter, test_loader, batch_max_length,\
                                     ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                result.append( (box, pred1[0], pred1[1]) )
            else:
                result.append( (box, pred2[0], pred2[1]) )
        else:
            result.append( (box, pred1[0], pred1[1]) )
    return result