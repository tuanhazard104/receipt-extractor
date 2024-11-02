import os
import string
import argparse
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image


from arshot.aideploy.text_recognition.CRNN.utils import CTCLabelConverter, AttnLabelConverter, NormalizePAD
from arshot.aideploy.text_recognition.CRNN.model import Model
from arshot.utility.utils import ObjectView
from arshot.aideploy.text_recognition.base import BaseTextRecognizer

class CRNNRecognizer(BaseTextRecognizer):
    def __init__(self, opt, cuda=True):
        if isinstance(opt, dict):
            opt = ObjectView(opt)
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

            cudnn.benchmark = True
            cudnn.deterministic = True
            opt.num_gpu = torch.cuda.device_count()

        self.opt = opt
        self.batch_size = opt.batch_size


        if cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        #TODO
        self.imgW = opt.imgW
        self.imgH = opt.imgH
        opt = self.opt
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:

            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)
        if opt.rgb:
            opt.input_channel = 3
        self.model = Model(opt)

        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        best_model_names = ['best_norm_ED.pth', 'best_accuracy.pth']
        if os.path.exists(opt.saved_model):
            if any(model_name in opt.saved_model for model_name in best_model_names):
                self.model.load_state_dict(torch.load(opt.saved_model), strict=False)
            else:
                checkpoint = torch.load(opt.saved_model)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model.eval()

    def recognize(self, images):

        #print('Batchsize %d'%len(images))
        #print(type(images[0]))
        if not isinstance(images, list):
            raise ValueError('Input must be list of image')
        pil_images = []
        batch_size = len(images)
        #start = time.time()
        for image in images:
            if type(image) == np.ndarray:
                image = Image.fromarray(image).convert('L') #.convert('L') for 1 channel
                pil_images.append(image)
        #print('CONVERT :', time.time() - start)
        #print(type(images[0]))
        """
        ratio_list = np.array([image.size[0] / float(image.size[1]) for image in pil_images])
        mean_ratio = np.mean(ratio_list)

        #Input image: PIL image
        #w, h = image.size
        #ratio = w / float(h)
        transform = ResizeNormalize((np.clip(int(self.imgH * mean_ratio), 40, None), self.imgH))
        #image = transform(image)
        #image = image.to(self.device)[None, :, :, :]
        image_tensors = []
        for image in pil_images:
            image_tensor = transform(image)
            # print(image_tensor.size())
            image_tensors.append(image_tensor)
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        """
        resized_max_w = self.imgW
        #start = time.time()
        transform = NormalizePAD((1, self.imgH, resized_max_w))
        #print('INIT TRANSFORM: ', time.time() - start)

        resized_images = []
        #start = time.time()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        for image in pil_images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            #start = time.time()
            resized_images.append(transform(resized_image))
        # end.record()
        # torch.cuda.synchronize()
        #print('PADDING: ',start.elapsed_time(end)/1000)  # milliseconds
        #print('PADDING: ', time.time() - start)


        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        #start = time.time()
        image = image_tensors.to(self.device)
        #print('TO DEVICE: ', time.time() - start)
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

        #########
        if 'CTC' in self.opt.Prediction:
            #start = time.time()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            preds = self.model(image, "text_for_pred").log_softmax(2)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            # print('FEEDFW: ',start.elapsed_time(end)/1000)  # milliseconds


            preds_size = torch.IntTensor([preds.size(1)] * batch_size) #
            _, preds_index = preds.permute(1, 0, 2).max(2)
            #_, pred_index = pred.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            #print('FORWARD: ', time.time() - start)
            #pred_index = pred_index.view(-1)
            #start = time.time()
            preds_str = self.converter.decode(preds_index.data, preds_size.data)
            #print('DECODE: ', time.time() - start)
            #print('Result ocr: {}'.format(pred_str))
        else:
            #print(text_for_pred)
            preds = self.model(image, text_for_pred, is_train=False)
            #pred_size = torch.IntTensor([pred.size(1)] * len(images)).to(self.device)
            _, preds_index = preds.max(2)
            #print('Pred index: s', pred_index, pred_index.shape)
            #print('Length for pred: ', pred_size, pred_size.shape)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        #start = time.time()
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        # print('='*20, preds_max_prob, _)
        #print('MAX: ', time.time() - start)

        '''
        for im_name, pred, pred_max_prob in zip(images, preds_str, preds_max_prob):
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]
        '''
        res_ocrs = []
        for i, (pred, pred_max_prob) in enumerate(zip(preds_str, preds_max_prob)):
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                # preds_str[i] = preds_str[i][:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            confidence_score = round(float(confidence_score.cpu().detach().numpy()), 4)
            res_ocrs.append([pred, confidence_score])
            # print(confidence_score.cpu().detach().numpy())
        return res_ocrs#preds_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='/project/UBD_OCR/src/deep-text-recognition-benchmark/demo_image/exp', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='recognize/saved_models/TPS-ResNet-BiLSTM-CTC-Seed6969/best_norm_ED.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789./-', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    recognize = CRNNRecognizer(opt, cuda=True)


    # img = plt.imread()
    img = Image.open('/project/UBD_OCR/src/deep-text-recognition-benchmark/demo_image/exp/0_DSC01194.JPG').convert('L')
    string = recognize.recognize(img)
    print(string)
