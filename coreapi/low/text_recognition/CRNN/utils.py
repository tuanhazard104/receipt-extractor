from __future__ import print_function

import math
import cv2
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def consecutive(data, mode ='first', stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    group = [item for item in group if len(item)>0]

    if mode == 'first': result = [l[0] for l in group]
    elif mode == 'last': result = [l[-1] for l in group]
    return result

def word_segmentation(mat, separator_idx =  {'th': [1,2],'en': [3,4]}, separator_idx_list = [1,2,3,4]):
    result = []
    sep_list = []
    start_idx = 0
    sep_lang = ''
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0: mode ='first'
        else: mode ='last'
        a = consecutive( np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [ [item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]: # start lang
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]: # end lang
                if sep_lang == lang: # check if last entry if the same start lang
                    new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                    if sep_start_idx > start_idx:
                        result.append( ['', [start_idx, sep_start_idx-1] ] )
                    start_idx = sep[0]+1
                    result.append(new_sep_pair)
                sep_lang = ''# reset

    if start_idx <= len(mat)-1:
        result.append( ['', [start_idx, len(mat)-1] ] )
    return result

# code is based from https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling
        self.simplified = True  # To run simplyfiy label

class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, maxCandidate, dict_list):
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        if len(sortedBeams) >  maxCandidate: sortedBeams = sortedBeams[:maxCandidate]

        for j, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ''
            for i,l in enumerate(idx_list):
                if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):
                    text += classes[l]

            if j == 0: best_text = text
            if text in dict_list:
                #print('found text: ', text)
                best_text = text
                break
            else:
                pass
                #print('not in dict: ', text)
        return best_text

def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
        c2 = classes[childBeam.labeling[-1]] # second char
        lmFactor = 0.01 # influence of language model
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
        childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
        childBeam.lmApplied = True # only apply LM once per beam entry

def simplify_label(labeling, blankIdx = 0):
    labeling = np.array(labeling)

    # collapse blank
    idx = np.where(~((np.roll(labeling,1) == labeling) & (labeling == blankIdx)))[0]
    labeling = labeling[idx]

    # get rid of blank between different characters
    idx = np.where( ~((np.roll(labeling,1) != np.roll(labeling,-1)) & (labeling == blankIdx)) )[0]

    if len(labeling) > 0:
        last_idx = len(labeling)-1
        if last_idx not in idx: idx = np.append(idx, [last_idx])
    labeling = labeling[idx]

    return tuple(labeling)

def fast_simplify_label(labeling, c, blankIdx=0):

    # Adding BlankIDX after Non-Blank IDX
    if labeling and c == blankIdx and labeling[-1] != blankIdx:
        newLabeling = labeling + (c,)

    # Case when a nonBlankChar is added after BlankChar |len(char) - 1
    elif labeling and c != blankIdx and labeling[-1] == blankIdx:

        # If Blank between same character do nothing | As done by Simplify label
        if labeling[-2] == c:
            newLabeling = labeling + (c,)

        # if blank between different character, remove it | As done by Simplify Label
        else:
            newLabeling = labeling[:-1] + (c,)

    # if consecutive blanks : Keep the original label
    elif labeling and c == blankIdx and labeling[-1] == blankIdx:
        newLabeling = labeling

    # if empty beam & first index is blank
    elif not labeling and c == blankIdx:
        newLabeling = labeling

    # if empty beam & first index is non-blank
    elif not labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    elif labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    # Cases that might still require simplyfying
    else:
        newLabeling = labeling + (c,)
        newLabeling = simplify_label(newLabeling, blankIdx)

    return newLabeling

def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()

def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list = []):
    blankIdx = 0
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()
        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]
        # go over best beams
        for labeling in bestLabelings:
            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            prev_labeling = labeling
            if not last.entries[labeling].simplified:
                labeling = simplify_label(labeling, blankIdx)

            # labeling = simplify_label(labeling, blankIdx)
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[prev_labeling].prText
            # beam-labeling not changed, therefore also LM score unchanged from

            #curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            # char_highscore = np.argpartition(mat[t, :], -5)[-5:] # run through 5 highest probability
            char_highscore = np.where(mat[t, :] >= 0.5/maxC)[0] # run through all probable characters
            for c in char_highscore:
            #for c in range(maxC - 1):
                # add new char to current beam-labeling
                # newLabeling = labeling + (c,)
                # newLabeling = simplify_label(newLabeling, blankIdx)
                newLabeling = fast_simplify_label(labeling, c, blankIdx)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)

                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

                # apply LM
                #applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state

        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    if dict_list == []:
        bestLabeling = last.sort()[0] # get most probable labeling
        res = ''
        for i,l in enumerate(bestLabeling):
            # removing repeated characters and blank.
            if l not in ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):
                res += classes[l]
    else:
        res = last.wordsearch(classes, ignore_idx, 20, dict_list)
    return res

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

        self.separator_list = separator_list
        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        ####### latin dict
        if len(separator_list) == 0:
            dict_list = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                        word_count =  input_file.read().splitlines()
                    dict_list += word_count
                except:
                    pass
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                    word_count =  input_file.read().splitlines()
                dict_list[lang] = word_count

        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:]==t[:-1])),0,True)
            # Returns a boolean array where true is when the value is not in the ignore_idx list
            b = ~np.isin(t,np.array(self.ignore_idx))
            # Combine the two boolean array
            c = a & b
            # Gets the corresponding character according to the saved indexes
            text = ''.join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        texts = []
        for i in range(mat.shape[0]):
            t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        texts = []
        argmax = np.argmax(mat, axis = 2)

        for i in range(mat.shape[0]):
            string = ''
            # without separators - use space as separator
            if len(self.separator_list) == 0:
                space_idx = self.dict[' ']

                data = np.argwhere(argmax[i]!=space_idx).flatten()
                group = np.split(data, np.where(np.diff(data) != 1)[0]+1)
                group = [ list(item) for item in group if len(item)>0]

                for j, list_idx in enumerate(group):
                    matrix = mat[i, list_idx,:]
                    t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None,\
                                      beamWidth=beamWidth, dict_list=self.dict_list)
                    if j == 0: string += t
                    else: string += ' '+t

            # with separators
            else:
                words = word_segmentation(argmax[i])

                for word in words:
                    matrix = mat[i, word[1][0]:word[1][1]+1,:]
                    if word[0] == '': dict_list = []
                    else: dict_list = self.dict_list[word[0]]
                    t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None, beamWidth=beamWidth, dict_list=dict_list)
                    string += t
            texts.append(string)
        return texts

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class ListDataset(torch.utils.data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, 'L')

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target = self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
