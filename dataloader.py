import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
from transformers import BertModel, BertTokenizer
# model = BertModel.from_pretrained("bert-base-uncased")
# model.cuda()
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]), \
               len(self.videoLabels[vid]), \
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid, torch.FloatTensor(self.videoSentence[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        # dat = pd.DataFrame(data)
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

        textf = pad_sequence([d[0] for d in data], True)    # 100
        visuf = pad_sequence([d[1] for d in data], True)    # 342
        acouf = pad_sequence([d[2] for d in data], True)    # 1582
        text_feature = pad_sequence([d[8] for d in data], True) # 1024
        qmask = pad_sequence([d[3] for d in data])
        length_mm = torch.LongTensor([d[4] for d in data])
        umask = pad_sequence([d[5] for d in data], True)
        labels = pad_sequence([d[6] for d in data], batch_first=True, padding_value=-1)
        return textf, visuf, acouf, qmask, umask, labels, length_mm, text_feature


class MELDDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]), \
               len(self.videoLabels[vid]), \
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid, torch.FloatTensor(self.videoSentence[vid])

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        # dat = pd.DataFrame(data)
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
        sentence = []
        sentence_all = None
        textf = pad_sequence([d[0] for d in data], True)    # 600
        visuf = pad_sequence([d[1] for d in data], True)    # 342
        acouf = pad_sequence([d[2] for d in data], True)    # 300
        qmask = pad_sequence([d[3] for d in data])
        length_mm = torch.LongTensor([d[4] for d in data])
        umask = pad_sequence([d[5] for d in data], True)
        labels = pad_sequence([d[6] for d in data], batch_first=True, padding_value=-1)
        text_feature = pad_sequence([d[8] for d in data], True) # 1024
        return textf, visuf, acouf, qmask, umask, labels, length_mm, text_feature



