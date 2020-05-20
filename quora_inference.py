# Import necessary libraries

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import LongTensor
from transformers import RobertaModel, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences as pad

# Define hyperparameters and paths

MAXLEN = 64
DROP_RATE = 0.3

# Load trained model for inference from path

model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model)

class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.dropout = nn.Dropout(DROP_RATE)
        self.dense_output = nn.Linear(768, 1)
        self.roberta = RobertaModel.from_pretrained(model)

    def forward(self, inp, att):
        inp = inp.view(-1, MAXLEN)
        _, self.feat = self.roberta(inp, att)
        return self.dense_output(self.dropout(self.feat))
    
network = Roberta()
parser = argparse.ArgumentParser()

parser.add_argument('train_model_path')
args = parser.parse_args(); path = args.train_model_path
network.load_state_dict(torch.load(path + 'insincerity_model.pt'))

# Move model to GPU and set it to evaluation mode

network = network.cuda().eval()

# Create inference function to predict insincerity

def predict_insincerity(question):
    pg, tg = 'post', 'post'
    ins = {0: 'sincere', 1: 'insincere'}
    quest_ids = tokenizer.encode(question.strip())

    attention_mask_idx = len(quest_ids) - 1
    if 0 not in quest_ids: quest_ids = 0 + quest_ids
    quest_ids = pad([quest_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)

    att_mask = np.zeros(MAXLEN)
    att_mask[1:attention_mask_idx] = 1
    att_mask = att_mask.reshape((1, -1))
    if 2 not in quest_ids: quest_ids[-1], attention_mask[-1] = 2, 0
    quest_ids, att_mask = torch.LongTensor(quest_ids), torch.LongTensor(att_mask)
    
    output = network.forward(quest_ids.cuda(), att_mask.cuda())
    return ins[int(np.round(nn.Sigmoid()(output.detach().cpu()).item()))]

# Predict insincerity on random question

parser.add_argument('quest')
args = parser.parse_args()
print(predict_insincerity(args.quest))
