import os
import gc
import numpy as np
import pandas as pd 

import keras
from keras.utils import to_categorical
from keras.preprocessing import sequence

import torch
from torch import LongTensor, FloatTensor, no_grad
from torch.backends import cudnn
from torch.nn import Module, Sequential, Linear, Sigmoid, ReLU, Dropout
from torch.backends import cudnn
from torch.utils import checkpoint
from torch.optim import Adam

import torchsample
import transformers
from torchsample.modules import ModuleTrainer
from transformers import BertForSequenceClassification, BertTokenizer

train_data_df = pd.read_csv('train.csv')
train_data = train_data_df.values

targets = []
sentences = []

for i in range(0, len(train_data)):
    targets.append(train_data[i][2])
    sentences.append(train_data[i][1])
    
targets = np.array(targets)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sequences = []
bag_of_words = []

for i in range(len(sentences)):
    tokenized_sentence = tokenizer.tokenize(sentences[i])
    bag_of_words.append(tokenized_sentence)
    sequences.append(tokenizer.convert_tokens_to_ids(tokenized_sentence))

TRAIN_VAL_SPLIT = 0.8

NUM_TEXT_FEATURES = 768
MAX_SEQUENCE_LENGTH = 128

targets = FloatTensor(targets.reshape(len(targets), 1))
sequences = LongTensor(sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH))

val_targets = targets[np.int32(TRAIN_VAL_SPLIT*len(targets)):]
train_targets = targets[:np.int32(TRAIN_VAL_SPLIT*len(targets))]
val_sequences = sequences[np.int32(TRAIN_VAL_SPLIT*len(sequences)):]
train_sequences = sequences[:np.int32(TRAIN_VAL_SPLIT*len(sequences))]

model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()

NUM_TRANSFORMER_BLOCKS = 12

# Fine - Tune only 2 Transformer Blocks and the Pooling Layer.
# Fine - Tune all parts of the model (excluding the Embeddings layer) when you have enough GPU !

for param in model.parameters():
  param.requires_grad = False
  
for param in model.bert.encoder.layer[0].parameters():
  param.requires_grad = True
  
for param in model.bert.pooler.parameters():
  param.requires_grad = True
  
for param in model.bert.encoder.layer[NUM_TRANSFORMER_BLOCKS - 1].parameters():
  param.requires_grad = True

model.classifier = Sequential(Linear(in_features=NUM_TEXT_FEATURES, out_features=100),
                              ReLU(), Linear(in_features=100, out_features=20, bias=True),
                              ReLU(), Linear(in_features=20, out_features=1, bias=True), Sigmoid()).cuda()

cudnn.benchmark = True

trainer = ModuleTrainer(model)

NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# GOOD BATCH SIZE + LEARNING RATE COMINATIONS :
# 1) 32 AND 0.00001

# Try decreasing the MAX SEQ LENGTH to 64 and increasing the BATCH SIZE to 64

trainer.compile(loss='binary_cross_entropy',
                optimizer=Adam(params=model.parameters(), lr=LEARNING_RATE))

trainer.fit(train_sequences.cuda(), train_targets.cuda(),
            val_data=(val_sequences.cuda(), val_targets.cuda()),
            num_epoch=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1)
