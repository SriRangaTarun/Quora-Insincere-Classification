# Import necessary libraries

import os
import gc
import numpy as np
import pandas as pd 

import keras
from keras.utils import to_categorical
from keras.preprocessing import sequence

import torch
from torch.optim import Adam
from torch.backends import cudnn
from torch.utils import checkpoint
from torch import LongTensor, FloatTensor, no_grad
from torch.nn import Module, Sequential, Linear, Sigmoid, ReLU, Dropout
    
import torchsample
import transformers
from torchsample.modules import ModuleTrainer
from transformers import BertForSequenceClassification, BertTokenizer

# Define hyperparameters and paths

parser = argparse.ArgumentParser()
parser.add_argument('train_data_path')
args = parser.parse_args(); path = args.train_data_path

NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

TRAIN_VAL_SPLIT = 0.8
NUM_TEXT_FEATURES = 768
MAX_SEQUENCE_LENGTH = 128

# Load training data

train_data_df = pd.read_csv(path)
train_data = train_data_df.values

targets = []
sentences = []

for i in range(0, len(train_data)):
    targets.append(train_data[i][2])
    sentences.append(train_data[i][1])

targets = np.array(targets)

# Define tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get sequences and split train/val

sequences = []
for i in range(len(sentences)):
    tokenized_sentence = tokenizer.tokenize(sentences[i])
    sequences.append(tokenizer.convert_tokens_to_ids(tokenized_sentence))

targets = FloatTensor(targets.reshape(len(targets), 1))
sequences = LongTensor(sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH))

val_targets = targets[np.int32(TRAIN_VAL_SPLIT*len(targets)):]
train_targets = targets[:np.int32(TRAIN_VAL_SPLIT*len(targets))]
val_sequences = sequences[np.int32(TRAIN_VAL_SPLIT*len(sequences)):]
train_sequences = sequences[:np.int32(TRAIN_VAL_SPLIT*len(sequences))]

# Get pretrained BERT model

NUM_TRANSFORMER_BLOCKS = 12
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()

# Fine - Tune only 2 Transformer Blocks and the Pooling Layer due to compute restrictions.

for param in model.parameters():
  param.requires_grad = False

for param in model.bert.encoder.layer[0].parameters():
  param.requires_grad = True

for param in model.bert.pooler.parameters():
  param.requires_grad = True

for param in model.bert.encoder.layer[NUM_TRANSFORMER_BLOCKS - 1].parameters():
  param.requires_grad = True

# Add custom double-dense head to the BERT model

model.classifier = Sequential(Linear(in_features=NUM_TEXT_FEATURES, out_features=100),
                              ReLU(), Linear(in_features=100, out_features=20, bias=True),
                              ReLU(), Linear(in_features=20, out_features=1, bias=True), Sigmoid()).cuda()

# Set CuDNN benchmark to speed up training and create Torchsample trainer

cudnn.benchmark = True
trainer = ModuleTrainer(model)

# Compile and train Torchsample trainer

trainer.compile(loss='binary_cross_entropy',
                optimizer=Adam(params=model.parameters(), lr=LEARNING_RATE))

trainer.fit(train_sequences.cuda(), train_targets.cuda(),
            val_data=(val_sequences.cuda(), val_targets.cuda()),
            num_epoch=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1)
