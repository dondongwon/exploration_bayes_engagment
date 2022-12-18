
import torch
import torch.nn as nn
import pdb
import utils 
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from collections import OrderedDict
class Video_Resnet_LSTM(nn.Module):
  #https://www.frontiersin.org/articles/10.3389/frobt.2020.00116/full
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step


      self.LSTM = nn.LSTM(2048,  2048, num_layers = 1, batch_first = True)
      # self.dropout = nn.Dropout(0.5)
      self.fc1 = nn.Linear(2048, label_levels)

    def forward(self, openpose, video_feats):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]

      x = video_feats

      output, (hidden,cell) = self.LSTM(x)
      output = self.fc1(hidden).squeeze()
      

      return output
