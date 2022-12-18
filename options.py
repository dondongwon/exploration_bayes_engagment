import os, sys, pdb

import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description='Parameters for Engagemet Prediction')

parser.add_argument('--data', default='speeddating', choices=("speeddating","speeddatingNoVid", "roomreader", "roomreaderNoVid"))
parser.add_argument('--data_split', default='bysample', choices=("bysample", "bygroup", "debug"))
parser.add_argument('--group_num', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--seed', default = 0)
parser.add_argument('--epochs', default = 50)
parser.add_argument('--context_secs', default = 8)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--lr',  default = 0.0005)
parser.add_argument('--save_dir',  default = "main")
parser.add_argument('--train_level', choices=("individual", "group", 'individual+group'), default = 'group')
parser.add_argument('--get_n_frames_per_sec',  default = 8)
parser.add_argument('--batch_size',  default = 32)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--classify', action='store_true')
parser.add_argument('--pickled_dataset', action='store_true')
parser.add_argument('--personas', action='store_true')
