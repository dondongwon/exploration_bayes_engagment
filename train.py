import os 
import pandas as pd

import torch
import torch.nn as nn


from tqdm import tqdm
from datetime import datetime


from options import parser
from trainer import Trainer
import utils
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pdb
from joblib import Parallel, delayed
import random
import utils 
import pickle 
import time
from losses import SupConLoss, FocalLoss
import torchvision
from torchsampler import ImbalancedDatasetSampler

args = parser.parse_args()


if args.classify:
    from model_classify import *
else:   
    from model import *

########################################################################
#Group Num Specific Datasets & Dataloader
########################################################################

context_secs = int(args.context_secs) 
if args.data == 'roomreader':
    group_nums_dict = utils.rr_group_nums
    from dataset_vid import RoomReader as DS
    input_feats = 135 #len(utils.roomreader_raw_feats)
    context_frames = (context_secs) * int(args.get_n_frames_per_sec)
if args.data == 'speeddating':
    group_nums_dict = utils.sd_group_nums
    from dataset_vid import SpeedDating as DS
    context_frames = context_secs * int(args.get_n_frames_per_sec)
if args.data == 'roomreaderNoVid':
    group_nums_dict = utils.rr_group_nums
    from dataset_novid import RoomReaderNoVid as DS
    input_feats = 135 #len(utils.roomreader_raw_feats)
    context_frames = (context_secs) * int(args.get_n_frames_per_sec)
if args.data == 'speeddatingNoVid':
    group_nums_dict = utils.sd_group_nums
    from dataset_novid import SpeedDatingNoVid as DS
    context_frames = context_secs * int(args.get_n_frames_per_sec)


group_ids = group_nums_dict[int(args.group_num)]

    

#parallelize this (?)
# def init_dataset(group_id, context_secs = context_secs):
#     group_specific_dataset =  DS(group_id = group_id, context_secs = context_secs)
#     return group_specific_dataset

# group_all_dataset = Parallel(n_jobs=-2)(delayed(init_dataset)(group_id) for group_id in tqdm(group_ids))
video_feat = args.video_feat

if args.pickled_dataset: 
    start_time = time.time()
    print('start loading data')
    with open('roomreader_5_all_data_novid.pickle', 'rb') as handle:
        group_all_dataset = pickle.load(handle)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('done instantiating dataset... concatenating')

else:
    if args.data_split == 'debug':
        
        group_ids = [group_nums_dict[int(args.group_num)][0]]
        val_group_id = [group_nums_dict[int(args.group_num)][0]]
        test_group_id =  [group_nums_dict[int(args.group_num)][0]]

    group_all_dataset = []
    for group_id in tqdm(group_ids):   
        group_specific_dataset = DS(group_id = group_id, context_secs = context_secs, get_n_frames_per_sec = args.get_n_frames_per_sec, video_feat = video_feat) 
        group_all_dataset.append(group_specific_dataset)
    

if args.data_split == 'debug':
    train_set = group_all_dataset[0]
    val_set = group_all_dataset[0]
    val_group_id = val_set.group_id
    test_set = group_all_dataset[0]

    test_group_id = test_set.group_id

# with open('roomreader_5_all_data.pickle', 'wb') as handle: pickle.dump(group_all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


if args.data_split == 'bygroup':
    with open('roomreader_5_all_data_video.pickle', 'wb') as handle: pickle.dump(group_all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('roomreader_5_all_data_novid.pickle', 'wb') as handle: pickle.dump(group_all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    random.Random(int(args.seed)).shuffle(group_all_dataset)
    train_set = torch.utils.data.ConcatDataset(group_all_dataset[:-4])
    val_set =  torch.utils.data.ConcatDataset(group_all_dataset[-4:-1])
    # val_group_id = val_set.group_id
    test_set = group_all_dataset[-1]
    test_group_id = test_set.group_id
    print('done setting up splits')


if args.data_split == 'bysample':
    all_data = torch.utils.data.ConcatDataset(group_all_dataset)
    test_len = len(all_data)//10
    val_len = len(all_data)//10
    train_len = len(all_data) - (test_len + val_len)
    train_set, val_set, test_set = torch.utils.data.random_split(all_data, (train_len, val_len, test_len), generator=torch.Generator().manual_seed(int(args.seed))) 

batch_size = int(args.batch_size)
epochs = int(args.epochs)

print('here')
trainloader = DataLoader(train_set, batch_size = batch_size, sampler = ImbalancedDatasetSampler(train_set), shuffle = False, drop_last=False, num_workers = 0)

#for idx, batch in enumerate(tqdm(trainloader)): pass

#  sampler = ImbalancedDatasetSampler(train_set),
valloader = DataLoader(val_set, batch_size = batch_size, shuffle = False, drop_last=False,num_workers = 0)
testloader = DataLoader(test_set, batch_size = batch_size, shuffle = False, drop_last=False,num_workers = 0)
print('Done loading data!')


########################################################################
#logging
########################################################################
group_num = int(args.group_num)
model_name = args.model_name
print("Model Chosen: {}".format(model_name))

model_unique = "{model}".format(model = model_name + "_seed{}_lr{}_test{}".format(int(args.seed), float(args.lr), test_group_id))

weight_dir = "./model_weights/{}/{}/group_{}".format(args.save_dir, args.train_level, group_num)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

log_dir = "./logs/{}/{}/group_{}".format(args.save_dir, args.train_level, group_num)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_path = log_dir + "/{model}.log".format(model = model_unique)
json_path = log_dir + "/{model}.json".format(model = model_unique)
weight_path = weight_dir + "/{model}.pth".format(model = model_unique)
print(log_path)

########################################################################
#loss
########################################################################

print('loss')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.classify:
    #class_weights = torch.tensor(utils.rr_4class_weight)

    # pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = FocalLoss(alpha = class_weights, gamma = 1)
    label_levels = 4

    print('loss chosen classification')


    # criterion = nn.CrossEntropyLoss()
    # label_levels = 9

else: 
    criterion = nn.MSELoss() 
    label_levels = None 




########################################################################
#model selection & parallelization
########################################################################




print("model selection & parallelization")

if args.model_name == 'Video_Resnet_LSTM':
    model = Video_Resnet_LSTM(input_feats=input_feats, out_feats = 1, label_levels = label_levels)
    print('selected {}'.format(args.model_name))

if args.parallel: # and args.data_split != 'debug'
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))

model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model.to(f'cuda:{model.device_ids[0]}')


print('Done loading model!')


########################################################################
#optimizer
########################################################################

if 'Frozen' in args.model_name:     
    if 'CNNLSTM' in args.model_name:
        params = [{'params': model.module.classifier.parameters()},{'params': model.module.CNN.parameters()},{'params': model.module.LSTM.parameters()},{'params': model.module.fc1.parameters()}]
    else:
        params = [{'params': model.module.classifier.parameters()},{'params': model.module.encoder.parameters()},{'params': model.module.temporal_encoder.parameters()},{'params': model.module.spatial_encoder.parameters()}]
    optimizer = torch.optim.Adam(params, lr=float(args.lr))
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


########################################################################
#train
########################################################################


trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    log_path = log_path, 
    weight_path = weight_path,
    json_path = json_path,
    args = args
    )

losses = trainer.fit(trainloader, valloader, testloader, epochs)
