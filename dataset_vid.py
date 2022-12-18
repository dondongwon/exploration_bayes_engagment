import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os 
import pandas as pd
import pdb
from collections import defaultdict
import utils
from tqdm import tqdm
import time
import decord 
from decord import VideoReader
from decord import cpu, gpu
import time
from PIL import Image
import torchvision.transforms as transforms
import math
import pickle

class SpeedDating(Dataset):
    def __init__(self, group_id ='F1_Interaction_1', context_secs = 5, get_n_frames_per_sec = 10, prev_context_secs = 20, dir_path = "../data/speeddating/SpeedDating_labeled_BBox", utils = utils, transform=None, video_feat = None ):
        
        self.fps = 30
        
        self.get_n_frames_per_sec = get_n_frames_per_sec
        self.sample_fps = self.fps//self.get_n_frames_per_sec
        df = pd.read_csv(os.path.join(os.path.realpath(dir_path), "Speed_dating_avg_eng.txt"))
        self.annotations_df = df[df['video_file'] == group_id]
        self.start_times =  self.annotations_df['start_seconds'].unique()

        self.data_path = os.path.join(os.path.realpath(dir_path), group_id ) #+ ".mp4.zip.zip"
        self.context_secs = context_secs
        self.prev_context_secs = prev_context_secs
        self.participant_ids = list(utils.sd_person_order[group_id].keys())
        self.skip_secs = 5
        self.desired_feats = ['face_name', 'face_landmarks', 'keypoints', '3d_keypoints', 'gaze', 'box']

        vid_path = os.path.join(dir_path, 'videos', group_id +".mp4")


    def __len__(self):
        return len(self.start_times) - 1 

    def __getitem__(self, idx):
        
        #skip first index 
        start_time = self.start_times[idx + 1]

        start_frame = int(start_time * self.fps)
        end_frame = int((start_time + self.context_secs) * self.fps)

        #find relevant engagement labels - find previous engagement labels as well TODO

        engage_df = self.annotations_df[self.annotations_df['start_seconds'].le(start_time)]
        engage_df = engage_df[engage_df['start_seconds'].ge(start_time - self.prev_context_secs)]

        # startidx = max(idx - (self.prev_context_secs//5) - 1,0)
        # engage_df = self.annotations_df.iloc[startidx:idx + 1]

        # engage_df = self.annotations_df[self.annotations_df['start_seconds'] == start_time]
        # pdb.set_trace() 


        dictbyperson =  defaultdict(list,{ k:[] for k in self.participant_ids})
        #collate all data 

        # input_dict = defaultdict(list,{ k:[] for k in ['keypoints', 'kp_score', 'proposal_score', 'idx', 'face_name', 'box', '3d_keypoints', 'body_pose', 'face_landmarks', 'head_pose', 'head_bbox', 'gaze', 'deep_head_pose', 'gaze_follow'] })
        for i in range(start_frame, end_frame):
            if i%(self.fps//self.get_n_frames_per_sec):



                frame_path = os.path.join(self.data_path, str(start_frame).zfill(6) + ".npy")
                frame_data = np.load(frame_path, allow_pickle = True)
                
                for person_data in frame_data:
                    person_id = person_data['face_name']
                    if person_id in dictbyperson:
                        dictbyperson[person_id].append(person_data)
            
        # for i, (k, v)  in enumerate(dictbyperson.items()):            
        #     dictbyperson[k] = {k: np.stack([dic[k] for dic in v]) for k in v[0]}

        batchlist = []
        for i, (k, v)  in enumerate(dictbyperson.items()):            
            batchlist.append({k: np.stack([dic[k] for dic in v]) for k in v[0] if k in self.desired_feats})
        

        # for k in batchlist[0]:
        #     try:
        #         batchdict = {k: np.stack([dic[k] for dic in batchlist])}
        #     except Exception:
        #         pdb.set_trace()

        try:
            batchdict = {k: np.stack([dic[k] for dic in batchlist]) for k in batchlist[0]}
        except Exception:

            pdb.set_trace()

        batchdict['eng1'] = []
        batchdict['eng2'] = []
        batchdict['avg_eng'] = []

        for participant_id in self.participant_ids:

            batchdict['eng1'].append(engage_df[engage_df['person'].str.contains(participant_id)]['molly_eng_level'].to_numpy())
            batchdict['eng2'].append(engage_df[engage_df['person'].str.contains(participant_id)]['nana_eng_level'].to_numpy())
            batchdict['avg_eng'].append(engage_df[engage_df['person'].str.contains(participant_id)]['avg_eng_level'].to_numpy())
        
        # pdb.set_trace() 

        batchdict['eng1'] = np.stack(batchdict['eng1'])
        batchdict['eng2'] = np.stack(batchdict['eng2'])
        batchdict['avg_eng'] = np.stack(batchdict['avg_eng'])

        del batchdict['face_name']

        #pad context engagement values if not exist 
        context_eng_len =  (self.prev_context_secs//5 + 1)
        if batchdict['eng1'].shape[1] < context_eng_len:
            npad = ((0, 0), ( context_eng_len - batchdict['eng1'].shape[1], 0))
            batchdict['eng1'] = np.pad(batchdict['eng1'], pad_width=npad, mode='constant', constant_values=0)
            batchdict['eng2'] = np.pad(batchdict['eng2'], pad_width=npad, mode='constant', constant_values=0)
            batchdict['avg_eng'] = np.pad(batchdict['avg_eng'], pad_width=npad, mode='constant', constant_values=0)

                #find relevant video frames 

        frame_list = list(range(start_frame,end_frame,self.sample_fps))
        
        start_time = time.time()
        batchdict['video_frames'] = self.vr.get_batch(frame_list).asnumpy()

        #initialize videos


        print("--- %s seconds ---" % (time.time() - start_time))

        # for k,v in batchdict.items():
        #     print(k)
        #     print(v.shape)

        return batchdict


class RoomReader(Dataset):
    def __init__(self, group_id ='S01', context_secs = 5, get_n_frames_per_sec = 10, prev_context_secs = None, dir_path = "../data/roomreader/room_reader_corpus_db", utils = utils, transform=None, video_feat = 'resnet' ):
        
        decord.bridge.set_bridge('torch')
        

        self.group_id = group_id


        self.fps = 60
        self.get_n_frames_per_sec = get_n_frames_per_sec
        self.sample_fps = self.fps//self.get_n_frames_per_sec

        self.speakers = utils.roomreader_dict[group_id]

        self.all_sp_feats = []
        self.all_sp_eng = []
        anot_path = os.path.join(os.path.realpath(dir_path), "annotations/engagement/AllAnno") 
        anotlist = os.listdir(anot_path)
        vid_path = os.path.join(os.path.realpath(dir_path), "video/individual_participants", "individual_participants_individual_audio", group_id) 
        self.all_sp_ids = []  
        # self.all_sp_personas = []
        # self.persona_df = utils.persona_df
        self.video_frame_paths= []
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)]) #Resize
        
        self.frame_df = None 

        video_frames = []
        for sp in tqdm(self.speakers): 
            
            if 'T0' in sp: 
                self.teacher_feats = pd.read_csv(os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp)).iloc[:,5:].to_numpy()
            else:
                print(sp)
                anot_csv_path = [s for s in anotlist if sp.lower() in s.lower()][0] #multiple engagment labels -- try 1 now
                eng_df = pd.read_csv(os.path.join(anot_path, anot_csv_path), skiprows=8).iloc[:,1:].to_numpy()

                feat_csv_path = os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp)
                desired_feats = utils.roomreader_raw_feats

                feats_df = pd.read_csv(feat_csv_path, usecols = desired_feats) #, on_bad_lines='skip'   
                
                feats_df = feats_df.iloc[::self.sample_fps, :].to_numpy()
                
                # feats_df = feats_df.iloc[:,5:].to_numpy()

                self.all_sp_feats.append(feats_df)
                self.all_sp_eng.append(eng_df)
                
                sp_id = sp.split("_")[1]
                self.all_sp_ids.append(int(sp.split("_")[1][1:]))
                # self.all_sp_personas.append(self.persona_df[self.persona_df['Participant ID'] == sp_id].iloc[:,1:].to_numpy())

                sp_vid_path = os.path.join(vid_path, sp.replace("_all.csv", "_frames_fps30.npy"))
                # vl = de.VideoLoader(videos, ctx=ctx, shape=shape, interval=interval, skip=skip, shuffle=0)

                if video_feat == 'resnet':
                    video_frames.append(np.load(sp_vid_path))
                # self.frame_df = pd.read_csv(feat_csv_path, usecols = ['frame']).iloc[::(self.sample_fps), :].to_numpy()
                # frame_list = ["img-" + str(math.ceil(fr_ind.item()/2)).zfill(7) + ".png" for fr_ind in self.frame_df]
                # sp_video_frames = [] 
                # for frame in tqdm(frame_list):
                #     frame_path = os.path.join(sp_vid_path, frame)
                #     sp_video_frames.append(self.transform(Image.open(frame_path)))
                # video_frames.append(torch.stack(sp_video_frames)
        
        self.video_frames = torch.from_numpy(np.stack(video_frames)).squeeze()

        #https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
                
        try:
            self.all_sp_feats = np.stack(self.all_sp_feats).squeeze()
            # self.all_sp_personas = np.stack(self.all_sp_personas).squeeze()
        except Exception:
            pdb.set_trace()
        
        print('\n')
        print(self.all_sp_feats.shape)
        self.all_sp_eng = np.stack(self.all_sp_eng).squeeze()
        

         
        self.context_secs = context_secs
        self.duration = min(utils.rr_seconds[group_id], self.all_sp_feats.shape[1]//get_n_frames_per_sec)



    def __len__(self):
        return self.duration - (self.context_secs + 1) 

    def __getitem__(self, idx):
        
        

        end_time = idx + self.context_secs
        start_time = idx 
        

        #end frame is time * number of frames per second 
        #recall dataframe is sampled with an interval of number of samples per second 

        
        end_frame = int(end_time * self.get_n_frames_per_sec)
        start_frame = int(start_time * self.get_n_frames_per_sec)

        #find relevant engagement labels

        batchdict = dict()
        
        # try:
        # batchdict['t_openface'] = self.teacher_feats[start_frame:end_frame, :] #teacher  features
        batchdict['s_openface'] = self.all_sp_feats[:,start_frame:end_frame,:] #student features

        batchdict['eng'] = self.all_sp_eng[:,start_time:end_time+1]
        batchdict['index'] = idx
        batchdict['person_id'] = np.array(self.all_sp_ids)
        # batchdict['personas'] = self.all_sp_personas
        
        # frame_list = list(range(start_time*(self.fps//2) + 1,end_time*(self.fps//2) + 1,self.sample_fps//2))
        # frame_list = ["img-" + str(fr_ind).zfill(7) + ".png" for fr_ind in frame_list]
        # video_frames = []
        # for sp_video_frame_path in self.video_frame_paths:
        #     sp_video_frames = []
        #     for frame in frame_list:
        #         frame_path = os.path.join(sp_video_frame_path, frame)
        #         sp_video_frames.append(self.transform(Image.open(frame_path)))
        #     video_frames.append(torch.stack(sp_video_frames))

        batchdict['video_feat'] = self.video_frames[:, start_frame:end_frame, :]
        #batchdict['video_frames'] = self.video_frames[:, int(end_time * self.get_n_frames_per_sec//2): int(start_time * self.get_n_frames_per_sec//2) ]
        


        # except Exception:
        #     pdb.set_trace()

        
       
        return batchdict


#build room reader dataset

if __name__ == '__main__':
    start = time.time()

    DS = RoomReader()
    print("Finished Loading...")
    # pdb.set_trace()
    # with open('roomreader_one_group_all_data_video.pickle', 'wb') as handle: pickle.dump(DS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('roomreader_one_group_all_data_video.pickle', 'rb') as handle:
    #     DS = pickle.load(handle)

    loader = DataLoader(DS, batch_size = 4, shuffle = False, num_workers = 0)

    #with open('roomreader_5_all_data_video.pickle', 'wb') as handle: pickle.dump(group_all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for idx, batch in enumerate(tqdm(loader)): pass

    end = time.time()
    print(end - start)
