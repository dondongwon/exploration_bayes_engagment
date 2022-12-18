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
# from decord import VideoReader
# from decord import cpu, gpu


class SpeedDatingNoVid(Dataset):
    def __init__(self, group_id ='F1_Interaction_1', context_secs = 5, get_n_frames_per_sec = 10, prev_context_secs = 20, dir_path = "../data/speeddating/SpeedDating_labeled_BBox", utils = utils, transform=None):
        
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


        # for k,v in batchdict.items():
        #     print(k)
        #     print(v.shape)

        return batchdict


class RoomReaderNoVid(Dataset):
    def __init__(self, group_id ='S03', context_secs = 5, get_n_frames_per_sec = 10, prev_context_secs = None, dir_path = "../data/roomreader/room_reader_corpus_db", utils = utils, transform=None):

        self.fps = 60

        self.group_id = group_id
        self.get_n_frames_per_sec = get_n_frames_per_sec

        self.speakers = utils.roomreader_dict[group_id]

        self.all_sp_feats = []
        self.all_sp_eng = []
        anot_path = os.path.join(os.path.realpath(dir_path), "annotations/engagement/AllAnno") 
        anotlist = os.listdir(anot_path)
        vid_path = os.path.join(os.path.realpath(dir_path), "video/individual_participants", "individual_participants_individual_audio", group_id) 
        self.all_sp_ids = []  
        self.all_sp_personas = []
        self.persona_df = utils.persona_df
        self.video_reader_list = []

        self.sample_fps = self.fps//self.get_n_frames_per_sec

        for sp in tqdm(self.speakers): 
            
            if 'T0' in sp: 
                self.teacher_feats = pd.read_csv(os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp)).iloc[:,5:].to_numpy()
            else:
                print(sp)
                anot_csv_path = [s for s in anotlist if sp.lower() in s.lower()][0] #multiple engagment labels -- try 1 now
                eng_df = pd.read_csv(os.path.join(anot_path, anot_csv_path), skiprows=8).iloc[:,1:].to_numpy()

                print(sp)
                feat_csv_path = os.path.join(os.path.realpath(dir_path), "features/OpenFace_Features", sp)
                desired_feats = utils.roomreader_raw_feats

                feats_df = pd.read_csv(feat_csv_path, usecols = desired_feats) #, on_bad_lines='skip'
                feats_df = feats_df.iloc[::self.sample_fps, :].to_numpy()
                
                # feats_df = feats_df.iloc[:,5:].to_numpy()

                self.all_sp_feats.append(feats_df)
                self.all_sp_eng.append(eng_df)
                
                sp_id = sp.split("_")[1]
                self.all_sp_ids.append(int(sp.split("_")[1][1:]))
                self.all_sp_personas.append(self.persona_df[self.persona_df['Participant ID'] == sp_id].iloc[:,1:].to_numpy())
                

        #https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
                
        try:
            self.all_sp_feats = np.stack(self.all_sp_feats).squeeze()
            self.all_sp_personas = np.stack(self.all_sp_personas).squeeze()
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
        
        #set time
        end_time = idx + self.context_secs
        start_time = idx 
        #roomreader is annotated continuously, so we set aligned labels as end time and find previous context times with start_time
    
        end_frame = int(end_time * self.get_n_frames_per_sec)
        start_frame = int(start_time * self.get_n_frames_per_sec)

        #find relevant engagement labels

        batchdict = dict()

        
        try:
            batchdict['t_openface'] = self.teacher_feats[start_frame:end_frame, :] #teacher  features
            batchdict['s_openface'] = self.all_sp_feats[:,start_frame:end_frame,:] #student features

            batchdict['eng'] = self.all_sp_eng[:,start_time:end_time+1]
            batchdict['index'] = idx
            batchdict['person_id'] = np.array(self.all_sp_ids)
            batchdict['personas'] = self.all_sp_personas
        

        except Exception:
            pdb.set_trace()
       
        return batchdict


#build room reader dataset

if __name__ == '__main__':
    start = time.time()

    DS = SpeedDating()
    print("Finished Loading...")

    loader = DataLoader(DS, batch_size = 4, shuffle = False)

    for idx, batch in enumerate(tqdm(loader)):
        pass

    end = time.time()
    print(end - start)
