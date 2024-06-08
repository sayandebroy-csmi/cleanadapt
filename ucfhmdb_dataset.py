
import os
import random
import pickle 
import numpy as np
import cv2

import torch

#from dataset import transforms as T
import transforms as T


class VideoRecord(object):
    def __init__(self, row, root_path):
        self._data = row
        self.root_path = root_path
        # #**
        # print('self.root_path = ', self.root_path)
        # print('Length self.root_path = ', len(self.root_path))

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        # We don't use num of frames from the split file as in some videos 
        # number of frames differs from the text file
        list_of_images = [fn for fn in os.listdir(
			os.path.join(self.root_path, self._data[0])) if fn.startswith("frame")]
        return len(list_of_images) - 1

    @property
    def label(self):
        return int(self._data[2])


    @property
    def frames(self):
        list_of_images = [fn for fn in os.listdir(
			os.path.join(self.root_path, self._data[0])) if fn.startswith("frame")]
        return sorted(list_of_images)


def load_rgb_frames(image_dir, vid, video_frames, frame_idx):
    frames = []
    for idx in frame_idx:
        img = cv2.imread(os.path.join(image_dir, vid, video_frames[idx]))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, video_frames, frame_idx):
    frames = []
    for idx in frame_idx:
        imgx = cv2.imread(os.path.join(image_dir, 'u', vid, video_frames[idx]), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, 'v', vid, video_frames[idx]), cv2.IMREAD_GRAYSCALE)

        w,h = imgx.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
            
        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


class UCFHMDBDataset(object):
    def __init__(self, args, transform, pseudo_labels = None):

        #self.split_path = args.split_path
        self.split_path = 'splits'

        self.load_type = args.load_type
        #self.load_type = 'train'

        self.dataset = args.dataset

        #self.data_path = args.data_path
        self.data_path = 'ucf101-hmdb51_datasets/data'

        #self.modality = args.modality
        self.modality = 'RGB'

        #self.num_frames = args.clip_length
        self.num_frames = 16

        #self.sampling_rate = args.sampling_rate
        self.sampling_rate = 2
        
        # transform would be a list for training.
        if self.load_type == 'train1':
            self.weak_transform, self.strong_transform = transform[0], transform[1]
        elif self.load_type == 'train2':
            self.weak_transform, self.strong_transform = transform[0], transform[1]
        elif self.load_type == 'train3':
            self.weak_transform, self.strong_transform = transform[0], transform[1]
        else:
            self.weak_transform, self.strong_transform = transform, transform

        self.pseudo_labels = pseudo_labels
        self.selected_sample_dict = {}

        print("Loading {} dataset..".format(self.dataset))

        self._parseVideos()

    def _parseVideos(self):

        """
        Read all information of all videos in the particular dataset with particular modality.
        It does not actually read the frames.
        """

        self.samples = []

        if self.dataset.lower() in ['ucf101', 'ucf101s']:
            folder_name = 'ucf101'
        else:
            folder_name = 'hmdb51'
        
        self.rgb_data_path = os.path.join(self.data_path, 'rgb', folder_name)

        self.flow_data_path = os.path.join(self.data_path, 'flow', folder_name)  
        
        if self.load_type in ["train1"]:
            full_split_file_path = os.path.join(self.split_path, "split_{}_{}.txt".format(self.dataset.lower(), 'train'))
        elif self.load_type in ["val1"]:
            full_split_file_path = os.path.join(self.split_path, "split_{}_{}.txt".format(self.dataset.lower(), 'val'))
        
        elif self.load_type in ["train2"]:
            full_split_file_path = os.path.join(self.split_path, "PL_split_{}_{}.txt".format(self.dataset.lower(), 'train'))
        elif self.load_type in ["val2"]:
            full_split_file_path = os.path.join(self.split_path, "split_{}_{}.txt".format(self.dataset.lower(), 'train'))
            
        elif self.load_type in ["train3"]:
            full_split_file_path = os.path.join(self.split_path, "cleanPL_split_{}_{}.txt".format(self.dataset.lower(), 'train'))
        elif self.load_type in ["val3"]:
            full_split_file_path = os.path.join(self.split_path, "split_{}_{}.txt".format(self.dataset.lower(), 'val'))

        # Load all the videos
        self.video_list = [VideoRecord(x.strip().split(' '), self.rgb_data_path) for x in open(os.path.join(full_split_file_path))]
            
        self.indices = np.arange(0, len(self.video_list))

        for x in self.video_list:
            self.selected_sample_dict[x.path] = 1

        # self._update_video_list(select_all = True)


    def _frameSampler(self, total):

        """
        Samples the frames from a video. Takes total number of frames as an input and
        returns the selected sequence of frames.
        """
        
        if total - self.num_frames * self.sampling_rate <= 0:
            sequence = np.arange(self.num_frames) * self.sampling_rate + np.random.choice(range(self.sampling_rate), 1)
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence < total]
            seq_idx[-len(sequence)::] = sequence
        else:
            start = np.random.choice(range(total - self.num_frames * self.sampling_rate), 1)
            seq_idx = np.arange(self.num_frames) * self.sampling_rate + start

        return seq_idx
    

    def _singleSampler(self, total):
        
        seq1 = self._frameSampler(total)
        return seq1
        
    def __len__(self):
        return len(self.video_list)
    
    def _update_pseudo_labels(self, pseudo_label_dict):
        print("Updating the pseudo label dict")
        self.pseudo_labels = pseudo_label_dict.copy()

    # def _update_video_list(self, select_all = True):
    #     '''
    #     Update the list of videos based on the small-loss trick
    #     '''
    #     self.selected_video_list = []        
    #     for x in self.video_list:
    #         if not select_all:
    #             # due to last batch drop in the dataloader, 
    #             # some videos may not be present
    #             if x.path in list(self.selected_sample_dict.keys()):
    #                 if self.selected_sample_dict[x.path] == 1:
    #                     self.selected_video_list.append(x)
    #             else:
    #                 continue
    #         else:
    #             self.selected_video_list.append(x)


    def __getitem__(self, idx):
        
        video = self.video_list[idx]
        video_length, video_label, video_frames = \
            video.num_frames, video.label, video.frames

        video_name = video._data[0]

        frame_idx = self._singleSampler(video_length)
        if self.modality == 'RGB':
            rgb_seq = load_rgb_frames(self.rgb_data_path, video.path, video_frames, frame_idx)
            
            weak_rgb_seq = self.weak_transform(rgb_seq)
            strong_rgb_seq = self.strong_transform(torch.from_numpy(rgb_seq.transpose(0, 3, 1, 2)))

        elif self.modality == 'Flow':
            flow_seq = load_flow_frames(self.flow_data_path, video.path, video_frames, frame_idx)
            weak_flow_seq = self.weak_transform(flow_seq)
            strong_flow_seq = self.strong_transform(flow_seq)

        elif self.modality == 'Joint':
            rgb_seq = load_rgb_frames(self.rgb_data_path, video.path, video_frames, frame_idx)
            flow_seq = load_flow_frames(self.flow_data_path, video.path, video_frames, frame_idx)
            
            
            weak_flow_seq = self.weak_transform(flow_seq)
            flow_seq_padded = np.concatenate((flow_seq, np.zeros((flow_seq.shape[:-1] + (1, )))), axis = 3)
            strong_flow_seq = self.strong_transform(torch.from_numpy(flow_seq_padded.transpose(0, 3, 1, 2)))[:, :, :, :-1]
            
            weak_rgb_seq = self.weak_transform(rgb_seq)
            strong_rgb_seq = self.strong_transform(torch.from_numpy(rgb_seq.transpose(0, 3, 1, 2)))
            
        
        if self.modality == 'RGB':
            if self.pseudo_labels is not None:
                pseudo_label = self.pseudo_labels[video_name]
                return T.video_to_tensor(weak_rgb_seq), strong_rgb_seq.permute([3,0,1,2]), torch.from_numpy(np.array(pseudo_label)), \
                    torch.from_numpy(np.array(video_label)), video._data[0]
                
            return T.video_to_tensor(weak_rgb_seq), torch.from_numpy(np.array(video_label)), video._data[0]
        
        elif self.modality == 'Flow':
            if self.pseudo_labels is not None:
                pseudo_label = self.pseudo_labels[video_name]
                return T.video_to_tensor(weak_flow_seq), T.video_to_tensor(strong_flow_seq), torch.from_numpy(np.array(pseudo_label)), \
                    torch.from_numpy(np.array(video_label)), video._data[0]
                    
            return T.video_to_tensor(flow_seq), torch.from_numpy(np.array(video_label)), video._data[0]

        elif self.modality == 'Joint':
            
            if self.pseudo_labels is not None:
                
                # if self.selected_sample_dict[video_name] == 1:
                pseudo_label = self.pseudo_labels[video_name]
                is_selected = self.selected_sample_dict[video_name]
                # else:
                #     pseudo_label = self.updated_pseudo_labels[video_name]
                
                return [T.video_to_tensor(weak_rgb_seq), T.video_to_tensor(weak_flow_seq)], \
                    [strong_rgb_seq.permute([3,0,1,2]), strong_flow_seq.permute([3,0,1,2])], \
                    torch.from_numpy(np.array(pseudo_label)), \
                    torch.from_numpy(np.array(video_label)), video._data[0], is_selected
                    
            return [T.video_to_tensor(weak_rgb_seq), T.video_to_tensor(weak_flow_seq)], \
                torch.from_numpy(np.array(video_label)), video._data[0]

        
