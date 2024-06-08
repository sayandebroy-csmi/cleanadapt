import json
import pandas as pd
import torch
import random
import numpy as np

from torchvision import transforms
from torch.utils import data
from pytorchvideo.transforms import RandAugment

#from dataset.ucfhmdb_dataset import UCFHMDBDataset
from ucfhmdb_dataset import UCFHMDBDataset

#from dataset.epic_dataset import EpicKitchenDataset
#from epic_dataset import EpicKitchenDataset

#from dataset import transforms as T
import transforms as T

from torch.utils.data import DataLoader
#from utils.utils import seed_worker
#from utils import seed_worker

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data(transform, args, mode, dataset, pseudo_label_path = None):

    args.dataset = dataset
    args.load_type = mode
    
    """if pseudo_label_path is not None:
        if args.target_dataset in ["D1", "D2", "D3"]:
            pseudo_labels = pd.read_pickle(pseudo_label_path)
        else:
            pseudo_labels = json.load(open(pseudo_label_path, 'rb'))
        print("Loading pseudo-labels from {}".format(pseudo_label_path))
    else:
        pseudo_labels = None"""
    pseudo_labels = None
    if args.dataset in ['ucf101', 'hmdb51', 'ucf101s', 'hmdb51s']:
        dataset = UCFHMDBDataset(args, transform = transform, pseudo_labels = pseudo_labels)
    #else:
    #    dataset = EpicKitchenDataset(args, transform = transform, pseudo_labels = pseudo_labels)

    return dataset



def get_dataloader(args, mode, dataset):
    
    args.load_type = mode
    args.modality = "RGB"
    if mode == 'train':
        if args.modality == "Joint":
            batch_size = args.batch_size // 2
        else:
            batch_size = args.batch_size 
            """data_loader = DataLoader(
                dataset, batch_size = batch_size , shuffle = True,
                num_workers = args.num_workers, worker_init_fn=seed_worker,
                pin_memory = False, drop_last = False
            )"""
            data_loader = DataLoader(
                dataset, batch_size = batch_size , shuffle = True,
                num_workers = 15,worker_init_fn=seed_worker,
                pin_memory = False, drop_last = False
            )

    elif mode == 'val':
        if args.modality == "Joint":
            batch_size = args.batch_size // 2
        else:
            batch_size = args.batch_size 
            """data_loader = DataLoader(
            dataset, batch_size = batch_size, shuffle = False,
            worker_init_fn=seed_worker,
            num_workers = args.num_workers, pin_memory = False,
            drop_last = False
            )"""
            data_loader = DataLoader(
                dataset, batch_size = batch_size, shuffle = False,
                num_workers = 15, worker_init_fn=seed_worker,pin_memory = False,
                drop_last = False
            )

    elif mode == "generate-pseudo-label" or mode == "feature":
        
        data_loader = DataLoader(
            dataset, batch_size = 1, shuffle = False,
            num_workers = 2, pin_memory = False,
            drop_last = False
        )
    
    print("Mode: {} Size: {}".format(mode, len(dataset)))

    return data_loader

    


def get_weak_transforms(args, mode):

    if mode == 'train':
        transform = transforms.Compose([T.RandomCrop(224),
        T.RandomHorizontalFlip()
        ])
    else:
        transform = transforms.Compose([T.CenterCrop(224)
        ])

    return transform


def get_strong_transforms(args, mode):

    transform = transforms.Compose([
                                    RandAugment(
                                    magnitude = 2,
                                    num_layers = 3,
                                    prob = 0.5
                                    ),
                                    T.change_time_dimension(),
                                    T.CenterCrop(224)
                                    ]
                                )
    
    return transform