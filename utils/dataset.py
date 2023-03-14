import os
import sys
import torch
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from .transforms import *
from monai.transforms import Compose, ToTensorD
from monai.data import CacheDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler

valid_range = {
    "acc_X" : (-19.6, 19.6),
    "acc_Y" : (-19.6, 19.6),
    "acc_Z" : (-19.6, 19.6),
    "gyr_X" : (-573, 573),
    "gyr_Y" : (-573, 573),
    "gyr_Z" : (-573, 573),
    "heartRate" : (0, 255),
    "rRInterval" : (0, 2000),
}

"""
E-Prevention Dataset 
"""
class EPreventionDataset(CacheDataset):
    def __init__(self, split_path, split, transforms, max_samples=None, subject=None, cache_num = sys.maxsize, cache_rate=1.0, num_workers=1):    
        
        self.split = split
        self.max_samples = max_samples
        self.subject = subject
        
        data = self._generate_data_list(split_path/f"{split}_dataset.csv")

        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    # generate data list
    def _generate_data_list(self, split_path):

        # open csv with observations
        data_list = pd.read_csv(split_path, index_col=0, nrows=self.max_samples)
        if self.subject is not None:
           # filter subject
            data_list = data_list[data_list['user_id']==self.subject]
        # filter valid
        data_list = data_list[data_list.valid.astype(bool)]
        # save ditribution
        count_distribution = data_list.label.value_counts().sort_index().to_numpy()
        num_samples = len(data_list)
        self.distribution = count_distribution / num_samples

        return data_list.to_dict('records')  
    
    def get_label_proportions(self):

        return self.distribution

def _get_train_transforms(args):

    if args.track == 1:

        basics_1 = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file', 'step_file'], args.root_dir)
        ]

        train_load = [LoadDataD(['data_file'], split='train', use_sleeping=args.use_sleeping)]

        basics_2 = [       
                ExtractTimeD(['data']),
                ToArrayD(['data']),
                NormalizeDataD(['data'], valid_ranges=args.valid_ranges, use_sleeping=args.use_sleeping),
                InterpolateDataD(['data']),
        ]

        if args.use_steps:
            basics_2 = [
                    *basics_2,
                    LoadStepD(['step_file'], use_calories=args.use_calories),
                    ConvertToSequenceD(['step']),
                    NormalizeStepD(['step']),
                    ConcatenateStepD(['data']),
                ]
        else:
            basics_2 = [
                    *basics_2,
                    DeleteTimeD(['time'])
                ]
            
        basics_2 = [
                    *basics_2,
                    ToTensorD(['data'], dtype=torch.float)
                ]

        if not args.drop_short_sequences:
            basics_2 = [
                *basics_2,
                PadShortSequenceD(['data'], output_size=args.window_size, padding=args.padding_mode, mode=args.padding_loc)
            ]  

        train_transforms = [
            *basics_1,
            *train_load,
            *basics_2
        ]

    elif args.track==2:

        if args.data_type == 'aggregated':

            train_transforms = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file'], args.root_dir),
                LoadAggregatedDataD(['data_file'], 'train'),
                ImputeMedianD(['data'], stats_dir=args.split_path),
                ToNumpyD(['data']),
                ToTensorD(['data'], dtype=torch.float),
                StandardizeD(['data'], stats_dir=args.split_path),
                TransposeD(['data']),
                #FlattenD(['data']),
            ]

        else:
             train_transforms = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file'], args.root_dir),
                LoadDataD(['data_file'], split='train', use_sleeping=args.use_sleeping),
                ExtractTimeD(['data']),
                DeleteTimeD(['time']),
                ImputeMedianD(['data'], stats_dir=args.split_path),
                ToNumpyD(['data']),
                ToTensorD(['data'], dtype=torch.float),
                StandardizeD(['data'], stats_dir=args.split_path),
                TransposeD(['data']),
            ]
    else:
        raise Exception("Use valid track id")

    return train_transforms

def _get_eval_transforms(args, split):

    if args.track == 1:

        basics_1 = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file', 'step_file'], args.root_dir)
        ]

        val_load = [LoadDataD(['data_file'], split=split, use_sleeping=args.use_sleeping)]

        basics_2 = [       
                ExtractTimeD(['data']),
                ToArrayD(['data']),
                NormalizeDataD(['data'], valid_ranges=args.valid_ranges, use_sleeping=args.use_sleeping),
                InterpolateDataD(['data']),
        ]

        if args.use_steps:
            basics_2 = [
                    *basics_2,
                    LoadStepD(['step_file'], use_calories=args.use_calories),
                    ConvertToSequenceD(['step']),
                    NormalizeStepD(['step']),
                    ConcatenateStepD(['data']),
                ]
        else:
            basics_2 = [
                    *basics_2,
                    DeleteTimeD(['time'])
                ]
            
        basics_2 = [
                    *basics_2,
                    ToTensorD(['data'], dtype=torch.float)
                ]
        
        eval_transforms = [
            *basics_1,
            *val_load,
            *basics_2,
            CreateVotingBatchD(['data']),
            PadShortSequenceD(['data'], output_size=args.window_size, padding=args.padding_mode, mode=args.padding_loc)
        ]

    elif args.track==2:

        if args.data_type=='aggregated':

            eval_transforms = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file'], args.root_dir),
                LoadAggregatedDataD(['data_file'], split),
                ImputeMedianD(['data'], stats_dir=args.split_path),
                ToNumpyD(['data']),
                ToTensorD(['data'], dtype=torch.float),
                StandardizeD(['data'], stats_dir=args.split_path),
                TransposeD(['data']),
                CreateVotingBatchD(['data']),
                PadShortSequenceD(['data'], output_size=args.window_size, padding=args.padding_mode, mode=args.padding_loc),
                #FlattenD(['data']),
            ]
        
        else:

            eval_transforms = [
                ToTensorD(['label'],dtype=torch.long),
                AppendRootDirD(['data_file'], args.root_dir),
                LoadDataD(['data_file'], split=split, use_sleeping=args.use_sleeping),
                ExtractTimeD(['data']),
                DeleteTimeD(['time']),
                ImputeMedianD(['data'], stats_dir=args.split_path),
                ToNumpyD(['data']),
                ToTensorD(['data'], dtype=torch.float),
                StandardizeD(['data'], stats_dir=args.split_path),
                TransposeD(['data']),
                CreateVotingBatchD(['data']),
                PadShortSequenceD(['data'], output_size=args.window_size, padding=args.padding_mode, mode=args.padding_loc),
            ]

    else:
        raise Exception("Use valid track id")
    
    return eval_transforms

def get_loader(args):
    if not os.path.isdir(args.root_dir):
        raise ValueError(f"Root directory root_dir must be a directory: Given dir name {args.root_dir}")

    train_transforms = _get_train_transforms(args)
    val_transforms = _get_eval_transforms(args, 'val')

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)
    
    try:
        if args.subject is not None:
            dataset = torch.load(args.data_dir/f"data{args.subject}.pth")
        else:
            dataset = torch.load(args.data_dir/"data.pth")
    except FileNotFoundError:
        dataset = {}
        dataset["train"] = EPreventionDataset(split_path=args.split_path, split='train', transforms=train_transforms, max_samples=args.max_samples, subject=args.subject, cache_rate=args.cache_rate)
        dataset["val"] = EPreventionDataset(split_path=args.split_path, split='val', transforms=val_transforms, max_samples=args.max_samples, subject=args.subject, cache_rate=args.cache_rate)
        dataset["test"] = EPreventionDataset(split_path=args.split_path, split='test', transforms=val_transforms, max_samples=args.max_samples, subject=args.subject, cache_rate=args.cache_rate)
        if args.subject is not None:
            torch.save(dataset, args.data_dir/f"data{args.subject}.pth")
        else:
            torch.save(dataset, args.data_dir/"data.pth")

    samplers = {}
    samplers["train"] = RandomSampler(dataset["train"])
    samplers["val"] = SequentialSampler(dataset["val"])
    samplers["test"] = SequentialSampler(dataset["test"])

    batch_sampler = {}
    batch_sampler['train'] = BatchSampler(samplers["train"], args.batch_size, drop_last=True)
    batch_sampler['val'] = BatchSampler(samplers["val"], 1, drop_last=False)
    batch_sampler['test'] = BatchSampler(samplers["test"], 1, drop_last=False)
    
    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_sampler = batch_sampler['train'], num_workers=2, pin_memory=True, persistent_workers=True)
    loaders["val"] = DataLoader(dataset["val"], batch_sampler = batch_sampler['val'], num_workers=2, pin_memory=True, persistent_workers=True)
    loaders["test"] = DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=2, pin_memory=True, persistent_workers=True)
    
    if args.track == 2:
        batch_sampler['train_distribution'] = BatchSampler(samplers["train"], args.batch_size, drop_last=False)
        loaders["train_distribution"] = DataLoader(dataset["train"], batch_sampler = batch_sampler['train_distribution'], num_workers=2, pin_memory=True, persistent_workers=True)

    loss_weights = torch.Tensor(dataset["train"].get_label_proportions())

    return loaders, samplers, loss_weights

def get_test_loader(args):
    if not os.path.isdir(args.root_dir):
        raise ValueError(f"Root directory root_dir must be a directory: Given dir name {args.root_dir}")

    
    test_transforms = _get_eval_transforms(args, 'test')
    test_transforms = Compose(test_transforms)
    
    dataset = {}
    try:
        dataset = torch.load(args.data_dir/"test_data.pth")
    except FileNotFoundError:
        dataset['test'] = EPreventionDataset(split_path=args.split_path, split='test', transforms=test_transforms, max_samples=args.max_samples, cache_rate=args.cache_rate)
        torch.save(dataset, args.data_dir/"test_data.pth")

    samplers = {}
    samplers["test"] = SequentialSampler(dataset["test"])

    batch_sampler = {}
    batch_sampler['test'] = BatchSampler(samplers["test"], 1, drop_last=False)

    loaders = {}
    loaders["test"] = DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=2, pin_memory=True, persistent_workers=True)
    
    return loaders, samplers