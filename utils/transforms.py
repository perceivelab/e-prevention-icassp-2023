import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from monai.config import KeysCollection
from monai.transforms import MapTransform
from torch.nn import ConstantPad1d, ReplicationPad1d

## Transforms Track 1

valid_ranges = {
    "acc_X" : (-19.6, 19.6),
    "acc_Y" : (-19.6, 19.6),
    "acc_Z" : (-19.6, 19.6),
    "gyr_X" : (-573, 573),
    "gyr_Y" : (-573, 573),
    "gyr_Z" : (-573, 573),
    "heartRate" : (0, 255),
    "rRInterval" : (0, 2000),
    "sleeping" : (0, 1)
}

class AppendRootDirD(MapTransform):

    def __init__(self, keys: KeysCollection, root_dir):
        super().__init__(keys)
        self.root_dir = root_dir
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = os.path.join(self.root_dir,d[k])
        return d

class LoadDataD(MapTransform):
    
    def __init__(self, keys: KeysCollection, split, use_sleeping):
        super().__init__(keys)
        self.split = split
        if use_sleeping:
            self.cols = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'heartRate', 'rRInterval', 'timecol', 'sleeping']
        else:
            self.cols = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'heartRate', 'rRInterval', 'timecol']


    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            if self.split == 'train':
                d['data'] = pd.read_csv(d[k],
                    skiprows=lambda x : x in range(1, d['start_data_row']+1),
                    nrows=d['end_data_row']-d['start_data_row'],
                    usecols=self.cols) 
            else:
                d['data'] = pd.read_csv(d[k], usecols=self.cols)
            if self.split == 'test':
                d['sample_id'] = d['data_file'].split("/")[-2]
            del d[k]
        if 'valid' in d.keys(): del d['valid']
        if 'start_data_row' in d.keys(): del d['start_data_row']
        if 'end_data_row' in d.keys(): del d['end_data_row']
        return d

class ExtractTimeD(MapTransform):

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d['time'] = d[k].timecol.astype('datetime64[ns]')
            d[k].drop('timecol', inplace=True, axis=1)
        return d

class DeleteTimeD(MapTransform):

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            del d[k]
        return d

class LoadStepD(MapTransform):
    
    def __init__(self, keys: KeysCollection, use_calories):
        super().__init__(keys)
        if use_calories:
            self.cols = ['start_time', 'end_time', 'totalSteps', 'distance', 'calories']
        else:
            self.cols = ['start_time', 'end_time', 'totalSteps', 'distance']


    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d['step'] = pd.read_csv(data[k],
                usecols=self.cols)
            del d[k] 
        return d

class ConvertToSequenceD(MapTransform):
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            if 'calories' in d[k].columns:
                vm, vs, c = self._create_step_sequences(d['time'], d[k])
                d['step'] = np.stack([vm, vs, c], axis=0)
            else:
                vm, vs = self._create_step_sequences(d['time'], d[k])
                d['step'] = np.stack([vm, vs], axis=0)
        if 'time' in d.keys():
            del d['time']
        return d

    def _create_step_sequences(self, time, step):
    
        # create empty velocity vectors
        velocity_m = np.zeros(len(time))
        velocity_s = np.zeros(len(time))
        if 'calories' in step.columns:
            calories = np.zeros(len(time))

        # add a column of period
        step['start_time'] = pd.to_datetime(step['start_time'])
        step['end_time'] = pd.to_datetime(step['end_time'])
        step['period'] = step['end_time']-step['start_time']

        for _, s in step.iterrows():
            # get the period index in time array
            idx = np.where((time > s.start_time) & (time < s.end_time))[0]
            if len(idx) != 0:
                # assign velocity in m/s in the period
                velocity_m[idx] = s.distance / s.period.seconds
                # assign velocity in steps/5s in the period
                velocity_s[idx] = s.totalSteps / len(idx)
                if 'calories' in step.columns:
                    # assign calories in 5s
                    calories[idx] = s.calories / len(idx)
        if 'calories' in step.columns:
            return velocity_m, velocity_s, calories
        return velocity_m, velocity_s

class ToArrayD(MapTransform):
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = d[k].to_numpy().transpose()
        return d

class NormalizeDataD(MapTransform):
    
    def __init__(self, keys: KeysCollection, valid_ranges, use_sleeping):
        super().__init__(keys)
        self.valid_ranges = valid_ranges
        vr_keys = list(valid_ranges.keys())
        if not use_sleeping:
            vr_keys.remove('sleeping')
        self.min = np.array([valid_ranges[k][0] for k in vr_keys])
        self.max = np.array([valid_ranges[k][1] for k in vr_keys])

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = ((d[k].transpose() - self.min)/(self.max - self.min)).transpose()
        return d

class NormalizeStepD(MapTransform):

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            mn = d[k].min(axis=1)
            mx = d[k].max(axis=1)
            r = mx - mn
            r[np.where(r==0)] = 1
            d[k] = ((d[k].transpose() - mn) / r).transpose()
        return d

class InterpolateDataD(MapTransform):

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = self._impute_invalid_values(d[k])
        return d


    def _impute_invalid_values(self, signals):
        
        # save input
        input_signals = copy.deepcopy(signals)

        # set a treshold for detect artifacts
        signals[np.where(signals<0)] = -1.
        signals[np.where(signals>1)] = -1.
        
        # interpolate
        imputer = KNNImputer(missing_values=-1., n_neighbors=5, weights="distance")
        signals = imputer.fit_transform(signals)

        # Preserve the dimensionality of short invalid sequences
        if signals.shape[0] == 0:
            return input_signals
            
        return signals

class ConcatenateStepD(MapTransform):

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = np.concatenate([d[k], d['step']], axis=0)
        if 'step' in d.keys():
            del d['step']
        return d

class PadShortSequenceD(MapTransform):
    
    def __init__(self, keys: KeysCollection, output_size, padding, mode):
        super().__init__(keys)
        assert padding in ['replication', 'zero'], "Select Proper Padding Mode: Allowed same and zero"
        assert mode in ['head', 'center', 'tail'], "Select Proper Mode: Allowed head, center and tail"
        self.output_size = output_size
        self.padding = padding
        self.mode = mode
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        w_in = d['data'].shape[-1]
        if w_in >= self.output_size:
            return d
        pad_size = self.output_size - w_in
        if self.mode == 'head':
            padding = (pad_size, 0)
        elif self.mode == 'tail':
            padding = (0, pad_size)
        elif self.mode == 'center' and pad_size%2==0:
            padding = pad_size//2
        elif self.mode == 'center' and pad_size%2==1:
            padding = (pad_size//2, pad_size//2+1)
        pad_fn = self._get_pad_fn(padding)
        for k in self.keys:
            d[k] = pad_fn(d[k])
        return d

    def _get_pad_fn(self, padding):
        return ConstantPad1d(padding, 0) if self.padding == 'zero' else ReplicationPad1d(padding)

class CreateVotingBatchD(MapTransform):
    
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        offsets = eval(d['offsets'])
        for k in self.keys:
            windows = [d[k][:, start:stop].unsqueeze(0) for (start, stop) in offsets]
            d[k] = torch.cat(windows, dim=0)
        if 'offsets' in d.keys():
            del d['offsets']
        return d

## Transforms Track 2

class LoadAggregatedDataD(MapTransform):
    
    def __init__(self, keys: KeysCollection, split):
        super().__init__(keys)
        self.split = split

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            if self.split == 'train':
                d['data'] = pd.read_csv(d[k],
                    skiprows=lambda x : x in range(1, d['start_data_row']+1),
                    nrows=d['end_data_row']-d['start_data_row']
                ) 
            else:
                d['data'] = pd.read_csv(d[k])
            del d[k]
            del d['data']['Unnamed: 0']
        if 'valid' in d.keys(): del d['valid']
        if 'start_data_row' in d.keys(): del d['start_data_row']
        if 'end_data_row' in d.keys(): del d['end_data_row']
        return d

class ImputeMedianD(MapTransform):
    
    def __init__(self, keys: KeysCollection, stats_dir):
        super().__init__(keys)
        with open(stats_dir/"subject_stats.json", "r") as f:
            stats = json.load(f)
        self.stats = stats

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            # impute median
            d[k] = d[k].replace([np.inf, -np.inf], np.nan)
            d[k] = d[k].fillna(d[k].median())
            # check whole nan cols
            user = str(d['user_id'])
            for col in d[k].columns:
                if d[k][col].isna().all():
                    d[k][col] = self.stats[user][col]['mean']
        return d

class ToNumpyD(MapTransform):
    
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = d[k].to_numpy()
        return d

class StandardizeD(MapTransform):
    
    def __init__(self, keys: KeysCollection, stats_dir):
        super().__init__(keys)
        with open(stats_dir/"subject_stats.json", "r") as f:
            stats = json.load(f)
        self.stats = stats

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            user = str(d['user_id'])
            means = torch.tensor([stat['mean'] for _, stat in self.stats[user].items()])
            stds = torch.tensor([stat['std'] for _, stat in self.stats[user].items()])
            means[7:] = 0.
            stds[7:] = 1.
            d[k] = (d[k] - means)/stds
        return d

class TransposeD(MapTransform):
    
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = d[k].t()
        return d

class FlattenD(MapTransform):
    
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            if len(d[k].shape) == 2:
                d[k] = d[k].flatten()
            else:
                d[k] = d[k].flatten(start_dim=1)
        return d

