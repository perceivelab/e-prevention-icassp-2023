import os
import torch
import GPUtil
import warnings
import argparse
import pandas as pd
from utils import models
from pathlib import Path
from os.path import getmtime
from utils.saver import Saver
from utils.trainer import Trainer
from utils.dataset import valid_ranges
from sklearn.metrics import accuracy_score
from utils.dataset import get_test_loader, get_loader
warnings.filterwarnings("ignore")

def parse():
    '''Returns args passed to the test.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=Path, default=None)
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--scheme', type=str, default='sum')

    args = parser.parse_args()
    return args

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_model_name(base_path, experiment_name):
    model_dir = base_path/Path(experiment_name)
    for model in os.listdir(model_dir):
        ckpt = model_dir/Path("ckpt")
        file_list = sorted(ckpt.glob('*.pth'), key=getmtime)
        if len(file_list) == 0:
            raise Exception("No model found")
        elif len(file_list) == 1:
            return file_list[0], None
        else:
            config = sorted(ckpt.glob('config.pth'), key=getmtime)[0]
            file_list.remove(config)
            return file_list[0], config

def get_predictions(base_path, experiment, split):
    # get file names
    model_name, config = get_model_name(base_path, experiment_name=experiment)

    # load model and config
    config = torch.load(config)
    state_dict = torch.load(model_name)

    args = Args(**config)

    # load state dict according to model confifuration
    module = getattr(models, config['model'])
    model = getattr(module, 'Model')(config)
    model.load_state_dict(state_dict)

    # select device
    if args.device == 'cuda': # choose the most free gpu
        #mem = [(torch.cuda.memory_allocated(i)+torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
        mem = [gpu.memoryUtil for gpu in GPUtil.getGPUs()]
        args.device = 'cuda:' + str(mem.index(min(mem)))
        args.device = torch.device(args.device)
        device = args.device
        print('Using device', args.device)

    if split == 'test':

        # Load data
        loaders, samplers = get_test_loader(args)
        model.to(args.device)

        # Define a trainer
        trainer = Trainer(
            net=model,
            optim=None,
            class_weights=None
        )

        # Evaluate
        args.loaders = loaders
        args.samplers = samplers

        return trainer.eval(args, split=split)
    
    else:

        if 'subject' not in config:
            args.subject = None

        # Load data
        loaders, samplers, _ = get_loader(args)
        model.to(args.device)

        # Define a trainer
        trainer = Trainer(
            net=model,
            optim=None,
            class_weights=None
        )

        # Evaluate
        args.loaders = loaders
        args.samplers = samplers
        
        print(f"Loading model: {experiment}")
        return trainer.eval(args, split=split)

def ensemble(logits, ensemble_scheme):

    #for l in logits:
    #    print(l.size())
    
    logits = torch.stack(logits)
    #print(logits.shape)
    sum_prediction = logits.sum(dim=0)
    max_prediction = logits.max(dim=0)[0]
    min_prediction = logits.min(dim=0)[0]
    
    if ensemble_scheme == 'sum':
        ensemble_prediction = sum_prediction
    elif ensemble_scheme == 'max':
        ensemble_prediction = max_prediction
    elif ensemble_scheme == 'min':
        ensemble_prediction = min_prediction
    elif ensemble_scheme == 'all':
        return torch.max(sum_prediction, 1)[1], torch.max(max_prediction, 1)[1], torch.max(min_prediction, 1)[1]
    
    #print(ensemble_prediction.shape)
    _, predicted_labels = torch.max(ensemble_prediction, 1)
    #print(ensemble_prediction, predicted_labels, predicted_labels.shape)
    
    return predicted_labels

def main():

    # parse arguments
    args = parse()

    base_path = Path("experiments")
    
    # set attributes
    split = args.split
    use_ensemble = args.exp == None
    scheme = args.scheme

    if use_ensemble:
        # open exps
        with open("ensemble.txt", "r") as f:
            ensemble_exps = f.readlines()
            ensemble_exps = [exp.strip() for exp in ensemble_exps]

        preds = []
        for exp in ensemble_exps:
            preds.append(get_predictions(base_path, exp, split))
        
        logits = [torch.tensor(prediction['logits']) for prediction in preds]
        ensemble_predictions = ensemble(logits, scheme)

        if split == 'val':
            for exp, pred in zip(ensemble_exps, preds):
                print(f"model {exp}: {pred['accuracy']}")#, preds.keys())
            accuracy = accuracy_score(preds[0]['label'], ensemble_predictions)
            print(f"---------------------------------------")
            print(f"{scheme} ensemble: {accuracy}")#, preds[0].keys())
        else:
            day = preds[0]['day']
            preds = {
                'day': day,
                'user': ensemble_predictions
            }

    else:
        
        preds = get_predictions(base_path, args.exp, split)
        if 'accuracy' in preds:
            print(f"{preds['accuracy']}", preds.keys())

    # print results
    for k in ['user', 'day']:
        if k in preds:
            preds[k] = list(map(lambda x: str(int(x)).zfill(2), preds[k]))

    if split == 'test':
        preds = pd.DataFrame(preds)
        preds.to_csv(f"{split}_track1.csv", index=False)
       
if __name__ == '__main__':
    main()