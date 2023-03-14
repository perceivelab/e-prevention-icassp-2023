import os
import torch
import GPUtil
import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from utils import models
from pathlib import Path
from os.path import getmtime
from utils.saver import Saver
from utils.trainer import Trainer
import sklearn.metrics
from torchmetrics import MeanMetric
from utils.dataset import get_test_loader, get_loader
from utils.dataset import valid_ranges
from statsmodels.distributions.empirical_distribution import ECDF
warnings.filterwarnings("ignore")


def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=Path, default=None)
    parser.add_argument('--split', type=str, default="test")

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

def getTrainDist(loader, model, args):
    model.eval()
    torch.set_grad_enabled(False)
    # Compute the difference between signals and their reconstruction
    diff = []

    # Disable gradient
    with torch.no_grad():
        for batch in tqdm(loader):

            # load batch data
            sig = batch['data'].to(args.device)
            # fed to the net and reconstruct
            try:
                rec, _, _ = model(sig)
            except Exception as e:
                rec = model(sig)

            # append the difference to the diff vector
            diff.append((sig-rec).detach().cpu())

    diff = torch.vstack(diff)

    # Compute the distance between signal and reconstructions
    return diff.pow(2).sum(2).sqrt().as_tensor()

def getValDist(loader, model, args):
    model.eval()
    torch.set_grad_enabled(False)

    # Compute the difference between signals and their reconstruction
    diff = []
    labels = []
    ids = []
    user_ids = []

    # disable gradient
    with torch.no_grad():

        for batch in tqdm(loader):

            batch_diff = []

            # load batch data
            sig = batch['data'].to(args.device)
            
            # correct fake batch dimension
            if sig.size(0) == 1:
                sig.squeeze_(0)
            # fed to the net and reconstruct
            try:
                rec, _, _ = model(sig)
            except Exception as e:
                rec = model(sig)

            # append the distance to to the diff vector
            batch_diff = (sig-rec).detach().cpu().pow(2).sum(2).sqrt().as_tensor()
            diff.append(batch_diff)
            if 'label' in batch:
                labels.append(batch['label'])
            if 'sample_id' in batch:
                ids.append(batch['sample_id'])
            if 'user_id' in batch:
                user_ids.append(batch['user_id'])

    # Compute the distance between signal and reconstructions
    return diff, torch.stack(labels).squeeze(1), torch.tensor(user_ids).tolist(), torch.tensor(ids).tolist()

def get_predictions(base_path, experiment, split):
    # get file names
    model_name, config = get_model_name(base_path, experiment_name=experiment)

    # load model and config
    config = torch.load(config)
    state_dict = torch.load(model_name)

    args = Args(**config)

    # Load data
    loaders, samplers, _ = get_loader(args)

    # load state dict according to model confifuration
    module = getattr(models, config['model'])
    model = getattr(module, 'Model')(config)
    model.load_state_dict(state_dict)
    module = getattr(models, config['model'])
    model = getattr(module, 'Model')(config)
    model.load_state_dict(state_dict)
    model.to(args.device)

    # Get Train distances
    train_dist = getTrainDist(loaders['train_distribution'], model, args)
    #print(train_dist.mean(), train_dist.mean(), train_dist.size())

    # Calculate CDS from trainSet distance
    #print("Calculating CDF from training set...")
    ecdf = []
    for channel in range(train_dist.size(1)):
        ecdf.append(ECDF(train_dist[:, channel]))
    #print("Finished")
    #print(ecdf[0].x, ecdf[0].y)
    
    val_dist, labels, user_ids, ids = getValDist(loaders[split], model, args)

    anomaly_scores = []
    # Calculate PDF from testSet distance
    for dist in val_dist:
        #print(dist.shape)
        #print("Calculating PDF of testSet...")
        for channel in range(dist.size(1)):
            dist[:,channel] = torch.Tensor(ecdf[channel](dist[:,channel]))

        anomaly_scores.append(dist.mean(1).median())
    anomaly_scores = torch.stack(anomaly_scores).tolist()

    if split == 'test':
        return {
            'user': user_ids,
            'day': ids,
            'status': anomaly_scores,
        }
    else:
        return {
            'user': user_ids,
            'label': labels,
            'status': anomaly_scores,
        }

def main():

    # parse arguments
    args = parse()

    base_path = Path("experiments")

    # set attributes
    exp = args.exp
    split = args.split

    target_metric = MeanMetric()

    if exp == None:
        # each line the best model for a subject
        with open("best_models.txt", "r") as f:
            exps = f.readlines()
            exps = [exp.strip() for exp in exps if not exp.startswith('#')]
    else:
        exps = [exp]

    results = []
    for exp in exps:
        result = get_predictions(base_path, exp, split)
        results.append(result)
        if split == 'val':
            # Compute metrics
            precision, recall, _ = sklearn.metrics.precision_recall_curve(result['label'], result['status'])
            fpr, tpr, _ = sklearn.metrics.roc_curve(result['label'], result['status'])
            # Compute AUROC
            auroc = sklearn.metrics.auc(fpr, tpr)
            # Compute AUPRC
            auprc = sklearn.metrics.auc(recall, precision)
            # Compute harmonic mean of AUROC and AUPRC
            score = target_metric(torch.tensor([auroc, auprc]))
            #scores.append(result['score'])
            print(f"Subject {result['user'][0]}: auroc {auroc}, auprc {auprc}, score {score}")

    res = {k: [] for k in results[0].keys() if k != 'score'}
    #print(res.keys())
    for result in results:

        for k in res.keys():
            res[k].extend(result[k])
    
    if split == 'val':
        # Compute metrics
        precision, recall, _ = sklearn.metrics.precision_recall_curve(res['label'], res['status'])
        fpr, tpr, _ = sklearn.metrics.roc_curve(res['label'], res['status'])
        # Compute AUROC
        auroc = sklearn.metrics.auc(fpr, tpr)
        # Compute AUPRC
        auprc = sklearn.metrics.auc(recall, precision)
        # Compute harmonic mean of AUROC and AUPRC
        final_score = target_metric(torch.tensor([auroc, auprc]))
        print(f"Total: auroc {auroc}, auprc {auprc}, score {final_score}")

    for k in ['user', 'day']:
        if k in res:
            res[k] = list(map(lambda x: str(x).zfill(2), res[k]))
    
    res = pd.DataFrame(res)
    res.to_csv(f"{split}_track2.csv", index=False)
        
if __name__ == '__main__':
    main()