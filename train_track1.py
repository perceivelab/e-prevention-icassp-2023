import torch
import wandb 
import GPUtil
import argparse
from pathlib import Path
from utils import models
from utils.saver import Saver
from utils.trainer import Trainer
from utils.dataset import get_loader
from utils.transforms import valid_ranges
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    #parser = Parser()
    parser.add_argument('--track', type=int, default=1)
    parser.add_argument('--root_dir', type=Path, help='dataset folder path', default=Path("../datasets/SPGC_challenge_track_1_release/"))
    parser.add_argument('--fold', type=int, help='test fold for cross-validation', default=None)
    parser.add_argument('--split_path', type=Path, help='json dataset metadata', default=Path(f"data/track1/"))
    parser.add_argument('--data_dir', type=Path, default=Path(f"data/track1/"))
    parser.add_argument('--cache_rate', type=float, help='fraction of dataset to be cached in RAM', default=1.0)
    parser.add_argument('--project', type=str, default="your-wandb-project")
    parser.add_argument('--subject', type=int, default=None)

    parser.add_argument('--use_sleeping', type=bool, default=True)
    parser.add_argument('--valid_ranges', type=dict, default=valid_ranges)
    parser.add_argument('--use_steps', type=bool, default=True)
    parser.add_argument('--use_calories', type=bool, default=True)
    parser.add_argument('--window_size', type=int, default=int(3*60*12))# 3h, 60 minutes per hour, 12 samples per minute (1 sample = 5s)
    parser.add_argument('--padding_mode', type=str, default='replication')
    parser.add_argument('--padding_loc', type=str, default='center')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--drop_short_sequences', type=int, default=True)

    parser.add_argument('--model', type=str, help='model', default='transformer')

    parser.add_argument('--in_channels', type=int, default=12)
    parser.add_argument('--input_features', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--d_hid', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=46)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--dropout', type=int, default=0.1)
   
    parser.add_argument('--optimizer', type=str, help='optimizer (SGD, Adam, AdamW, RMSprop, LBFGS)', choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'], default='Adam')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization weight', default=5e-4)
    parser.add_argument('--enable_scheduler', type=int, help='enable learning rate scheduler', choices=[0,1], default=1)
    parser.add_argument('--scheduler_factor', type=float, help='if using scheduler, factor of increment/redution', default=5e-1)
    parser.add_argument('--scheduler_threshold', type=float, help='if using scheduler, threshold for learning rate update', default=1e-2)
    parser.add_argument('--scheduler_patience', type=int, help='if using scheduler, number of epochs before updating the learning rate', default=10)

    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=300)
    parser.add_argument('--experiment', type=str, help='experiment name (in None, default is timestamp_modelname)', default=None)
    parser.add_argument('--ckpt_every', type=int, help='checkpoint saving frequenct (in epochs); -1 saves only best-validation and best-test checkpoints', default=-1)
    parser.add_argument('--resume', help='if not None, checkpoint path to resume', default=None)
    # experiments should be saved in "experiments/*"

    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')

    args = parser.parse_args()

    # Generate experiment tags if not defined
    if args.experiment == None:
        if args.model in ['cnn1d', 'transformer_ablation']: # transformer ablation is a cnn + positional embedding
            args.experiment = f"{args.model}_wsize{args.window_size}_len{args.seq_len}_lr{args.learning_rate}"
        else: 
            args.experiment = f"{args.model}_d{args.d_model}_l{args.nlayers}_h{args.nhead}_ffh{args.d_hid}_wsize{args.window_size}_len{args.seq_len}_lr{args.learning_rate}"
    
    if args.fold != None:
        args.split_path = Path(f"data/track1/fold{args.fold}")
        args.data_dir = Path(f"data/track1/fold{args.fold}")
        
    return args

def main():

    # parse arguments
    args = parse()

    # select device
    if args.device == 'cuda': # choose the most free gpu
        mem = [gpu.memoryUtil for gpu in GPUtil.getGPUs()]
        args.device = 'cuda:' + str(mem.index(min(mem)))
        args.device = torch.device(args.device)
        device = args.device
        print('Using device', args.device)
    else:
        args.device = torch.device(args.device)
        device = args.device

    # Dataset e Loader
    loaders, samplers, loss_weights = get_loader(args)
    if 'test' in loaders:
        del loaders['test']

    # Model
    module = getattr(models, args.model)
    model = getattr(module, 'Model')(vars(args))
    if args.resume is not None:
        model.load_state_dict(Saver.load_model(args.resume))
    model.to(device)

    # Enable model distribuited if it is
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')

    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer chosen not implemented!")

    # Scheduler
    if args.enable_scheduler:
        scheduler = ReduceLROnPlateau(optimizer,
                        mode='min',
                        factor=args.scheduler_factor,
                        patience=args.scheduler_patience,
                        threshold=args.scheduler_threshold,
                        threshold_mode='rel',
                        cooldown=0,
                        min_lr=0,
                        eps=1e-08,
                        verbose=True)
    else:
        scheduler = None

    # Trainer
    trainer = Trainer(
        net=model,
        optim=optimizer,
        class_weights=loss_weights.to(args.device),
    )

    # Saver
    saver = Saver(base_output_dir=Path('experiments/'), tag=args.experiment)
    
    # train
    entity = "<wandb-entity>"

    run = wandb.init(project=args.project, entity=entity, config=args)
    run.name = args.experiment

    args.loaders = loaders
    args.samplers = samplers
    args.scheduler = scheduler
    args.saver = saver

    trainer.train(args)

    run.finish()    

if __name__ == '__main__':
    main()
