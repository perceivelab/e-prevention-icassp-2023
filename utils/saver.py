import torch
import wandb
from time import time
from pathlib import Path
from typing import Union
from datetime import datetime
from os.path import getmtime

class Saver(object):
    """
    Saver allows for saving and restore networks.
    """
    def __init__(self, base_output_dir : Path, tag=''):

        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
        if isinstance(tag, str) and len(tag) > 0:
            # Append tag
            timestamp_str += f"_{tag}"
        self.path = base_output_dir/f'{timestamp_str}'
        self.path.mkdir(parents=True, exist_ok=True)

        # Create checkpoint sub-directory
        self.ckpt_path = self.path/'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Create a buffer for metrics
        self.buffer = {}

    @staticmethod
    def init_wandb():
        return

    def watch_model(self, model):
        wandb.watch(model, log='all', log_freq=500)

    def save_configuration(self, config: dict):
        drop_keys = ['loaders', 'samplers', 'scheduler', 'saver']
        for key in drop_keys:
            if key in config:
                del config[key]
        torch.save(config, self.ckpt_path/f"config.pth")

    @staticmethod
    def load_configuration(model_path: Union[str, Path]):
        return torch.load(model_path/f"config.pth")

    def log_configuration(self):
        return

    def save_model(self, net: torch.nn.Module, name: str, epoch: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        # Save
        torch.save(state_dict, self.ckpt_path/f"{name}_{epoch:05d}.pth")

    def save_checkpoint(self, net: torch.nn.Module, optim: torch.optim.Optimizer, config: dict, stats: dict, name: str, epoch: int):
        """
        Save model parameters and stats in the checkpoint directory.
        """
        # Get state dict
        net_state_dict = net.state_dict()
        optim_state_dict = optim.state_dict()

        # Copy to CPU
        for k, v in net_state_dict.items():
            net_state_dict[k] = v.cpu()
        for k, v in optim_state_dict.items():
            optim_state_dict[k] = v.cpu()

        # Save
        torch.save({
            'net_state_dict': net_state_dict,
            'optim_state_dict': optim_state_dict,
            'config': config,
            'stats':stats}, 
            self.ckpt_path/f"{name}_{epoch:05d}.pth")

    def log(self):
        """
        Empty the buffer and log all elements
        """
        wandb.log(self.buffer)
        self.buffer = {}

    def add_scalar(self, name: str, value: float, iter_n: int, iter_name='epoch'):
        """
        Add a scalar to buffer
        """
        self.buffer[name] = value
        self.buffer[iter_name] = iter_n

    def add_images(self, name: str, images_vector: torch.Tensor, iter_n: int, iter_name='epoch'):
        """
        Add image to buffer
        """
        images = wandb.Image(images_vector, caption=name)
        self.buffer[name] = images
        self.buffer[iter_name] = iter_n

    def log_scalar(self, name: str, value: float, iter_n: int, iter_name='epoch'):
        '''
        Log to wandb
        '''
        wandb.log({name: value, iter_name: iter_n})

    def log_images(self, name: str, images_vector: torch.Tensor, iter_n: int, iter_name='epoch'):
        '''
        Log images to wandb
        image_vector.shape = (C, W, H)
        '''
        images = wandb.Image(images_vector, caption=name)
        wandb.log({name: images, iter_name: iter_n})

    @staticmethod
    def load_model(model_path: Union[str, Path], verbose: bool = True, return_epoch: bool = False):
        """
        Load state dict from pre-trained checkpoint. In case a directory is
          given as `model_path`, the last checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restore checkpoint.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            checkpoint = file_list[-1]
            if verbose:
                print(f'Last checkpoint found: {checkpoint} .')
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        if return_epoch:
            return torch.load(checkpoint), int(str(checkpoint).split("_")[-1][:-4])
        else:
            return torch.load(checkpoint)

    @staticmethod
    def load_checkpoint(model_path: Union[str, Path], verbose: bool = True, return_epoch: bool = False):
        """
        Load state dict e stats from pre-trained checkpoint. In case a directory is
          given as `model_path`, the best (minor loss) checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restore checkpoint.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            # Chose best checkpoint based on minor loss
            if verbose:
                print(f'Search best checkpoint (minor loss)...')
            loss = torch.load(file_list[0])['stats']['mse_loss']
            checkpoint = file_list[0]
            for i in range(1,len(file_list)):
                loss_tmp = torch.load(file_list[i])['stats']['mse_loss']
                if loss_tmp < loss:
                    loss = loss_tmp
                    checkpoint = file_list[i]
            if verbose:
                print(f'Best checkpoint found: {checkpoint} (loss: {loss}).')
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        if return_epoch:
            return torch.load(checkpoint)['state_dict'], int(str(checkpoint).split("_")[-1][:-4])
        else:
            return torch.load(checkpoint)['state_dict']