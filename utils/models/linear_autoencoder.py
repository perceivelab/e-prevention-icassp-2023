import torch 
import torch.nn as nn

class Model(nn.Module):
  
    def __init__(self, args):
        
        super(Model, self).__init__()

        args_defaults = dict(
            input_features = 10,
            input_timepoints = 48,
            bottleneck = 60
        )
            
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        self.encoder = torch.nn.Sequential(
            nn.Linear(self.input_features*self.input_timepoints, 120),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, self.bottleneck),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(self.bottleneck, 120),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, self.input_features*self.input_timepoints),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:

        latent_space = self.encoder(src.view(src.size(0), -1))
        reconstruction = self.decoder(latent_space)

        return reconstruction