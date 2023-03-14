import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, args):

        # init args
        super(Model, self).__init__()
        args_defaults = dict(
                in_channels=8, 
                num_classes=64, 
                verbose=False,
                dropout=0.25
            )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.convolutions = nn.Sequential(

            # Layer 1
            nn.Conv1d(self.in_channels, 32, 72, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),

            # Layer 2
            nn.Conv1d(32, 32, 12, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),

            # Layer 3
            nn.Conv1d(32, 64, 24),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 4
            nn.Conv1d(64, 64, 12, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 5
            nn.Conv1d(64, 128, 24),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout)
        )

        self.pooling = nn.AdaptiveAvgPool1d(64)

        self.fully_connected = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.num_classes),
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        if self.verbose:
            print(x.size())

        if len(x.shape) == 4:
            x = x.squeeze(0)

        x = self.convolutions(x)

        if self.verbose:
            print(x.size())

        x = self.pooling(x)

        if self.verbose:
            print(x.size())

        B = x.size(0)
        x = x.view(B, -1)
        logits = self.fully_connected(x)

        if self.verbose:
            print(logits.size())    

        return logits