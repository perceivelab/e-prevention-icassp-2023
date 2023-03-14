import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, args):

        # init args
        super(Model, self).__init__() 
        args_defaults = dict(
                in_channels=12,
                verbose=False,
                dropout=0.25,
                data_type='raw',
                enable_variational=False
            )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        if self.data_type == 'raw':

            self.encoder = nn.Sequential(

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

            self.bottleneck = nn.Sequential(
                nn.Conv1d(128, 256, 1),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(

                # Layer 1
                nn.ConvTranspose1d(128, 64, 24),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 2
                nn.ConvTranspose1d(64, 64, 12, stride=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 3
                nn.ConvTranspose1d(64, 32, 24),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 4
                nn.ConvTranspose1d(32, 32, 12, stride=2 , output_padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 5
                nn.ConvTranspose1d(32, self.in_channels, 72, stride=2),
            )
        
        else: # aggregated
            self.encoder = nn.Sequential(

                # Layer 1
                nn.Conv1d(self.in_channels, 32, 4),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 2
                nn.Conv1d(32, 32, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 3
                nn.Conv1d(32, 64, 4),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 4
                nn.Conv1d(64, 64, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 5
                nn.Conv1d(64, 128, 4),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(self.dropout)
            )

            self.bottleneck = nn.Sequential(
                nn.Conv1d(128, 256, 1),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(

                # Layer 1
                nn.ConvTranspose1d(128, 64, 4),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 2
                nn.ConvTranspose1d(64, 64, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(self.dropout),

                # Layer 3
                nn.ConvTranspose1d(64, 32, 4),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 4
                nn.ConvTranspose1d(32, 32, 3, stride=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(self.dropout),

                # Layer 5
                nn.ConvTranspose1d(32, self.in_channels, 4),
            )

        """
        self.pooling = nn.AdaptiveAvgPool1d(64)

        self.fully_connected = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.num_classes),
        )

        self.softmax = nn.Softmax(dim=1)
        """

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):

        if self.verbose:
            print(x.size())

        if len(x.shape) == 4:
            x = x.squeeze(0)

        x = self.encoder(x)

        if self.verbose:
            print(x.size())

        # Bottleneck
        x = self.bottleneck(x)

        # Separate mu and logvar
        _, O, _ = x.size()
        mu = x[:, :O//2, :]
        logvar = x[:, O//2:, :]
        x = mu
        
        if self.enable_variational and self.training:
            # Reparameterization
            z = self.reparameterize(mu, logvar)
            x = z

        x = self.decoder(x)

        if self.verbose:
            print(x.size())

        #B = x.size(0)
        #x = x.view(B, -1)
        #logits = self.fully_connected(x)   

        # Return
        if self.enable_variational and self.training:
            return x, mu, logvar
        return x

    def loss(self, recon_x, x, mu, logvar):
        # MSE loss
        loss = F.mse_loss(recon_x, x, reduction='mean')

        # Only if variational AE
        if self.enable_variational:
            # KLD loss
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss