import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstBlock(nn.Module):
    def __init__(self, i, o, k, s, d):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        - k (int): kernel size
        - s (int): stride
        - d (int): dilation
        """
        super().__init__()
        # Compute padding
        p = (k//2)*d
        # Layers
        self.conv = nn.Conv1d(i, o, k, s, p, d)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class Block(nn.Module):
    def __init__(self, i, o, k, s=1, d=1):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        - k (int): kernel size
        - s (int): stride
        - d (int): dilation
        """
        super().__init__()
        # Compute padding
        p = (k//2)*d
        # Layers
        self.conv1 = nn.Conv1d(i, o, k, s, p, d)
        self.bn1 = nn.BatchNorm1d(o)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(o, o, k, s, p, d)
        self.bn2 = nn.BatchNorm1d(o)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ChangeChannels(nn.Module):
    def __init__(self, i, o, act=True):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm1d(i)
        self.conv = nn.Conv1d(i, o, 1)
        self.act = act

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        if self.act:
            x = F.relu(x)
        return x


class TransDown(nn.Module):
    def __init__(self, i, o):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm1d(i)
        self.conv = nn.Conv1d(i, o, 4, 2, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = F.relu(x)
        return x


class TransUp(nn.Module):
    def __init__(self, i, o):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm1d(i)
        self.conv = nn.ConvTranspose1d(i, o, 4, 2, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = F.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, i, o, reduce):
        """
        Args:
        - i (int): input channels
        - o (int): output channels (must be even)
        - reduce (int): max reduction factor between consecutive layers
        """
        super().__init__()
        # Set args
        self.o = o
        # Layers
        self.reduction_layers = nn.ModuleList()
        curr_channels = i
        while curr_channels//reduce > o:
            self.reduction_layers.append(ChangeChannels(
                curr_channels, curr_channels//reduce))
            curr_channels //= reduce
        self.out = ChangeChannels(curr_channels, o, False)

    def forward(self, x):
        # Reduce channels
        for i in range(len(self.reduction_layers)):
            x = self.reduction_layers[i](x)
        # Compute output
        x = self.out(x)
        # Split channels between mu and logvar
        mu = x[:, :self.o//2, :]
        logvar = x[:, self.o//2:, :]
        # Return
        return mu, logvar


class Model(nn.Module):
    def __init__(self, args):
        """
        Args (dictionary):
        - data_len (int): length of input signal
        - data_channels (int): channels of input signal
        - layers_base (int): see below
        - channels_base (int): starting number of channels
        - min_spatial_size (int): minimum spatial size to keep
        - start_dilation (int): initial dilation value
        - min_sig_dil_ratio (int): min ratio between signal length and dilation
        - max_channels (int): max number of channels per layer
        - h_size (int): bottleneck (i.e., mu/logvar) size
        - enable_variational (boolean): to enable variational AE
        In the encoder, after every group of layers_base layers, a downsampling
        block is added, as long as the spatial size is greater than or equal to
        min_spatial_size. Same in the decoder.
        """
        super().__init__()

        args_defaults = dict(
            data_len = 2160, 
            data_channels = 10, 
            layers_base=1, 
            channels_base=16, 
            min_spatial_size = 2, 
            start_dilation = 3, 
            min_sig_dil_ratio = 50, 
            max_channels = 1024, 
            h_size = 64, 
            enable_variational = False
        )
            
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        # Check data length
        if (self.data_len & (self.data_len-1)) != 0:
            raise AttributeError(f'Warning: model expects input to be power of 2 (got {self.data_len})')

        # Track number of channels per block of layers
        layer_channels = []

        # Encoder (temp)
        self.encoder = nn.ModuleList()
        curr_data_len = self.data_len
        curr_channels = self.channels_base
        curr_dilation = self.start_dilation

        # Add encoder first block
        self.encoder.append(FirstBlock(self.data_channels,
                                       curr_channels, 3, 1, curr_dilation))

        # Add encoder blocks
        while curr_data_len > self.min_spatial_size:
            # Track channels
            layer_channels.append(curr_channels)
            # Add blocks
            for _ in range(self.layers_base):
                # Add block
                self.encoder.append(
                    Block(curr_channels, curr_channels, 3, 1, curr_dilation))
            # Add downsampling block
            self.encoder.append(TransDown(curr_channels, min(
                curr_channels*2, self.max_channels)))
            # Update values
            if curr_channels < self.max_channels:
                curr_channels *= 2
            curr_data_len /= 2
            while curr_dilation > 1 and curr_data_len/curr_dilation < self.min_sig_dil_ratio:
                curr_dilation -= 1

        # Bottleneck
        self.bottleneck = Bottleneck(curr_channels, self.h_size*2, 4)

        # Decoder
        self.decoder = nn.ModuleList()

        # Add decoder first block
        self.decoder.append(ChangeChannels(self.h_size, curr_channels))

        # Add decoder blocks
        while curr_data_len < self.data_len:
            # Add blocks
            for i in range(self.layers_base):
                # Add block
                self.decoder.append(Block(curr_channels, curr_channels, 3))
            # Add upsampling block
            prev_curr_channels = curr_channels
            curr_channels = layer_channels.pop()
            self.decoder.append(TransUp(prev_curr_channels, curr_channels))
            # Update values
            curr_data_len *= 2

        # Add decoder final block
        self.decoder.append(ChangeChannels(
            curr_channels, self.data_channels, False))

        # Create sequential containers
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #eps = torch.ones(std.size()).to(std.device)
        return mu + eps*std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Bottleneck
        mu, logvar = self.bottleneck(x)
        x = mu

        # Only if variational AE
        if self.enable_variational and self.training:
            # Reparameterization
            z = self.reparameterize(mu, logvar)
            x = z

        # Decoder
        x = self.decoder(x)

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