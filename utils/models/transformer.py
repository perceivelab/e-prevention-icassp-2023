import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils.models.time2vec import Model as Time2Vec
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Module, Linear

class Embedding(Module):
    """
    The input embedding. Convert a sequence of size (batch_size, feature_dim, seq_len)
    into a different space of size (batch_size, depth_model, output_seq_len) with depth_model > feature_dim and output_seq_len < seq_len
    """

    def __init__(self, input_features, output_features, output_size, dropout=0.25):
        super(Embedding, self).__init__()
        self.in_channels = input_features
        self.out_channels = output_features
        self.output_size = output_size
        self.dropout = dropout

        self.convolutions = nn.Sequential(

            # Layer 1
            nn.Conv1d(self.in_channels, 64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 2
            nn.Conv1d(64, 64, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 3
            nn.Conv1d(64, 256, 8),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),

            # Layer 4
            nn.Conv1d(256, 256, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),

            # Layer 5
            nn.Conv1d(256, self.out_channels, 8),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
            nn.Dropout(self.dropout),
        )

        self.pooling = nn.AdaptiveAvgPool1d(self.output_size)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.pooling(x)
        return x

class Embeddings(nn.Module):
    """
    The set of the embeddings. It computes also the Positional Encoding and adds the CLS token
    """

    def __init__(self, input_features, output_features, output_size, dropout=0.25):
        super(Embeddings, self).__init__()

        self.input_embeddings = Embedding(input_features, output_features, output_size-1)
        self.position_embeddings = Time2Vec(activation='sin', hidden_dim=output_features)
        self.cls_token = nn.Parameter(torch.zeros(1, output_features, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # get size and expand cls
        B, _, _ = x.size()
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # get input embeddings
        x = self.input_embeddings(x)
        # cat the CLS token
        x = torch.cat((cls_tokens, x), dim=2)
        _, _, T = x.size()

        # compute position embeddings
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(1).to(x.device)
        position_embeddings = self.position_embeddings(pos).t()

        embeddings = x + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerModel(nn.Module):
    """
    Transformer Encoder: it includes embeddings and Endoder layers.
    Note that the input is (batch_size, feature_dim, seq_len), the output (seq_len, batch_size, depth_model)
    """
    
    def __init__(self, input_features:int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int = 512, dropout: float = 0.5):
        super().__init__()
        self.embeddings = Embeddings(input_features, d_model, seq_len, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # size (B, F, L)
        src = self.embeddings(src)
        src = src.permute(2, 0, 1)
        #print(src.size())
        
        output = self.transformer_encoder(src)
        # output_size is (L, B, F)
        
        return output

class Model(nn.Module):
    """
    Final model with the classification head
    """
    
    def __init__(self, args):
        
        super(Model, self).__init__()

        args_defaults = dict(
            input_features = 8, 
            d_model = 128, 
            nhead = 12, 
            d_hid = 1024,
            nlayers = 12, 
            num_classes = 46, 
            dropout = 0.25,
            seq_len = 512
        )
            
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        self.transformer = TransformerModel(
                input_features = self.input_features, 
                d_model = self.d_model, 
                nhead = self.nhead, 
                d_hid = self.d_hid, 
                nlayers = self.nlayers, 
                seq_len = self.seq_len, 
                dropout = self.dropout
            )

        
        self.classification_head = torch.nn.Sequential(
            Linear(self.d_model, 256),
            nn.Sigmoid(),
            Linear(256, self.num_classes)
        )

    def forward(self, src: Tensor) -> Tensor:
        if len(src.size()) == 4:
            src = src.squeeze(0)
        last_hidden_state = self.transformer(src)
        cls_token = last_hidden_state[0, :, :]
        #print(cls_token.size())
        logits = self.classification_head(cls_token)
        #print(logits.size())

        return logits
