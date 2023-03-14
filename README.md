<div align="center">

# Ensemble and personalized transformer models for subject identification and relapse detection in e-Prevention Challenge
Salvatore Calcagno, Raffaele Mineo, Daniela Giordano, Concetto Spampinato

</div>

# Overview
Official PyTorch implementation of paper: <b>"Ensemble and personalized transformer models for subject identification and relapse detection in e-Prevention Challenge"</b>

# Abstract
We present the devised solutions for subject identification and relapse detection of the <a href='https://robotics.ntua.gr/eprevention-sp-challenge/'>e-Prevention Challenge</a> hosted at the ICASSP 2023 conference <a id="1">[1]</a> <a id="1">[2]</a>  <a id="1">[3]</a>. 
We specifically design an ensemble scheme of six models - five transformer-based ones and a CNN model - for the identification of subjects from wearable devices, while a personalized - one for each subject - scheme is used for relapse detection in psychotic disorder. Our final submitted solutions yield top performance on both tracks of the challenge: we ranked second on the subject identification task (with an accuracy of 93.85\%) and first on the   relapse detection task (with a ROC-AUC and PR-AUC of about 0.65).


# Method
## Track 1

We show below the employed architectures for the ensemble model

<table class="data-table", style="border-collapse: collapse;">
    <tr>
        <th class="border-top border-bottom", style="border-top: 1px solid #000; border-bottom: 1px solid #000;">Model Type</th>
        <th class="border-top border-bottom", style="border-top: 1px solid #000; border-bottom: 1px solid #000;">Architecture Details</th>
        <th class="border-top border-bottom", style="border-top: 1px solid #000; border-bottom: 1px solid #000;">Training Setting</th>
    </tr>
    <tr>
        <td>CNN (Transformer Ablation)</td>
        <td>5 convolutional layers (conv1D, ReLU, BatchNorm, Dropout), AdaptiveAvgPool1d, Time2Vec, Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 1e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>Embedding (5 convolutional layers, AdaptiveAvgPool1d)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 128, nlayers 2 , nhead 2, d_hid 512)<br/>Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>Embedding (5 convolutional layers, AdaptiveAvgPool1d)<br/>Positional Embedding (Time2Vec)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 2, d_hid 128 )<br/>Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>Embedding (5 convolutional layers, AdaptiveAvgPool1d)<br/>Positional Embedding (Time2Vec)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 2, d_hid 128 )<br/>Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>Embedding (5 convolutional layers, AdaptiveAvgPool1d)<br/>Positional Embedding (Time2Vec)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 2, d_hid 768)<br/>Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>Embedding (5 convolutional layers, AdaptiveAvgPool1d)<br/>Positional Embedding (Time2Vec)<br/>Transformer Encoder (model depth 128, nlayers 2 , nhead 2, d_hid 768)<br/>Fully Connected Classification Head</td>
        <td>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
</table>

## Track 2
Best configurations were found using grid search for each subject:

For CNN-based models we tested the following parameters:
```
"parameters": {
    "subject": {"values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "data_type": {"values": ["aggregated", "raw"]},
    "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
    "enable_variational": {"values": [0, 1]},
    "model": {"values": ["cnn1d_autoencoder", "volund"]}
}
```
For Transformer-based models we tested the following parameters:
```
"parameters": {
    "subject": {"values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "data_type": {"values": ["aggregated"]},
    "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
    "enable_variational": {"values": [0, 1]},
    "model": {"values": ["transformer_autoencoder"]},
    "d_model": {"values": [32, 64, 128]},
    "nhead": {"values": [4, 8, 16]},
    "nlayers": {"values": [2, 4]},
}
```
We show below the employed architectures for each subject.

<table class="data-table">
    <tr>
        <th class="border-top border-bottom">Subject</th>
        <th class="border-top border-bottom">Model Type&nbsp;</th>
        <th class="border-top border-bottom">Architecture Details</th>
        <th class="border-top border-bottom">Training Setting</th>
    </tr>
    <tr>
        <td>0</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Transformer Decoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 128, nlayers 2 , nhead 16, d_hid 2048)<br/>Transformer Decoder (model depth 128, nlayers 2 , nhead 16, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>2</td>
        <td>CNN</td>
        <td>CNN Encoder: 5 convolutional layers (conv1D, ReLU, BatchNorm, Dropout)<br/>Bottleneck: conv1D, ReLU<br/>CNN Decoder: 5 transposed convolutional layers (convTranspose1D, ReLU, BatchNorm, Dropout)</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 4, d_hid 2048)<br/>Transformer Decoder (model depth 32, nlayers 2 , nhead 4, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-4, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Transformer Decoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Volund</td>
        <td></td>
        <td>data type raw<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>6</td>
        <td>CNN</td>
        <td>CNN Encoder: 5 convolutional layers (conv1D, ReLU, BatchNorm, Dropout)<br/>Bottleneck: conv1D, ReLU<br/>CNN Decoder: 5 transposed convolutional layers (convTranspose1D, ReLU, BatchNorm, Dropout)<br/>Linear Mapping</td>
        <td>data type raw<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>7</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Transformer Decoder (model depth 32, nlayers 2 , nhead 8, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>8</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 128, nlayers 2 , nhead 8, d_hid 2048)<br/>Transformer Decoder (model depth 128, nlayers 2 , nhead 8, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
    <tr>
        <td>9</td>
        <td>Transformer</td>
        <td>Embedding (linear projection)<br/>Positional Embedding (sin and cos encoding)<br/>Transformer Encoder (model depth 128, nlayers 2 , nhead 8, d_hid 2048)<br/>Transformer Decoder (model depth 128, nlayers 2 , nhead 8, d_hid 2048)<br/>Linear Mapping</td>
        <td>data type aggregated<br/>batch size 64<br/>Adam optimizer<br/>scheduler reduceLROnPlateau (initial learning rate 5e-3, factor 0.5, patience 10 epochs)</td>
    </tr>
</table>

# How to run

## Pre-requisites
- NVIDIA GPU (Tested on Nvidia A6000 GPUs )
- Wandb account (change entity and project name in scripts)
- The datasets provided for track 1 and track 2 should be placed in ../datasets
- [Requirements](requirements.txt)

## Track 1

### **Train Ensemble Models**
To start training, simply run the following commands.
Each command shows a model configuration, which will be used in the ensemble during validation and test.
Please note that the first two commands are the same, since the same model was used with a weight of 2 in the voting scheme.

```
python train_track1.py --window_size 2160 --model transformer --d_model 32 --nhead 2 --d_hid 128 --nlayers 2 --learning_rate 5e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width3_stride3 --data_dir data/track1/width3_stride3
python train_track1.py --window_size 2160 --model transformer --d_model 32 --nhead 2 --d_hid 128 --nlayers 2 --learning_rate 5e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width3_stride3 --data_dir data/track1/width3_stride3
python train_track1.py --window_size 2160 --model transformer --d_model 32 --nhead 2 --d_hid 768 --nlayers 2 --learning_rate 5e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width3_stride3 --data_dir data/track1/width3_stride3
python train_track1.py --window_size 1080 --model transformer --d_model 128 --nhead 2 --d_hid 768 --nlayers 2 --learning_rate 5e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width1_5_stride1_5 --data_dir data/track1/width1_5_stride1_5
python train_track1.py --window_size 2160 --model transformer_ablation_time2vec --d_model 128 --nhead 2 --d_hid 512 --nlayers 2 --learning_rate 5e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width3_stride3 --data_dir data/track1/width3_stride3
python train_track1.py --window_size 2160 --model transformer_ablation --learning_rate 1e-4 --enable_scheduler 1 --batch_size 64 --split_path data/track1/width3_stride3 --data_dir data/track1/width3_stride3

```
### **Test Example**
The code expects a txt file `ensemble.txt` with the list of names of models (the structure is shown below). The file should be placed in the root directory.

```
YYYY-MM-DD_hh-mm-ss_<model1>
YYYY-MM-DD_hh-mm-ss_<model2>
YYYY-MM-DD_hh-mm-ss_<model3>
YYYY-MM-DD_hh-mm-ss_<model4>
YYYY-MM-DD_hh-mm-ss_<model5>
YYYY-MM-DD_hh-mm-ss_<model6>
```
Model names `YYYY-MM-DD_hh-mm-ss_<model*>` should be retrieved from the directory list in the experiments folder, after training.

Run the following to retrieve accuracies of single and ensemble models on the provided validation set.
```
python test_track1.py --split val
```

Use `--split test` if you want to obtain predictions over test samples. Predictions will be saved into a file named `test_track1.csv` We don't have the ground truth for this split.

The default essemble scheme is sum. You can use the `--scheme` argument if you want to change the ensemble scheme. Allowed schemes are min, max and sum.


## Track 2

### **Train Best Configurations**

To start training, simply run the following commands.
```
python train_track2.py --subject 0 --model transformer_autoencoder --d_model 32 --nhead 8 --nlayers 2 --data_type aggregated --learning_rate 5e-3
python train_track2.py --subject 1 --model transformer_autoencoder --d_model 128 --nhead 16 --nlayers 2 --data_type aggregated --learning_rate 5e-4
python train_track2.py --subject 2 --model cnn1d_autoencoder --data_type aggregated --learning_rate 5e-3
python train_track2.py --subject 3 --model transformer_autoencoder --d_model 32 --nhead 4 --nlayers 2 --data_type aggregated --learning_rate 5e-4
python train_track2.py --subject 4 --model transformer_autoencoder --d_model 32 --nhead 8 --nlayers 2 --data_type aggregated --learning_rate 5e-3
python train_track2.py --subject 5 --model volund --data_type raw --learning_rate 5e-3
python train_track2.py --subject 6 --model cnn1d_autoencoder --data_type raw --learning_rate 5e-3
python train_track2.py --subject 7 --model transformer_autoencoder --d_model 128 --nhead 8 --nlayers 2 --data_type aggregated --learning_rate 5e-3
python train_track2.py --subject 8 --model transformer_autoencoder --d_model 128 --nhead 8 --nlayers 2 --data_type aggregated --learning_rate 5e-3
python train_track2.py --subject 9 --model transformer_autoencoder --d_model 128 --nhead 8 --nlayers 2 --data_type aggregated --learning_rate 5e-3
```
### **Test Example**
The code expects a txt file `best_models.txt` with the list of names of models, the same as for the first track. The file should be placed in the root directory.

Run the following to retrieve performace (ROC-AUC, PRC-AUC and the harmonic mean of the previous two) of single models on the provided validation set.
```
python test_track2.py --split val
```

Use `--split test` if you want to obtain predictions over test samples. Predictions will be saved into a file named `test_track2.csv` We don't have the ground truth for this split.

# References
[1] A Zlatintsi, P P Filntisis, C Garoufis, N Efthymiou,
P Maragos, A Menychtas, I Maglogiannis, et al., “E-
prevention: Advanced support system for monitoring and
relapse prevention in patients with psychotic disorders
analyzing long-term multimodal data from wearables and
video captures,” Sensors, vol. 22, no. 19, 2022.

[2] G Retsinas, P P Filntisis, N Efthymiou, E Theodosis,
A Zlatintsi, and P Maragos, “Person identification using
deep convolutional neural networks on short-term signals
from wearable sensors,” in ICASSP. IEEE, 2020.

[3] M Panagiotou, A Zlatintsi, PP Filntisis, AJ Roumeliotis,
N Efthymiou, and P Maragos, “A comparative study of
autoencoder architectures for mental health analysis us-
ing wearable sensors data,” in EUSIPCO. IEEE, 2022.

[4] S M Kazemi, R Goel, S Eghbali, J Ramanan, J Sahota,
S Thakur, S Wu, C Smyth, P Poupart, and M Brubaker,
“Time2vec: Learning a vector representation of time,”
arXiv preprint arXiv:1907.05321, 2019.


<!--- # Acknowledgements
This code is taken from https://github.com/IngRaffaeleMineo/3D-BCPTcode and modified to our purposes. -->
