# MHCARes-SRNet
The implementation of "MHARes-SRNet： A Multi-Head Convolutional Attention Residual Network for Super-Resolution Reconstruction with Fine Feature Extraction"

Dependencies：
Python 3.7
NumPy 1.21.6
Pytorch 1.13.0
opencv-python
torchsummaryX
tensorboardX
tensorboard 
lmdb
pyyaml
tb-nightly
future
scikit-image

# Dataset:
Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Baidu Drive](https://pan.baidu.com/s/1cZM76IAvaRAtwcXMBHlpnw?pwd=fa0j) 
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from [Baidu Drive](https://pan.baidu.com/s/1nQUN1Hi17zaOpqndg9z62A?pwd=6681)

# LR dataset prepare
1.cd to /MHCARes-SRNet/codes/data_scripts 
2.set and run generate_mod_LR_bic.py

# Train:
1.open a terminal and cd to codes
2.type and run: 
```
python train.py -opt ./options/train/train_SRResNet_X4.yml ( yml file path of train)
```
Training weights and logs will be saved to the ./expetiments/

# Test:
1.open a terminal and cd to codes
2.type and run: 
```
python test.py -opt ./options/test/test_SRResNet_X4.yml (yml file path of test)
```
Test results and logs will be saved to ./results/

# File description： 
./codes/models/archs/ —— Network
./codes\options/ —— Configuration file


Thanks Zhao et.al. for their excellent open source project at: https://github.com/zhaohengyuan1/PAN.
