# Rewrite-for-Chinese-character
This is a project to achieve style transfer for Chinese character, it is based on vae structure.
## Introduction
Structure of this model is different from encoder-decoder, input of the network are three random character pictures of target style, the embedding_ids control the label of charactor content.
## How to Use
### Step Zero
Prepare you own fonts to train the model, save your fonts at './font/'.
### Requirement
* Python 3.6
* CUDA
* cudnn
* Tensorflow >= 1.0.1
* Pillow(PIL)
* numpy >= 1.12.1
* scipy >= 0.18.1
* imageio
### Step One
Generate train data, you can dirrectly run the command:
```sh
pythone3 preprocess.py
```
images will be saved as arrays at './data.npy' and './test.npy'
### Step Two
Train your own model
```sh
python3 train.py
```
## Infer
The infer step for this model is generate a random code and input the code into decoder.
```sh
python3 infer.py
```
