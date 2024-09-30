## DDOPaI

This is the implementation of the paper:
"Leveraging Discriminative Data: A Pathway to High-Performance, Stable One-shot Network Pruning at Initialization"

### Citation
```
@article{YANG2024128529,
title = {Leveraging discriminative data: A pathway to high-performance, stable One-shot Network Pruning at Initialization},
journal = {Neurocomputing},
volume = {610},
pages = {128529},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128529},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224013006},
author = {Yinan Yang and Ying Ji and Jien Kato},
keywords = {Network pruning, One-shot, Pruning at initialization, Discriminative data}
}
```

## Step 0. File and Data Structure
- ..
  - data
    - imagenet
      - train
      - val
    - tiny_imagenet
      - train
      - val
  - DDOPaI
      - tmp # Generated Discriminative Data is here.

## Step 1. Extract Discriminative Data
Extract Discriminative Data using `extractdd.py` with TensorFlow, as in the ACE implementation. The environment is created using Docker to manage package versions. The Docker build command lines are as follows:
```bash
cd env
docker rm --force extractdd
docker-compose build
docker-compose up -d
```

After the container is set up, the following command line extracts Discriminative Data from a pre-trained Resnet-50:
```bash
python extracdd.py --teacher_model resnet50
```
We are pleased to provide discriminative data samples extracted from a pre-trained ResNet-50 using ImageNet-1k. However, due to ImageNet's data policy, we cannot offer these samples directly. If users need access to these discriminative data, please contact the authors. We will provide a cloud download link to facilitate access.


## Step 1.5. Stitching Concepts
The stitching_concept.ipynb notebook provides a demo to stitch concepts into a stitching image patch.

## Step 2. Pruning with Discrimnative Data

The pruning method implementations are from: 
https://github.com/alecwangcq/GraSP
https://github.com/JingtongSu/sanity-checking-pruning/


```bash
conda create -n DDOPaI python=3.8
conda activate DDOPaI
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install matplotlib
pip install tqdm
pip install tensorboardX
pip install easydict
pip install einops
pip install scikit-learn
pip install scikit-image
```

The pruning experiments of resent50 on Imagenet is:
```bash
python main_finetune_imagenet.py --config configs/imagenet/resnet50/SNIP_DD_60.json
```

DD is for Discriminative Data-based Pruning.
SD is for Stitching Data-based Pruning.

The fine-tune process is:
```bash
python main_finetune_imagenet.py --resume_pruned <your_pruned_model_path>
```


## Ablations on Data
In the results/ablation directory, we provide two implementation notebooks corresponding to the data quality ablation experiments presented in our paper (Figure 6). By altering the brightness of input images, we examine the impact of data quality changes on pruning methods.


## PyTorch implementation and Swin-Transformer experiment.
In the ACE/pytorch directory, we offer a PyTorch ACE implementation available from this repository. https://github.com/amiratag/ACE

The main_vit_cifar10.py script demonstrates pruning experiments on the Swin-Transformer model using the CIFAR-10 dataset.
