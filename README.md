# CAP5610-Project
This repository provides the code for CAP5610 course project.

## Requirement
	- Python >= 3.6, numpy, matplotlib, pillow, ptflops, timm
    - PyTorch >= 1.8.1, torchvision >= 0.11.1
	
## Training and Evaluation
For training: 

    python train.py

Simply run the scripts like:

    sh eval.sh

You may need to specify the GPUs for training in "train.py". Remove the second line if you want to train the simple stage-1 model. Change the "--dataset" to train on other datasets. The code follows the multiprocessing distributed training style from [PyTorch](https://github.com/pytorch/examples/tree/main/imagenet) and [Moco](https://github.com/facebookresearch/moco), but it only uses one GPU by default for training. You may need to tune the learning rate for multi-GPU training, e.g. [linear scaling rule](https://arxiv.org/pdf/1706.02677.pdf). There may be memory issue if training with too many GPUs with our sampler. 

We follow timm, ViT and [Deit](https://github.com/facebookresearch/deit) for pytorch implementation of vision transformer. We use the pytorch implementation of [ASAM](https://github.com/davda54/sam).

## Reference
    - http://mvrl.cs.uky.edu/datasets/cvusa/
    - https://github.com/Jeff-Zilence/VIGOR
    - https://github.com/Liumouliu/OriCNN
    - https://github.com/facebookresearch/deit
    - https://github.com/facebookresearch/moco
    - https://github.com/davda54/sam
    - https://github.com/david-husx/crossview_localisation.git

