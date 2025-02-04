#!/bin/bash
# mushrooms mnist_mlp cifar10_resnet cifar10_densenet cifar100_res cifar100_dense fashion_effb1 svhn_wrn trans_enc trans_xl
CUDA_VISIBLE_DEVICES=7 python trainval.py -e fashion_effb1 -sb results -d data -r 1 &
CUDA_VISIBLE_DEVICES=2 python trainval.py -e cifar10_densenet -sb results -d data -r 1 &
CUDA_VISIBLE_DEVICES=3 python trainval.py -e cifar10_resnet -sb results -d data -r 1 &
CUDA_VISIBLE_DEVICES=6 python trainval.py -e cifar100_res -sb results -d data -r 1 &
CUDA_VISIBLE_DEVICES=4 python trainval.py -e cifar100_dense -sb results -d data -r 1 &
CUDA_VISIBLE_DEVICES=5 python trainval.py -e svhn_wrn -sb results -d data -r 1 &
# CUDA_VISIBLE_DEVICES=2 python trainval.py -e w8a -sb results -d data -r 1 &
# CUDA_VISIBLE_DEVICES=1 python trainval.py -e ijcnn -sb results -d data -r 1
# python plot.py -p ijcnn