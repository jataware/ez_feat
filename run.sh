#!/bin/bash

# run.sh

# --
# Extract features from HF datasets

python -m ez_feat --model microsoft/resnet-50 --dataset cifar10
python -m ez_feat --model microsoft/resnet-50 --dataset Multimodal-Fatima/OxfordPets_train
python -m ez_feat --model microsoft/resnet-50 --dataset Multimodal-Fatima/StanfordCars_train
python -m ez_feat --model microsoft/resnet-50 --dataset jonathan-roberts1/NWPU-RESISC45
python -m ez_feat --model microsoft/resnet-50 --dataset nelorth/oxford-flowers
python -m ez_feat --model microsoft/resnet-50 --dataset fashion_mnist
python -m ez_feat --model microsoft/resnet-50 --dataset food101

python -m ez_feat --model facebook/convnext-base-224 --dataset cifar10 --img_field img
python -m ez_feat --model facebook/convnext-base-224 --dataset Multimodal-Fatima/OxfordPets_train
python -m ez_feat --model facebook/convnext-base-224 --dataset Multimodal-Fatima/StanfordCars_train
python -m ez_feat --model facebook/convnext-base-224 --dataset jonathan-roberts1/NWPU-RESISC45
python -m ez_feat --model facebook/convnext-base-224 --dataset nelorth/oxford-flowers
python -m ez_feat --model facebook/convnext-base-224 --dataset fashion_mnist
python -m ez_feat --model facebook/convnext-base-224 --dataset food101

python -m ez_feat --model google/vit-base-patch16-224 --dataset cifar10 --img_field img
python -m ez_feat --model google/vit-base-patch16-224 --dataset Multimodal-Fatima/OxfordPets_train
python -m ez_feat --model google/vit-base-patch16-224 --dataset Multimodal-Fatima/StanfordCars_train
python -m ez_feat --model google/vit-base-patch16-224 --dataset jonathan-roberts1/NWPU-RESISC45
python -m ez_feat --model google/vit-base-patch16-224 --dataset nelorth/oxford-flowers
python -m ez_feat --model google/vit-base-patch16-224 --dataset fashion_mnist
python -m ez_feat --model google/vit-base-patch16-224 --dataset food101


python -m ez_feat --model openai/clip-vit-base-patch32 --dataset Multimodal-Fatima/OxfordPets_train