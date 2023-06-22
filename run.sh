#!/bin/bash

# run.sh

# --
# Extract features from HF datasets

python hf_featurize.py --model microsoft/resnet-50 --dataset cifar10
python hf_featurize.py --model microsoft/resnet-50 --dataset Multimodal-Fatima/OxfordPets_train
python hf_featurize.py --model microsoft/resnet-50 --dataset Multimodal-Fatima/StanfordCars_train
python hf_featurize.py --model microsoft/resnet-50 --dataset jonathan-roberts1/NWPU-RESISC45
python hf_featurize.py --model microsoft/resnet-50 --dataset nelorth/oxford-flowers
python hf_featurize.py --model microsoft/resnet-50 --dataset fashion_mnist
python hf_featurize.py --model microsoft/resnet-50 --dataset food101

python hf_featurize.py --model facebook/convnext-base-224 --dataset cifar10 --img_field img
python hf_featurize.py --model facebook/convnext-base-224 --dataset Multimodal-Fatima/OxfordPets_train
python hf_featurize.py --model facebook/convnext-base-224 --dataset Multimodal-Fatima/StanfordCars_train
python hf_featurize.py --model facebook/convnext-base-224 --dataset jonathan-roberts1/NWPU-RESISC45
python hf_featurize.py --model facebook/convnext-base-224 --dataset nelorth/oxford-flowers
python hf_featurize.py --model facebook/convnext-base-224 --dataset fashion_mnist
python hf_featurize.py --model facebook/convnext-base-224 --dataset food101

python hf_featurize.py --model google/vit-base-patch16-224 --dataset cifar10 --img_field img
python hf_featurize.py --model google/vit-base-patch16-224 --dataset Multimodal-Fatima/OxfordPets_train
python hf_featurize.py --model google/vit-base-patch16-224 --dataset Multimodal-Fatima/StanfordCars_train
python hf_featurize.py --model google/vit-base-patch16-224 --dataset jonathan-roberts1/NWPU-RESISC45
python hf_featurize.py --model google/vit-base-patch16-224 --dataset nelorth/oxford-flowers
python hf_featurize.py --model google/vit-base-patch16-224 --dataset fashion_mnist
python hf_featurize.py --model google/vit-base-patch16-224 --dataset food101

