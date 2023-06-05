#!/usr/bin/env python

"""
    hf_featurize.py
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from datasets import load_dataset, Image, ClassLabel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, default="microsoft/resnet-50")
    parser.add_argument('--dataset', type=str, default="pcuenq/oxford-pets")
    parser.add_argument('--split',   type=str, default="train")
    args = parser.parse_args()
    
    args.outdir = os.path.join('output', args.dataset, args.split, args.model)
    os.makedirs(args.outdir, exist_ok=True)
    
    return args


args = parse_args()

# --
# Helper to unwrap an HF dataset?  I'm sure there's a better way

def normalize_dataset_keys(dataset):
    img_cols = [k for k,v in dataset.features.items() if isinstance(v, Image)]
    lab_cols = [k for k,v in dataset.features.items() if isinstance(v, ClassLabel)]
    
    assert len(img_cols) == 1
    assert len(lab_cols) == 1
    
    dataset = dataset.rename_columns({img_cols[0]:"img", lab_cols[0]:"lab"})
    return dataset


class HFDataset:
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset   = normalize_dataset_keys(dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x        = self.dataset[idx]
        img, lab = x['img'], x['lab']
        
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'][0]
        
        return img, lab


# --
# Load model

print('loading model...', file=sys.stderr)
tfms       = AutoFeatureExtractor.from_pretrained(args.model)
model      = AutoModelForImageClassification.from_pretrained(args.model)
model      = model.eval()
model      = model.cuda()

# !! TODO: Getting the "features" from different models might require dropping different heads w/ different names
assert hasattr(model, 'classifier')
model.classifier = nn.Sequential()

# --
# Load dataset

print('loading dataset....', file=sys.stderr)
hf_dataset = load_dataset(args.dataset)
ds_train   = HFDataset(hf_dataset[args.split], transform=tfms)
dl_train   = DataLoader(ds_train, batch_size=128, num_workers=8)

# --
# Run

print('extracting features...', file=sys.stderr)

X = []
y = []

with torch.inference_mode():
    for xx, yy in tqdm(dl_train, total=len(dl_train)):
        xx  = xx.cuda()
        enc = model(xx).logits
        enc = enc.cpu().numpy() # not logits ...
        
        X.append(enc.squeeze())
        y.append(yy)

X = np.row_stack(X)
y = np.hstack(y)

# --
# Save

np.save(os.path.join(args.outdir, 'X.npy'), X)
np.save(os.path.join(args.outdir, 'y.npy'), y)