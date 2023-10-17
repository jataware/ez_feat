#!/usr/bin/env python

"""
    hf_featurize.py
    
    ?? Is there a better / faster way to do this natively w/ huggingface?
        - Want the simplest code that can extract embeddings for (model, dataset) pairs as simply as possible
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

from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    CLIPFeatureExtractor,
    CLIPModel
)

from datasets import load_dataset, Image, ClassLabel

# device = 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, default="microsoft/resnet-50")

    parser.add_argument('--dataset',   type=str, default="pcuenq/oxford-pets")
    parser.add_argument('--split',     type=str, default="train")
    parser.add_argument('--img_field', type=str, default="image")
    parser.add_argument('--lab_field', type=str, default="label")
    args = parser.parse_args()
    
    args.outdir = os.path.join('output', args.dataset, args.split, args.model)
    os.makedirs(args.outdir, exist_ok=True)
    
    return args


args = parse_args()

# --
# Helper to unwrap an HF dataset?  I'm sure there's a better way

class HFDataset:
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset   = dataset
        
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

print('hf_featurize: loading model...', file=sys.stderr)

# <<
if 'clip' not in args.model:
    tfms  = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model)
    assert hasattr(model, 'classifier')
    model.classifier = nn.Sequential()
else:
    tfms  = CLIPFeatureExtractor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).vision_model
# >>

model = model.eval()
model = model.to(device)

# !! TODO: Getting the "features" from different models might require dropping different heads w/ different names

# --
# Load dataset

print('hf_featurize: loading dataset....', file=sys.stderr)
hf_dataset = load_dataset(args.dataset)
hf_dataset = hf_dataset[args.split]
hf_dataset = hf_dataset.rename_columns({args.img_field:"img", args.lab_field:"lab"})

ds_train   = HFDataset(hf_dataset, transform=tfms)
dl_train   = DataLoader(ds_train, batch_size=128, num_workers=8)

# --
# Run

print('hf_featurize: extracting features...', file=sys.stderr)

X = []
y = []

with torch.inference_mode():
    for xx, yy in tqdm(dl_train, total=len(dl_train)):
        xx  = xx.to(device)
        
        # <<
        if 'clip' not in args.model:
            enc = model(xx).logits
        else:
            enc = model(xx).pooler_output
        # >>
        
        enc = enc.cpu().numpy() # not logits ...
        
        X.append(enc.squeeze())
        y.append(yy)

X = np.row_stack(X)
y = np.hstack(y)

# --
# Save

np.save(os.path.join(args.outdir, 'X.npy'), X)
np.save(os.path.join(args.outdir, 'y.npy'), y)