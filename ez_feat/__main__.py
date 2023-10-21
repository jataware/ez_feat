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
from glob import glob
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    CLIPFeatureExtractor,
    CLIPModel
)

from datasets import load_dataset

# --
# Helpers

def get_device(force=None):
    if force is not None: return force

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def img_safeload(fname, dim=32):
    try:
        img = Image.open(fname).convert('RGB')
        bad = False
    except:
        img = Image.new('RGB', (dim, dim))
        bad = True

    if (img.size[0] == 1) or (img.size[1] == 1):
        img = Image.new('RGB', (dim, dim))
        bad = True
    
    return img, bad

# --
# Datasets

class HFDataset(Dataset):
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

class EZDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames    = fnames
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname    = self.fnames[idx]
        img, bad = img_safeload(fname)

        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'][0]
        
        return img, [bad]


def hf_load_model(model, with_text_encoder=False):
    print('hf_load_model: start', file=sys.stderr)

    # <<
    # !! TODO: Getting the "features" from different models might require dropping different heads w/ different names
    if 'clip' not in model:
        tfms  = AutoFeatureExtractor.from_pretrained(model)
        model = AutoModelForImageClassification.from_pretrained(model)
        assert hasattr(model, 'classifier')
        model.classifier = nn.Sequential()
    else:
        tfms  = CLIPFeatureExtractor.from_pretrained(model)
        model = CLIPModel.from_pretrained(model).vision_model
    # >>

    model = model.eval()
    return model, tfms


def hf_load_dataset(dataset, split, img_field, lab_field):
    print('hf_load_dataset: start', file=sys.stderr)
    hf_dataset = load_dataset(dataset)
    hf_dataset = hf_dataset[split]
    hf_dataset = hf_dataset.rename_columns({img_field:"img", lab_field:"lab"})
    return hf_dataset


def featurize(model, model_name, ds, progress_bar=tqdm, force_device=None):
    print('featurize: start', file=sys.stderr)

    device = get_device(force=force_device)
    model  = model.to(device)
    
    dl     = DataLoader(ds, batch_size=128)

    X = [] # features
    M = [] # metadata

    with torch.inference_mode():
        for xx, *mm in progress_bar(dl, total=len(dl)):
            xx  = xx.to(device)
            
            # <<
            # !! TODO: Getting the "features" from different models might require dropping different heads w/ different names
            if 'clip' not in model_name:
                enc = model(xx).logits
            else:
                enc = model(xx).pooler_output
            # >>
            
            enc = enc.cpu().numpy()
            
            X.append(enc.squeeze())
            M += mm

    X = np.row_stack(X)
    M = np.column_stack(M)

    return X, M


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, default="microsoft/resnet-50")

    parser.add_argument('--dataset',   type=str, default="pcuenq/oxford-pets")
    parser.add_argument('--split',     type=str, default="train")
    parser.add_argument('--img_field', type=str, default="image")
    parser.add_argument('--lab_field', type=str, default="label")

    parser.add_argument('--outdir',     type=str, default="output")
    parser.add_argument('--datadir',    type=str, default="data")

    args = parser.parse_args()
    
    args.outdir  = os.path.join(args.outdir, args.dataset, args.split, args.model)
    args.datadir = os.path.join(args.datadir, args.dataset, args.split)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)

    return args


if __name__ == '__main__':
    args = parse_args()

    model, tfms = hf_load_model(args.model)
    ds          = hf_load_dataset(args.dataset, args.split, args.img_field, args.lab_field)
    
    # # featurize
    # print('ez_feat: featurizing')
    # X, M = featurize(model, args.model, ds=HFDataset(ds, transform=tfms))
    # y = np.row_stack(M)
    # np.save(os.path.join(args.outdir, 'X.npy'), X)
    # np.save(os.path.join(args.outdir, 'y.npy'), y)

    # save dataset
    print('ez_feat: writing images')
    def _save_img(x):
        dst = os.path.join(args.datadir, f"{x['id']:06d}.png")
        x['img'].save(dst)

    ds.map(_save_img, num_proc=8)
