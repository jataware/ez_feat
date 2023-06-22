#!/bin/bash

# run.sh

# --
# Create env

conda create -y -n ez_feat_env python=3.10.10
conda activate ez_feat_env

# --
# Install dependencies

pip install Pillow
pip install transformers
pip install datasets

# pytorch version?