#!/bin/bash
echo "Installing..."
source ~/anaconda3/etc/profile.d/conda.sh && conda init && conda create -n quetzal_env -y python=3.8 && conda activate quetzal_env && pip install -e .
python -m ipykernel install --user --name=quetzal_env
