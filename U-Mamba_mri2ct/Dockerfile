# --------------------------------------------------
# Base image: Ubuntu 22.04 + CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# =========================
# Environment variables
# =========================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV nnUNet_preprocessed="/workspace/U-Mamba/data/nnUNet_preprocessed"
ENV nnUNet_raw="/workspace/U-Mamba/data/nnUNet_raw"
ENV nnUNet_results="/workspace/U-Mamba/data/nnUNet_results"

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install setuptools wheel packaging numpy scipy nibabel simpleitk

# PyTorch 2.0.1 (CUDA 11.8)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install torchinfo torchsummary tensorboard    
RUN pip install dynamic_network_architectures    

# Install Causal-conv1d & Mamba
RUN pip install --no-cache-dir causal-conv1d>=1.4.0 --no-build-isolation
RUN pip install --no-cache-dir mamba-ssm --no-build-isolation

# Copy local U-Mamba source into image
WORKDIR /workspace/U-Mamba
COPY . .

# Install U-Mamba 
WORKDIR /workspace/U-Mamba/umamba
RUN pip install -e .

ENV PYTHONPATH="/workspace/U-Mamba:/workspace/U-Mamba/umamba"

# Default working directory
WORKDIR /workspace/U-Mamba
