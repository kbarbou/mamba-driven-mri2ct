# =========================
# Base image (CUDA + Ubuntu)
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# =========================
# Environment variables
# =========================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3.10 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel packaging numpy scipy scikit-learn einops ninja batchgenerators acvl_utils nibabel simpleitk
RUN pip install transformers==4.38.2

# =========================
# Install PyTorch (CUDA 11.8)
# =========================
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install torchinfo torchsummary tensorboard
RUN pip install dynamic_network_architectures

# =========================
# Install MONAI (SegMamba dependency)
# =========================
RUN pip install monai

# =========================
# Copy SegMamba source code
# =========================
WORKDIR /opt/SegMamba
COPY . /opt/SegMamba

# =========================
# Build CUDA extensions ONCE
# =========================
WORKDIR /opt/SegMamba/causal-conv1d
RUN python setup.py install

WORKDIR /opt/SegMamba/mamba
RUN python setup.py install

# =========================
# Default working directory
# =========================
WORKDIR /opt/SegMamba
