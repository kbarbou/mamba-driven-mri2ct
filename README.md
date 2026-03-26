# Mamba-driven MRI-to-CT Synthesis for MRI-only Radiotherapy Planning

The goal of this work is to investigate MRI-to-CT image-to-image translation using multiple deep learning models, with a focus on Mamba-based architectures.

We provide a unified framework for training and evaluating:

- U-Mamba
- SegMamba
- nnU-Net (adapted for translation)
- U-Net
- SwinUNETR

### Environments

This repository uses **multiple environments** due to different dependencies across models:

- `requirements.txt`  
  → Used for preprocessing and evaluation

- `nnUNet_translation/`  
  → Requires a dedicated Python environment (see folder README.md)

- `U-Mamba_translation/`  
  → Uses Docker (see folder README.md)

- `SegMamba_translation/`  
  → Uses Docker (see folder README.md)

---

## Data Preparation

The data used in this study are derived from the training set of **Task 1** of the [SynthRAD2025](https://synthrad2025.grand-challenge.org/) challenge and can be downloaded from [Zenodo](https://zenodo.org/records/15373853).

### Environment Setup

First, create the required conda environment and install additional dependencies:

```bash
conda env create -f mri2ct.yml
conda activate mri2ct
pip install -r requirements.txt --no-cache-dir
```

After setting up the environment, run the following commands:
```bash
cd mamba-driven-mri2ct/
python data_preparation.py
python convert_mha_to_nifti.py
```

### Acknowledgements
This work builds upon several open-source projects. We express our appreciation to the authors of the following repositories:

- SegMamba — https://github.com/ge-xing/SegMamba
- U-Mamba — https://github.com/bowang-lab/U-Mamba
- nnU-Net_translation — https://github.com/Phyrise/nnUNet_translation
- SynthRAD2025 Evaluation metrics — https://github.com/SynthRAD2025/metrics

We also thank the organizers of the [SynthRAD2025](https://synthrad2025.grand-challenge.org/) for making their dataset available to the research community. In addition, we would like to thank the developers of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), and [MONAI](https://github.com/Project-MONAI/MONAI), which were essential for this work.

The authors gratefully acknowledge NVIDIA Corporation for the GPU hardware grant that facilitated the conducted computational experiments.
