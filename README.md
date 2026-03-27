# Mamba-driven MRI-to-CT Synthesis for MRI-only Radiotherapy Planning

The goal of this work is to investigate MRI-to-CT image-to-image translation using multiple deep learning models, with a focus on Mamba-based architectures.

We provide a unified framework for training and evaluating several models that have been adapted for MRI-to-CT translation, including:

- U-Mamba
- SegMamba
- nnU-Net
- U-Net
- SwinUNETR


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

### SegMamba

#### Docker Setup
We recommend running training and inference for the SegMamba framework inside dedicated Docker container.
```bash
cd SegMamba_mri2ct/
docker build -t segmamba:11.8.0-base-ubuntu22.04 .
```

Verify the docker installation
```bash
docker run --rm --gpus '"device=0"' segmamba:11.8.0-base-ubuntu22.04 python 0_dummy_inference.py
```
#### Preprocessing
The following preprocessing steps can be executed in the mri2ct environment:
```bash
python 1_prepare_raw_data.py --paths_dir /path/to/Paths_and_Sizes --out_base /path/to/out_base
python 2_preprocessing.py --base_dir   /path/to/raw_data --output_dir /path/to/preprocessed/data
```
#### Training


#### Infenrence


### U-Mamba
```bash
cd U-Mamba_mri2ct/
docker build -t umamba:11.8.0-base-ubuntu22.04 .
```

### nnUNet
```bash
cd nnUNet_mri2ct/
conda 
pip install -e .
export nnUNet_raw="set/path/to/data/nnUNet/raw"
export nnUNet_preprocessed="set/path/to/data/nnUNet/preprocessed"
export nnUNet_results="set/path/to/data/nnUNet/results"
```

#### Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d 100 -c 3d_fullres -pl nnUNetPlannerResUNet
```
#### Training
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DatasetY 3d_fullres 0 -tr nnUNetTrainerMRCT_compound_loss -pl nnResUNetPlans [optional: -pretrained_weights PATH_TO_CHECKPOINT]
```
#### Infenrence
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d 100 -i INPUT -o OUTPUT -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT_compound_loss -f FOLD [optional : -chk checkpoint_best.pth -step_size 0.3 --rec (mean,median)]
```

### Evaluation
Once training and inference are completed, set up the paths for ground truth and synthetic data and run the following scripts:
```bash
python compute_image_similarity_metrics.py
python compute_segmentation_metrics.py
```

## Acknowledgements
This work builds upon several open-source projects. We express our appreciation to the authors of the following repositories:

- SegMamba — https://github.com/ge-xing/SegMamba
- U-Mamba — https://github.com/bowang-lab/U-Mamba
- nnU-Net_translation — https://github.com/Phyrise/nnUNet_translation
- SynthRAD2025 Evaluation metrics — https://github.com/SynthRAD2025/metrics

We also thank the organizers of the [SynthRAD2025](https://synthrad2025.grand-challenge.org/) for making their dataset available to the research community. In addition, we would like to thank the developers of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), and [MONAI](https://github.com/Project-MONAI/MONAI), which were essential for this work.

The authors gratefully acknowledge NVIDIA Corporation for the GPU hardware grant that facilitated the conducted computational experiments.
