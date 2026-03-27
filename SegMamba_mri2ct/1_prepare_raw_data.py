"""
Reads per-anatomy CT NIfTI path lists, performs a per-region 90/10 train/test
split, copies MR, CT and mask files into the SegMamba raw data directory
structure, and writes the dataset.json expected by SegMamba/MONAI.
"""

import os
import json
import shutil
import argparse

# ── USER CONFIGURATION (overridable via CLI arguments) ────────────────────────
DEFAULT_PATHS_DIR = os.path.expanduser("~/Paths_and_Sizes")
DEFAULT_OUT_BASE  = os.path.expanduser("~/SegMamba_translation/raw_data/synthRAD2025")

ANATOMICAL_REGIONS = ["AB", "HN", "TH"]
TRAIN_RATIO        = 0.9
# ─────────────────────────────────────────────────────────────────────────────


def extract_patient_id(ct_path: str) -> str:
    return os.path.basename(ct_path).replace("ct_", "").replace(".nii.gz", "")


def ct_to_mr_path(ct_path: str) -> str:
    dirname  = os.path.dirname(ct_path)
    basename = os.path.basename(ct_path).replace("ct_", "mr_", 1)
    return os.path.join(dirname, basename)


def ct_to_mask_path(ct_path: str, patient_id: str) -> str:
    return os.path.join(os.path.dirname(ct_path), f"mask_{patient_id}.nii.gz")


def load_ct_paths(region: str, paths_dir: str) -> list[str]:
    txt = os.path.join(paths_dir, f"{region}_ct_paths_nifti.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SegMamba raw data directory.")
    parser.add_argument("--paths_dir", default=DEFAULT_PATHS_DIR,
                        help="Directory containing per-anatomy *_ct_paths_nifti.txt files")
    parser.add_argument("--out_base",  default=DEFAULT_OUT_BASE,
                        help="Root output directory for SegMamba raw data")
    args = parser.parse_args()

    PATHS_DIR = args.paths_dir
    OUT_BASE  = args.out_base

    os.makedirs(OUT_BASE, exist_ok=True)

    training_data   = []
    validation_data = []

    for region in ANATOMICAL_REGIONS:
        ct_paths  = load_ct_paths(region, PATHS_DIR)
        split_idx = int(TRAIN_RATIO * len(ct_paths))

        splits = [("train", ct_paths[:split_idx]), ("test", ct_paths[split_idx:])]

        for mode, group in splits:
            for ct_path in group:
                patient_id = extract_patient_id(ct_path)
                mr_path    = ct_to_mr_path(ct_path)
                mask_path  = ct_to_mask_path(ct_path, patient_id)

                if not os.path.exists(mr_path):
                    print(f"WARNING: MR not found for {ct_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"WARNING: mask not found for {ct_path}")
                    continue

                patient_dir = os.path.join(OUT_BASE, mode, patient_id)
                os.makedirs(patient_dir, exist_ok=True)

                shutil.copy(mr_path,   os.path.join(patient_dir, "mri.nii.gz"))
                shutil.copy(ct_path,   os.path.join(patient_dir, "ct.nii.gz"))
                shutil.copy(mask_path, os.path.join(patient_dir, "mask.nii.gz"))

                entry = {
                    "image": os.path.relpath(os.path.join(patient_dir, "mri.nii.gz"),  OUT_BASE),
                    "label": os.path.relpath(os.path.join(patient_dir, "ct.nii.gz"),   OUT_BASE),
                    "mask":  os.path.relpath(os.path.join(patient_dir, "mask.nii.gz"), OUT_BASE),
                }

                if mode == "train":
                    training_data.append(entry)
                else:
                    validation_data.append(entry)

    dataset_json = {
        "name":        "sCT_Generation",
        "description": "MRI to CT Synthesis",
        "modality":    {"0": "MRI"},
        "labels":      {"0": "CT_HU"},
        "numTraining": len(training_data),
        "numTest":     len(validation_data),
        "training":    training_data,
        "validation":  validation_data,
    }

    dataset_json_path = os.path.join(OUT_BASE, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Training cases : {len(training_data)}")
    print(f"Test cases     : {len(validation_data)}")
    print(f"dataset.json   : {dataset_json_path}")
