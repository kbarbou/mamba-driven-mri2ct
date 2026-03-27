"""
Runs the SegMamba/light_training preprocessor on each split (train / test):
  1. plan()         — computes intensity statistics and data properties
  2. process_train() — resamples and normalises the data
"""

import os
import argparse
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor

# ── USER CONFIGURATION (overridable via CLI arguments) ────────────────────────
DEFAULT_BASE_DIR   = os.path.expanduser("~/SegMamba_translation/raw_data/synthRAD2025")
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/SegMamba_translation/preprocessed/fullres")
DEFAULT_SPACING    = [3.0, 1.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────


def plan(mode, base_dir, data_filename, seg_filename, mask_filename=None):
    preprocessor = MultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=mode,
        data_filenames=data_filename,
        seg_filename=seg_filename,
        mask_filename=mask_filename,
    )
    preprocessor.run_plan()


def process_train(mode, base_dir, data_filename, seg_filename,
                  out_spacing, output_dir, mask_filename=None):
    preprocessor = MultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=mode,
        data_filenames=data_filename,
        seg_filename=seg_filename,
        mask_filename=mask_filename,
    )
    preprocessor.run(
        output_spacing=out_spacing,
        output_dir=os.path.join(output_dir, mode),
        all_labels=[1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SegMamba dataset.")
    parser.add_argument("--base_dir",   default=DEFAULT_BASE_DIR,
                        help="Root directory of the raw SegMamba dataset (output of 1_prepare_raw_data.py)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root directory for preprocessed output")
    parser.add_argument("--spacing",    default=DEFAULT_SPACING, nargs=3, type=float,
                        metavar=("Z", "Y", "X"),
                        help="Target voxel spacing after resampling (default: 3.0 1.0 1.0)")
    args = parser.parse_args()

    data_filename = ["mri.nii.gz"]
    seg_filename  = "ct.nii.gz"
    mask_filename = "mask.nii.gz"

    os.makedirs(args.output_dir, exist_ok=True)

    for mode in ["train", "test"]:
        # Running plan first calculates intensity stats and properties of your MRI/CT pairs
        plan(mode, args.base_dir, data_filename, seg_filename, mask_filename)
        # Running process_train resamples and normalises the data
        process_train(mode, args.base_dir, data_filename, seg_filename,
                      args.spacing, args.output_dir, mask_filename)
