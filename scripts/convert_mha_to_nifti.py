"""
Reads the .mha path lists produced by data_preparation.py and
converts every file to .nii.gz, writing the new paths to a matching
*_nifti.txt file in the same Paths_and_Sizes directory.
"""

import os
import SimpleITK as sitk

# Ensure the script runs on CPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ANATOMICAL_REGIONS = ["AB", "HN", "TH"]
MODALITIES          = ["mr", "ct", "mask"]
PATHS_DIR           = os.path.expanduser("~/Paths_and_Sizes")
# PATHS_DIR = /set/path/to/Paths_and_Sizes


def read_paths(txt_file: str) -> list[str]:
    with open(txt_file) as f:
        return [line.strip() for line in f if line.strip()]


def convert_and_save_nifti(mha_paths: list[str], output_txt: str):
    nii_paths = []
    for mha_path in mha_paths:
        mha_path = os.path.expanduser(mha_path)
        if not mha_path.endswith(".mha"):
            continue
        image    = sitk.ReadImage(mha_path)
        base     = os.path.splitext(os.path.basename(mha_path))[0]
        nii_path = os.path.join(os.path.dirname(mha_path), base + ".nii.gz")
        sitk.WriteImage(image, nii_path)
        nii_paths.append(nii_path)
        print(f"Converted: {mha_path} → {nii_path}")

    with open(output_txt, "w") as f:
        f.write("\n".join(nii_paths) + "\n")
    print(f"Saved NIfTI paths to: {output_txt}")


if __name__ == "__main__":
    for region in ANATOMICAL_REGIONS:
        for mod in MODALITIES:
            input_txt  = os.path.join(PATHS_DIR, f"{region}_{mod}_paths.txt")
            output_txt = os.path.join(PATHS_DIR, f"{region}_{mod}_paths_nifti.txt")
            mha_list   = read_paths(input_txt)
            convert_and_save_nifti(mha_list, output_txt)
