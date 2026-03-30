"""
Renames each patient's mr.mha / ct.mha / mask.mha to
<modality>_<patientID>.mha (e.g. mr_1ABA031.mha), then writes:
  - Paths_and_Sizes/<REGION>_<modality>_paths.txt  (e.g. AB_mr_paths.txt)
  - Paths_and_Sizes/image_metadata.json            (patientID → spacing & size)
"""

import os
import json
import SimpleITK as sitk

# Ensure the script runs on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ANATOMICAL_REGIONS = ["AB", "HN", "TH"]
MODALITIES          = ["mr", "ct", "mask"]
OUTPUT_DIR          = os.path.expanduser("~/Paths_and_Sizes")
# PATHS_DIR = /set/path/to/Paths_and_Sizes


def load_and_rename_files(base_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # paths[region][modality] = list of renamed file paths
    paths    = {r: {m: [] for m in MODALITIES} for r in ANATOMICAL_REGIONS}
    metadata = {}   # patientID → {"spacing": [...], "size": [...]}

    for region in ANATOMICAL_REGIONS:
        region_path = os.path.join(base_path, region)
        if not os.path.exists(region_path):
            print(f"Skipping missing region: {region}")
            continue

        for folder in sorted(os.listdir(region_path)):
            folder_path = os.path.join(region_path, folder)
            if not os.path.isdir(folder_path) or folder.lower() == "overviews":
                print("Skipping instance", folder_path)
                continue

            patient_id = folder

            for mod in MODALITIES:
                src = os.path.join(folder_path, f"{mod}.mha")
                dst = os.path.join(folder_path, f"{mod}_{patient_id}.mha")

                if not os.path.exists(src):
                    print(f"Missing: {src}")
                    continue

                # Collect spacing / size once per patient from the MR scan
                if mod == "mr":
                    img = sitk.ReadImage(src)
                    metadata[patient_id] = {
                        "spacing": list(img.GetSpacing()),
                        "size":    list(img.GetSize()),
                    }

                os.rename(src, dst)
                print(f"Renamed: {src} → {dst}")
                paths[region][mod].append(dst)

    # Write per-anatomy, per-modality .txt files
    for region in ANATOMICAL_REGIONS:
        for mod in MODALITIES:
            out_txt = os.path.join(OUTPUT_DIR, f"{region}_{mod}_paths.txt")
            with open(out_txt, "w") as f:
                f.write("\n".join(paths[region][mod]) + "\n")
            print(f"Saved: {out_txt}")

    # Write metadata JSON  {patientID: {spacing, size}}
    meta_path = os.path.join(OUTPUT_DIR, "image_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    base_path = os.path.expanduser("~/synthRAD2025_Task1_Train/Task1")
    # base_path = /set/path/to/synthRAD2025_Task1_Train/Task1
    load_and_rename_files(base_path)
