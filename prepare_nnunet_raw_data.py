import os
import json
import shutil

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
PATHS_DIR  = os.path.expanduser("~/Paths_and_Sizes")
# PATHS_DIR = /set/path/to/Paths_and_Sizes
NNUNET_RAW = os.path.expanduser("~/nnUNet_raw/Dataset100_sCTgeneration")
# NNUNET_RAW = /set/path/to/nnUNet_raw/Dataset100_sCTgeneration

ANATOMICAL_REGIONS = ["AB", "HN", "TH"]
TRAIN_RATIO        = 0.9
# ─────────────────────────────────────────────────────────────────────────────

imagesTr = os.path.join(NNUNET_RAW, "imagesTr")
labelsTr = os.path.join(NNUNET_RAW, "labelsTr")
imagesTs = os.path.join(NNUNET_RAW, "imagesTs")
labelsTs = os.path.join(NNUNET_RAW, "labelsTs")

for d in (imagesTr, labelsTr, imagesTs, labelsTs):
    os.makedirs(d, exist_ok=True)


def extract_patient_id(ct_path: str) -> str:
    """Extracts patient ID from ct_<ID>.nii.gz."""
    return os.path.basename(ct_path).replace("ct_", "").replace(".nii.gz", "")


def ct_to_mr_path(ct_path: str) -> str:
    """Converts .../ct_<ID>.nii.gz → .../mr_<ID>.nii.gz"""
    dirname  = os.path.dirname(ct_path)
    basename = os.path.basename(ct_path).replace("ct_", "mr_", 1)
    return os.path.join(dirname, basename)


def load_ct_paths(region: str) -> list[str]:
    txt = os.path.join(PATHS_DIR, f"{region}_ct_paths_nifti.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    mapping = {}

    for region in ANATOMICAL_REGIONS:
        ct_paths  = load_ct_paths(region)
        split_idx = int(TRAIN_RATIO * len(ct_paths))

        splits = [("train", ct_paths[:split_idx]), ("test", ct_paths[split_idx:])]

        for mode, group in splits:
            for ct_path in group:
                patient_id = extract_patient_id(ct_path)
                mr_path    = ct_to_mr_path(ct_path)

                if not os.path.exists(mr_path):
                    print(f"WARNING: MR not found for {ct_path}")
                    continue

                dest_mr = os.path.join(imagesTr if mode == "train" else imagesTs,
                                       f"{patient_id}_0000.nii.gz")
                dest_ct = os.path.join(labelsTr  if mode == "train" else labelsTs,
                                       f"{patient_id}.nii.gz")

                shutil.copy(mr_path, dest_mr)
                shutil.copy(ct_path, dest_ct)

                mapping[patient_id] = {
                    "ct_source":  ct_path,
                    "mr_source":  mr_path,
                    "nnunet_ct":  dest_ct,
                    "nnunet_mr":  dest_mr,
                    "set":        mode,
                }

    mapping_path = os.path.join(NNUNET_RAW, "patient_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)

    train_n = sum(1 for v in mapping.values() if v["set"] == "train")
    test_n  = sum(1 for v in mapping.values() if v["set"] == "test")
    print(f"Training cases : {train_n}")
    print(f"Test cases     : {test_n}")
    print(f"Mapping saved  : {mapping_path}")

    # ── Build dataset.json ────────────────────────────────────────────────────
    training_list = []
    test_list     = []

    for patient_id, info in mapping.items():
        mr_file = os.path.basename(info["nnunet_mr"])
        ct_file = os.path.basename(info["nnunet_ct"])
        if info["set"] == "train":
            training_list.append({
                "image": f"./imagesTr/{mr_file}",
                "label": f"./labelsTr/{ct_file}",
            })
        else:
            test_list.append(f"./imagesTs/{mr_file}")

    dataset_json = {
        "name":                          "sCTgeneration",
        "channel_names":                 {"0": "MRI"},
        "labels":                        {"background": 0},
        "numTraining":                   len(training_list),
        "training":                      training_list,
        "test":                          test_list,
        "file_ending":                   ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    dataset_path = os.path.join(NNUNET_RAW, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"dataset.json   : {dataset_path}")
