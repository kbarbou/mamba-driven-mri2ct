#!/usr/bin/env python3
"""Post-process nnUNet inference volumes.

For each .nii.gz file in the input directory:
- If the case ID appears in verified_matches.txt, copy the file unchanged.
- Otherwise, multiply the image data by 1000 and clip to [-1024, 3000].

The processed files are written to the output directory while preserving the
original affine and header metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process nnUNet inference volumes")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/path/to/output_data"),
        help="Directory containing nnUNet inference volumes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/postprocessed_output_data"),
        help="Directory where post-processed volumes will be written",
    )
    parser.add_argument(
        "--verified-list",
        type=Path,
        #if needed
        default=Path("path/to/already_postprocessed_cases/verified_matches.txt"),
        help="Text file listing case IDs that should be copied unchanged from the current output dir to the postprocessed one",
    )
    return parser.parse_args()


def load_verified_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    cases: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.add(Path(line).stem)
    return cases


def iter_volume_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(
        p for p in input_dir.glob("*.nii.gz") if p.is_file() and p.parent == input_dir
    )


def process_volume(input_path: Path, output_dir: Path, verified_cases: set[str]) -> None:
    case_id = input_path.stem.replace(".nii", "")
    output_path = output_dir / input_path.name

    if case_id in verified_cases:
        output_path.write_bytes(input_path.read_bytes())
        print(f"Copied unchanged: {case_id}")
        return

    img = nib.load(str(input_path))
    data = img.get_fdata(dtype=np.float32)
    processed = np.clip(data * 1000.0, -1024.0, 3000.0)

    new_img = nib.Nifti1Image(processed, img.affine, header=img.header)
    nib.save(new_img, str(output_path))
    print(f"Processed: {case_id}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    verified_list = args.verified_list.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    verified_cases = load_verified_cases(verified_list)

    volume_files = list(iter_volume_files(input_dir))
    if not volume_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    for input_path in volume_files:
        process_volume(input_path, output_dir, verified_cases)

    print(f"Completed processing {len(volume_files)} files into {output_dir}")


if __name__ == "__main__":
    main()
