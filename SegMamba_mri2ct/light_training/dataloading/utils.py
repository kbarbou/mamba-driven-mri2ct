import numpy as np 
import os 
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
import multiprocessing

def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    # try:
    a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
    if not isfile(npz_file[:-3] + "npy"):
        np.save(npz_file[:-3] + "npy", a['data'])
        print(f"Created and saved .npy data file for {npz_file[:-4]}")
    elif overwrite_existing:
        np.save(npz_file[:-3] + "npy", a['data'])
        print(f"Overwrote existing .npy data file for {npz_file[:-4]}")
    else:
        print(f".npy data file for {npz_file[:-4]} already exists. Skipping. Set overwrite_existing=True to overwrite.")

    if unpack_segmentation:
        if not isfile(npz_file[:-4] + "_seg.npy"):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
            print(f"Created and saved .npy seg file for {npz_file[:-4]}")
        elif overwrite_existing:
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
            print(f"Overwrote existing .npy seg file for {npz_file[:-4]}")
        else:
            print(f".npy seg file for {npz_file[:-4]} already exists. Skipping. Set overwrite_existing=True to overwrite.")
    else:        
        print(f"Skipping segmentation unpacking for {npz_file[:-4]}. Set unpack_segmentation=True to unpack.")

    if 'mask' in a.files:
        mask_npy = npz_file[:-4] + "_mask.npy"

        if not isfile(mask_npy):
            np.save(mask_npy, a['mask'])
            print(f"Created and saved .npy mask file for {npz_file[:-4]}")
        elif overwrite_existing:
            np.save(mask_npy, a['mask'])
            print(f"Overwrote existing .npy mask file for {npz_file[:-4]}")
        else:
            print(f".npy mask file for {npz_file[:-4]} already exists. Skipping. Set overwrite_existing=True to overwrite.")
    else:
        print(f"No mask found in {npz_file[:-4]}. Skipping mask unpacking.")
            
def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = 8):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )
