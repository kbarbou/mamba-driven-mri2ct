## Manual Setup for TotalSegmentator Weights

Install **TotalSegmentator** and download pretrained weights:
```bash
pip install totalsegmentator
totalseg_download_weights -t total
```

To store the weights in a custom location, set `TOTALSEG_HOME_DIR` before running `totalseg_download_weights` command:
```bash
export TOTALSEG_HOME_DIR=/new/path/.totalsegmentator
```
Navigate to `TotalSegmentator_Dataset297_Pretrained` and copy the checkpoint:
```bash
cd TotalSegmentator_Dataset297_Pretrained
cp ~/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth .
```
