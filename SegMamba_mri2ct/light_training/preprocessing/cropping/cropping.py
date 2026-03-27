import numpy as np

from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice

def crop_to_nonzero(data, mask=None, mask_value=0.0, apply_cropping=False):
    """
    Crop data to non-zero regions defined by a mask (outline mask).
    If no mask is provided, returns data as-is.

    Parameters
    ----------
    data : np.ndarray
        Shape (C, X, Y, Z) or (C, X, Y)
    mask : np.ndarray, optional
        Binary mask of shape (X, Y, Z) or (X, Y). True = region to keep.
    mask_value : float, default 0.0
        Value to fill outside the mask (optional, applied after crop if needed).

    Returns
    -------
    data : np.ndarray
        Cropped data, shape (C, cropped_X, cropped_Y, cropped_Z) or (C, cropped_X, cropped_Y)
    bbox : tuple
        Bounding box used for cropping.
    """
    if (mask is None) or (apply_cropping == False):
        # Full bbox = entire data shape
        spatial_shape = data.shape[1:]
        bbox = tuple((0, dim) for dim in spatial_shape)
        return data, None

    # Validate shapes
    assert data.shape[1:] == mask.shape, f"Mask shape {mask.shape} must match data spatial shape {data.shape[1:]}"
    
    # Get bounding box from mask
    bbox = get_bbox_from_mask(mask)
    slicer = bounding_box_to_slice(bbox)
    
    # Crop data to bbox
    cropped_data = data[tuple([slice(None), *slicer])]
    
    # Optional: mask outside the outline inside cropped region (if mask_value != 0)
    if mask_value != 0:
        cropped_mask = mask[slicer]
        for c in range(cropped_data.shape[0]):
            cropped_data[c][~cropped_mask] = mask_value

    return cropped_data, bbox
