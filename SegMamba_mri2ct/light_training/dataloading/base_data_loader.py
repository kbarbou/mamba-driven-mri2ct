import numpy as np 
from typing import Union, Tuple
import time 

class DataLoaderMultiProcess:
    def __init__(self, dataset,
                 patch_size,
                 batch_size=2,
                 num_patches_per_batch=1,
                 oversample_foreground_percent=0.33,
                 probabilistic_oversampling=False,
                 print_time=False):
        pass
        self.dataset = dataset
        self.patch_size = patch_size
        # self.annotated_classes_key = annotated_classes_key ## (1, 2, 3 ..)
        self.batch_size = batch_size
        self.num_patches_per_batch = num_patches_per_batch
        self.keys = [i for i in range(len(dataset))]
        self.thread_id = 0
        self.oversample_foreground_percent = oversample_foreground_percent
        self.need_to_pad = (np.array([0, 0, 0])).astype(int)

        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.data_shape = None
        self.seg_shape = None
        self.print_time = print_time

        self.min_body_coverage = 0.7   # based on the outline mask (if any)
        self.max_tries = 20            # Number of attemps to enusre body coverage is retained

    def determine_shapes(self, num_patches_per_batch=1):
        # load one case
        item = self.dataset.__getitem__(0)
        data, seg, properties = item["data"], item["seg"], item["properties"]
        num_color_channels = data.shape[0]
        num_output_channels = seg.shape[0]
        patch_size = self.patch_size
        data_shape = (self.batch_size * num_patches_per_batch, num_color_channels, patch_size[0], patch_size[1], patch_size[2])
        seg_shape = (self.batch_size * num_patches_per_batch, num_output_channels, patch_size[0], patch_size[1], patch_size[2])
        return data_shape, seg_shape
    
    def generate_train_batch(self):
        selected_keys = np.random.choice(self.keys, self.batch_size, True, None)
        if self.data_shape is None:
            self.data_shape, self.seg_shape = self.determine_shapes(self.num_patches_per_batch)

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        data_all_global = np.zeros(self.data_shape, dtype=np.float32)
        seg_all_global = np.zeros(self.seg_shape, dtype=np.float32)
        data_global = None
        seg_global = None
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []

        for j, key in enumerate(selected_keys):
            s = time.time()
            item = self.dataset.__getitem__(key)
            e = time.time()
            if self.print_time:
                print(f"read single data time is {e - s}")

            data, seg, properties = item["data"], item["seg"], item["properties"]
            mask = item.get("mask", None)

            if "data_global" in item:
                data_global = item["data_global"]

            if "seg_global" in item:
                seg_global = item["seg_global"]

            shape = data.shape[1:]
            dim = len(shape)

            #min_data_value = np.min(data)
            #min_seg_value = np.min(seg)

            # Extract num_patches_per_batch patches from the same loaded case
            for p in range(self.num_patches_per_batch):
                idx = j * self.num_patches_per_batch + p
                force_fg = self.get_do_oversample(idx)

                patch_accepted = False  # flag to track if coverage threshold met

                for attempt in range(self.max_tries):
                    s = time.time()
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
                    e = time.time()
                    if self.print_time:
                        print(f"Attempt {attempt}: get bbox time is {e - s}")
                        
                    valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                    valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

                    if mask is not None:
                        mask_slice = tuple([slice(0, mask.shape[0])] + [slice(lo, hi) for lo, hi in zip(valid_bbox_lbs, valid_bbox_ubs)])
                        mask_patch = mask[mask_slice]

                        coverage = mask_patch.mean()  # fraction of foreground voxels
                        if coverage >= self.min_body_coverage:
                            patch_accepted = True
                            break  # coverage threshold met
                    else:
                        patch_accepted = True
                        break  # no mask → accept immediately

                if not patch_accepted and mask is not None:
                    print(f"[WARNING] Patch {idx} for patient {properties['name']} "
                        f"has body coverage {coverage:.3f} < requested {self.min_body_coverage:.3f}. "
                        f"Max tries ({self.max_tries}) exhausted.")

                data_slice = tuple([slice(0, data.shape[0])] + [slice(lo, hi) for lo, hi in zip(valid_bbox_lbs, valid_bbox_ubs)])
                data_patch = data[data_slice]

                seg_slice = tuple([slice(0, seg.shape[0])] + [slice(lo, hi) for lo, hi in zip(valid_bbox_lbs, valid_bbox_ubs)])
                seg_patch = seg[seg_slice]

                s = time.time()
                padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                data_all[idx] = np.pad(data_patch, ((0, 0), *padding), 'constant', constant_values=0)
                #seg_all[idx] = np.pad(seg_patch, ((0, 0), *padding), 'constant', constant_values=0)
                #data_all[idx] = np.pad(data_patch, ((0, 0), *padding), 'constant', constant_values=min_data_value)
                seg_all[idx] = np.pad(seg_patch, ((0, 0), *padding), 'constant', constant_values=-2.0)

                if data_global is not None:
                    data_all_global[idx] = data_global

                if seg_global is not None:
                    seg_all_global[idx] = seg_global

                e = time.time()
                if self.print_time:
                    print(f"box is {bbox_lbs, bbox_ubs}, padding is {padding}")
                    print(f"setting data value time is {e - s}")

                case_properties.append(properties)

        if data_global is None:
            return {'data': data_all,
                    'seg': seg_all, 'properties': case_properties,
                    'keys': selected_keys}

        return {'data': data_all, "data_global": data_all_global,
                    "seg_global": seg_all_global,
                    'seg': seg_all, 'properties': case_properties,
                    'keys': selected_keys}

    def __next__(self):
    
        return self.generate_train_batch() 
    
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id
    
    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        total_patches = self.batch_size * self.num_patches_per_batch
        return not sample_idx < round(total_patches * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                    'have class_locations (missing key)'
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            # class_locations keys can also be tuple
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

            # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
            # strange formulation needed to circumvent
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            # tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
            # if any(tmp):
            #     if len(eligible_classes_or_regions) > 1:
            #         eligible_classes_or_regions.pop(np.where(tmp)[0][0])

            if len(eligible_classes_or_regions) == 0:
                # this only happens if some image does not contain foreground voxels at all
                selected_class = None
                if verbose:
                    print('case does not contain any foreground classes')
            else:
                # I hate myself. Future me aint gonna be happy to read this
                # 2022_11_25: had to read it today. Wasn't too bad
                selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            # print(f'I want to have foreground, selected class: {selected_class}')
          
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs