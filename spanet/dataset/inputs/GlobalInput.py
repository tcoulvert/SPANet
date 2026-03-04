from typing import Tuple, List

import h5py
import numpy as np

import torch
from torch import Tensor

from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput


class GlobalInput(BaseInput):

    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Try and load a mask for this global features. If none is present, assume all vectors are valid.
        try:
            source_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]).contiguous()
        except KeyError:
            source_mask = torch.ones(self.num_events, dtype=torch.bool)

        # First pass: determine total feature dimension (support 1D scalars and 2D vectors per feature).
        feature_dims: List[int] = []
        feature_flags: List[Tuple[bool, bool]] = []  # (normalize, log_scale) per logical feature
        for feat in self.event_info.input_features[self.input_name]:
            feature, normalize, log_scale = feat.name, feat.normalize, feat.log_scale
            dset = self.dataset(hdf5_file, input_group, feature)
            shape = dset.shape
            if len(shape) == 1:
                dims = 1
            elif len(shape) == 2:
                if shape[1] == 1:
                    dims = 1
                else:
                    dims = shape[1]  # vector: (num_events, K) -> K dimensions
            else:
                raise ValueError(
                    f"Global input '{self.input_name}/{feature}': expected 1D (num_events,) or 2D (num_events, K), got shape {shape}"
                )
            feature_dims.append(dims)
            for _ in range(dims):
                feature_flags.append((normalize, log_scale))

        num_features = sum(feature_dims)
        # noinspection PyAttributeOutsideInit
        self._normalized_features = np.array([f[0] for f in feature_flags], dtype=bool)
        # noinspection PyAttributeOutsideInit
        self._log_features = np.array([f[1] for f in feature_flags], dtype=bool)

        # Load in vector features.
        source_data = torch.empty(num_features, self.num_events, dtype=torch.float32)
        row_offset = 0
        for feat, dims in zip(self.event_info.input_features[self.input_name], feature_dims):
            feature, log_transform = feat.name, feat.log_scale
            dset = self.dataset(hdf5_file, input_group, feature)
            arr = np.asarray(dset)
            if arr.ndim == 1:
                source_data[row_offset].numpy()[:] = arr
                if log_transform:
                    source_data[row_offset] += 1
                    torch.log_(source_data[row_offset])
                    source_data[row_offset] *= source_mask
                row_offset += 1
            else:
                # (num_events, K) -> copy into source_data[row_offset:row_offset+K, :]
                if arr.shape[1] != dims:
                    raise ValueError(
                        f"Global input '{self.input_name}/{feature}': expected 2D with second dim {dims}, got {arr.shape}"
                    )
                for k in range(dims):
                    source_data[row_offset + k].numpy()[:] = arr[:, k]
                    if log_transform:
                        source_data[row_offset + k] += 1
                        torch.log_(source_data[row_offset + k])
                        source_data[row_offset + k] *= source_mask
                row_offset += dims

        # Reshape and limit data to the limiting index.
        source_data = source_data.transpose(0, 1)
        source_data = source_data[limit_index].contiguous()
        source_mask = source_mask[limit_index].contiguous()

        # Add a fake timestep dimension to global vectors.
        # noinspection PyAttributeOutsideInit
        self.source_data = source_data.unsqueeze(1)
        # noinspection PyAttributeOutsideInit
        self.source_mask = source_mask.unsqueeze(1)

    @property
    def reconstructable(self) -> bool:
        return False

    # noinspection PyAttributeOutsideInit
    def limit(self, event_mask):
        self.source_data = self.source_data[event_mask].contiguous()
        self.source_mask = self.source_mask[event_mask].contiguous()

    def compute_statistics(self) -> Tuple[Tensor, Tensor]:
        masked_data = self.source_data[self.source_mask]
        masked_mean = masked_data.mean(0)
        masked_std = masked_data.std(0)

        masked_std[masked_std < 1e-5] = 1

        # Use per-dimension flags (supports vector features with same normalize/log for all dims).
        masked_mean[~self._normalized_features] = 0
        masked_std[~self._normalized_features] = 1

        return Statistics(masked_mean, masked_std)

    def num_input_features(self) -> int:
        """Actual feature dimension (supports vector features: one key -> 2D array)."""
        return self.source_data.shape[2]

    def num_vectors(self) -> int:
        return self.source_mask.sum(1)

    def max_vectors(self) -> int:
        return self.source_mask.shape[1]

    def __getitem__(self, item):
        return Source(self.source_data[item], self.source_mask[item])
