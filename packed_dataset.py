# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import math

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

        # --- DDP Compatibility: Calculate length ---
        self._num_files = len(self._filenames)
        self._n_blocks = None
        if self._num_files > 0:
            # Read header from the first file to determine chunk size and calculate blocks per file
            try:
                # Need the _read_header logic here or similar
                with open(self._filenames[0], "rb") as f:
                    magic = f.read(len(HDR_MAGIC))
                    if magic == HDR_MAGIC:
                        version = struct.unpack("<Q", f.read(8))
                        (dtype_code,) = struct.unpack("<B", f.read(1))
                        # dtype = dtypes[dtype_code] # Not needed for length calculation
                        (chunk_size,) = struct.unpack("<Q", f.read(8))
                        if self._block_size > 0:
                             self._n_blocks = chunk_size // self._block_size
                        else:
                             self._n_blocks = 0
                    else:
                        print(f"Warning: File {self._filenames[0]} doesn't match expected format for length calculation.")
                        self._n_blocks = 0 # Or raise error
            except Exception as e:
                print(f"Warning: Could not read header from {self._filenames[0]} to determine length: {e}")
                self._n_blocks = 0 # Assume 0 if header read fails

        # Store these for __iter__ if needed, DDP Sampler doesn't need them directly
        self._num_processes = num_processes
        self._process_rank = process_rank
        # --- End DDP Compatibility ---

    # --- ADD THIS METHOD ---
    def __len__(self):
        """ Estimates the total number of blocks/samples in the dataset assigned to this instance. """
        if self._n_blocks is None or self._n_blocks == 0:
             # Try to calculate again if not done in init (e.g., empty filenames initially)
             # This is a basic recalculation, might need refinement if filenames change post-init
             if not self._filenames: return 0
             try:
                 with open(self._filenames[0], "rb") as f:
                     magic = f.read(len(HDR_MAGIC))
                     if magic == HDR_MAGIC:
                         version = struct.unpack("<Q", f.read(8))
                         (dtype_code,) = struct.unpack("<B", f.read(1))
                         (chunk_size,) = struct.unpack("<Q", f.read(8))
                         if self._block_size > 0:
                              n_blocks_calc = chunk_size // self._block_size
                         else:
                              n_blocks_calc = 0
                         # Update self._n_blocks if it was None/0
                         if self._n_blocks is None or self._n_blocks == 0:
                              self._n_blocks = n_blocks_calc
                         elif self._n_blocks != n_blocks_calc:
                              print(f"Warning: Calculated blocks per chunk changed unexpectedly ({self._n_blocks} vs {n_blocks_calc})")
                              # Optionally update self._n_blocks or keep original estimate
                     else:
                         if self._n_blocks is None or self._n_blocks == 0: return 0 # Failed
             except Exception:
                  if self._n_blocks is None or self._n_blocks == 0: return 0 # Failed

        # If _n_blocks is still None or 0 after trying, return 0
        if self._n_blocks is None or self._n_blocks <= 0:
            print(f"Warning: Could not determine number of blocks per file for PackedDataset. Returning length 0.")
            return 0

        # Return total estimated blocks
        return self._num_files * self._n_blocks
    # --- END ADDED METHOD ---

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # --- DDP Compatibility: File assignment is now handled by DistributedSampler ---
        # The DistributedSampler will provide indices, and the fetcher will get data.
        # We still need the iterator logic, but the list of filenames it operates on
        # should ideally be the full list, letting the sampler choose indices.
        # However, PackedDatasetIterator expects its *own* list.
        # A simple fix for now is to let PackedDatasetIterator iterate over *all* assigned files.
        # The DistributedSampler ensures each rank only gets indices corresponding to its slice.

        # We can simplify __iter__ if DistributedSampler handles file distribution implicitly.
        # The sampler gives indices, the dataset fetcher needs to map index to file+offset.
        # PackedDatasetIterator is designed differently (iterates files then blocks).
        # Let's keep the iterator but ensure it gets the *full list* if sampler manages indices.

        # --- Revised __iter__ logic for DDP with Sampler ---
        # NOTE: This assumes DistributedSampler works correctly by giving indices
        #       that the underlying fetcher can map back to file/block locations.
        #       If the fetcher *relies* on the iterator yielding blocks sequentially
        #       from *its assigned file subset*, this might break things.
        # Let's stick closer to the original idea: give iterator its file subset.
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id
        
        # Calculate files per shard - use ceil division
        num_all_files = len(self._filenames)
        files_per_shard = math.ceil(num_all_files / num_shards)
        start_idx = shard_id * files_per_shard
        end_idx = min(start_idx + files_per_shard, num_all_files)
        shard_filenames = self._filenames[start_idx:end_idx]
        #max_num_files = len(self._filenames) // num_shards * num_shards
        #filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            #filenames=filenames,
            filenames=shard_filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            # if not self._wrap:
            #     raise StopIteration
            self._file_idx = 0

        #num_files_to_load = min(self._n_chunks, len(self._filenames) - self._file_idx)

        #if num_files_to_load <= 0:
        #    if self._wrap:
        #        self._file_idx = 0
        #        num_files_to_load = min(self._n_chunks, len(self._filenames))
        #        if num_files_to_load <= 0:
        #            raise StopIteration
        #    else:
        #        raise StopIteration

        for i in range(self._n_chunks):
        #for i in range(num_files_to_load):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        #self._file_idx += num_files_to_load
        n_all_blocks = self._n_chunks * self._n_blocks
        #n_all_blocks = 0
        #if self._n_blocks is not None:
        #     n_all_blocks = num_files_to_load * self._n_blocks
        #else:
        #    if num_files_to_load == 0:
        #         raise StopIteration("No blocks to process after loading chunks.")

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    # --- ADD THIS METHOD ---
    def __len__(self):
        """ Returns the sum of lengths of underlying datasets. """
        total_len = 0
        for ds in self._datasets:
            try:
                # Ensure the underlying dataset actually has __len__ now
                total_len += len(ds)
            except TypeError:
                # Handle cases where an underlying dataset might still not have len
                print(f"Warning: Dataset {type(ds)} in CombinedDataset does not support len(). Combined length might be inaccurate.")
                # Option 1: Raise error
                # raise TypeError(f"Dataset {type(ds)} in CombinedDataset must implement __len__ for DistributedSampler.")
                # Option 2: Return an estimate or skip (returning 0 if any fails is safer for sampler)
                return 0 # Or handle more gracefully if possible
        return total_len
    # --- END ADDED METHOD --

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        # Create iterators for each underlying dataset
        self._dataset_iters = [iter(el) for el in datasets]
        self._datasets_with_data = list(range(len(datasets))) # Track datasets that still have data
        #self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        if not self._datasets_with_data: # Stop if all datasets are exhausted
            raise StopIteration
        # Adjust weights for datasets that still have data
        active_weights = [self._weights[i] for i in self._datasets_with_data]
        sum_active_weights = sum(active_weights)
        if sum_active_weights <= 0: # Avoid division by zero if weights are bad
             # Fallback to uniform sampling among remaining
             active_weights = [1.0] * len(self._datasets_with_data)
             sum_active_weights = len(self._datasets_with_data)

        normalized_weights = [w / sum_active_weights for w in active_weights]


        # Sample a dataset index from the *active* datasets
        (chosen_active_idx,) = self._rng.choices(range(len(self._datasets_with_data)), weights=normalized_weights, k=1)
        # Map back to the original dataset index
        chosen_orig_idx = self._datasets_with_data[chosen_active_idx]
        chosen_iter = self._dataset_iters[chosen_orig_idx]

        try:
            # Try to get the next item from the chosen dataset's iterator
            return next(chosen_iter)
        except StopIteration:
            # If this dataset is exhausted, remove it from the active list and try again
            self._datasets_with_data.pop(chosen_active_idx)
            # Recursively call next to sample from the remaining datasets
            return self.__next__()

        #(dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        #return next(dataset)
