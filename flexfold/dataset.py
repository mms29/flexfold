"""Classes for using particle image datasets in PyTorch learning methods.

This module contains classes that implement various preprocessing and data access
methods acting on the image data stored in a cryodrgn.source.ImageSource class.
These methods are used by learning methods such as those used in volume reconstruction
algorithms; the classes are thus implemented as children of torch.utils.data.Dataset
to allow them to inherit behaviour such as batch training.

For example, during initialization, ImageDataset initializes an ImageSource class and
then also estimates normalization parameters, a non-trivial computational step. When
image data is retrieved during model training using __getitem__, the data is whitened
using these parameters.

"""
import numpy as np
from collections import Counter, OrderedDict

import logging
import torch
from typing import Optional, Tuple, Union
from cryodrgn import fft
from cryodrgn.source import ImageSource, StarfileSource, parse_star
from cryodrgn.masking import spherical_window_mask

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class DataSplits(torch.utils.data.Dataset):
    def __init__(
        self,
        imageDataset,
        indices,
    ):    
        self.indices = indices
        self.N = len(indices)
        self.imageDataset = imageDataset
        self.D = self.imageDataset.D
        self.domain = self.imageDataset.domain
        self.src = self.imageDataset.src

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.imageDataset[self.indices[index]]
    
    def get_slice(self, start, stop) :
        return (
            self.src.images(slice(self.indices[start], self.indices[stop]), require_contiguous=True).numpy(),
            None,
        )
    
    def _process(self, data):
        return self.imageDataset._process(data)
    def _process_real(self, data):
        return self.imageDataset._process_real(data)
    def _process_ft(self, data):
        return self.imageDataset._process_ft(data)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mrcfile,
        lazy=True,
        norm=None,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        max_threads=16,
        device: Union[str, torch.device] = "cpu",
        domain="real",
    ):
        self.domain=domain
        datadir = datadir or ""
        self.ind = ind
        self.src = ImageSource.from_file(
            mrcfile,
            lazy=lazy,
            datadir=datadir,
            indices=ind,
            max_threads=max_threads,
        )
        ny = self.src.D
        assert ny % 2 == 0, "Image size must be even."
        self.N = self.src.n
        self.D = ny + 1  # after symmetrization
        self.invert_data = invert_data

        if window:
            self.window = spherical_window_mask(D=ny, in_rad=window_r, out_rad=0.99)
        else:
            self.window = None

        norm = norm or self.estimate_normalization()
        self.norm = [float(x) for x in norm]
        self.device = device
        self.lazy = lazy

        if np.issubdtype(self.src.dtype, np.integer):
            self.window = self.window.int()

    def estimate_normalization(self, n=1000):
        n = min(n, self.N) if n is not None else self.N
        indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??

        imgs = torch.stack([fft.ht2_center(img) for img in self.src.images(indices)])
        if self.invert_data:
            imgs *= -1

        imgs = fft.symmetrize_ht(imgs)
        norm = (0, torch.std(imgs))
        logger.info("Normalizing HT by {} +/- {}".format(*norm))

        return norm

    def _process(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if self.window is not None:
            data *= self.window

        data = fft.ht2_center(data)
        if self.invert_data:
            data *= -1
        data = fft.symmetrize_ht(data)
        data = (data - self.norm[0]) / self.norm[1]
        return data

    def _process_ft(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if self.window is not None:
            data *= self.window

        data = fft.fft2_center(data)
        if self.invert_data:
            data *= -1
        data = fft.symmetrize_ht(data)
        data = (data - self.norm[0]) / self.norm[1]
        return data

    def _process_real(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if self.invert_data:
            data *= -1
        data = (data - self.norm[0]) / self.norm[1]
        return data
    
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if isinstance(index, list):
            index = torch.Tensor(index).to(torch.long)

        imgs = self.src.images(index)

        particles_real = self._process_real(imgs)
        if self.domain=="hartley":
            particles = self._process(imgs)
        elif self.domain=="fourier":
            particles = self._process_ft(imgs)
        else:
            particles=None

        # this is why it is tricky for index to be allowed to be a list!
        if len(particles_real.shape) == 2:
            particles_real = particles_real[np.newaxis, ...]
            if particles is not None:
                particles = particles[np.newaxis, ...]

        if isinstance(index, (int, np.integer)):
            logger.debug(f"ImageDataset returning images at index ({index})")
        else:
            logger.debug(
                f"ImageDataset returning images for {len(index)} indices:"
                f" ({index[0]}..{index[-1]})"
            )

        return particles, particles_real, index

    def get_slice(
        self, start: int, stop: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return (
            self.src.images(slice(start, stop), require_contiguous=True).numpy(),
            None,
        )

class DataShuffler:
    def __init__(
        self, dataset: ImageDataset, batch_size, buffer_size, dtype=np.float32
    ):
        if not all(dataset.src.indices == np.arange(dataset.N)):
            raise NotImplementedError(
                "NotImplementedError: --ind is not supported for the data shuffler. "
                "The purpose of the shuffler is to load chunks contiguously during "
                "lazy loading on huge datasets, which doesn't work with --ind subsets. "
                "We recommend instead using --ind during preprocessing (e.g. with "
                "`cryodrgn downsample`) if you aim to use the shuffler or simply "
                "pass --lazy for on-the-fly data loading (potentially slower)."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dtype = dtype
        assert self.buffer_size % self.batch_size == 0, (
            self.buffer_size,
            self.batch_size,
        )  # FIXME
        self.batch_capacity = self.buffer_size // self.batch_size
        assert self.buffer_size <= len(self.dataset), (
            self.buffer_size,
            len(self.dataset),
        )

    def __iter__(self):
        return _DataShufflerIterator(self)


class _DataShufflerIterator:
    def __init__(self, shuffler: DataShuffler):
        self.dataset = shuffler.dataset
        self.buffer_size = shuffler.buffer_size
        self.batch_size = shuffler.batch_size
        self.batch_capacity = shuffler.batch_capacity
        self.dtype = shuffler.dtype

        self.buffer = np.empty(
            (self.buffer_size, 1, self.dataset.D - 1, self.dataset.D - 1),
            dtype=self.dtype,
        )
        self.index_buffer = np.full((self.buffer_size,), -1, dtype=np.int64)

        self.num_batches = (
            len(self.dataset) // self.batch_size
        )  # FIXME off-by-one? Nah, lets leave the last batch behind
        self.chunk_order = torch.randperm(self.num_batches)
        self.count = 0
        self.flush_remaining = -1  # at the end of the epoch, got to flush the buffer
        # pre-fill
        logger.info("Pre-filling data shuffler buffer...")
        for i in range(self.batch_capacity):
            chunk, chunk_indices = self._get_next_chunk()
            self.buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk
            self.index_buffer[
                i * self.batch_size : (i + 1) * self.batch_size
            ] = chunk_indices
        logger.info(
            f"Filled buffer with {self.buffer_size} images ({self.batch_capacity} contiguous chunks)."
        )

    def _get_next_chunk(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        chunk_idx = int(self.chunk_order[self.count])
        self.count += 1
        particles, _ = self.dataset.get_slice(
            chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
        )
        particle_indices = np.arange(
            chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
        )
        particles = particles.reshape(
            self.batch_size, 1, *particles.shape[1:]
        )
        return particles, particle_indices

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch of images, and the indices of those images in the dataset.

        The buffer starts filled with `batch_capacity` random contiguous chunks.
        Each time a batch is requested, `batch_size` random images are selected from the buffer,
        and refilled with the next random contiguous chunk from disk.

        Once all the chunks have been fetched from disk, the buffer is randomly permuted and then
        flushed sequentially.
        """
        if self.count == self.num_batches and self.flush_remaining == -1:
            logger.info(
                "Finished fetching chunks. Flushing buffer for remaining batches..."
            )
            # since we're going to flush the buffer sequentially, we need to shuffle it first
            perm = np.random.permutation(self.buffer_size)
            self.buffer = self.buffer[perm]
            self.index_buffer = self.index_buffer[perm]
            self.flush_remaining = self.buffer_size

        if self.flush_remaining != -1:
            # we're in flush mode, just return chunks out of the buffer
            assert self.flush_remaining % self.batch_size == 0
            if self.flush_remaining == 0:
                raise StopIteration()
            particles = self.buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            particle_indices = self.index_buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            self.flush_remaining -= self.batch_size
        else:
            indices = np.random.choice(
                self.buffer_size, size=self.batch_size, replace=False
            )
            particles = self.buffer[indices]
            particle_indices = self.index_buffer[indices]

            chunk, chunk_indices = self._get_next_chunk()
            self.buffer[indices] = chunk
            self.index_buffer[indices] = chunk_indices

        particles = torch.from_numpy(particles)
        particle_indices = torch.from_numpy(particle_indices)

        # merge the batch dimension
        particles = particles.view(-1, *particles.shape[2:])

        particles_real = self.dataset._process_real(particles)
        if self.dataset.domain=="hartley":
            particles = self.dataset._process(particles)
        elif self.dataset.domain=="fourier":
            particles = self.dataset._process_ft(particles)
        else:
            particles=None

        # print('ZZZ', particles.shape, tilt_indices.shape, particle_indices.shape)
        return particles, particles_real, particle_indices


def make_dataloader(
    data: ImageDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffler_size: int = 0,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    if shuffler_size > 0 and shuffle:
        assert data.lazy, "Only enable a data shuffler for lazy loading"
        return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
    else:
        # see https://github.com/zhonge/cryodrgn/pull/221#discussion_r1120711123
        # for discussion of why we use BatchSampler, etc.
        if shuffle:
            generator = None if seed is None else torch.Generator().manual_seed(seed)
            sampler = RandomSampler(data, generator=generator)
        else:
            sampler = SequentialSampler(data)

        return DataLoader(
            data,
            num_workers=num_workers,
            sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
            batch_size=None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
