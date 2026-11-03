import argparse
import pickle

import numpy as np
from tqdm import tqdm


def convert(src, dst):
    print(f"Loading {src} ...")
    with open(src, 'rb') as file:
        data = pickle.load(file)

    n = len(data)
    shape = np.asarray(data[0]).shape
    n_bytes = (int(np.prod(shape)) + 7) // 8
    packed = np.empty((n, n_bytes), dtype=np.uint8)

    for i in tqdm(range(n), desc="Packing"):
        cube = np.asarray(data[i])
        if cube.shape != shape:
            raise ValueError(f"Cube {i} has shape {cube.shape}, expected {shape}")
        if not np.isin(cube, (0, 1)).all():
            raise ValueError(f"Cube {i} contains values other than 0 and 1")
        packed[i] = np.packbits(cube.astype(np.uint8).ravel())

    print(f"Writing {dst} ...")
    np.savez_compressed(dst, packed=packed, shape=np.array(shape))

    # Round-trip check: reload and compare every cube against the original
    loaded = load_packed(dst)
    for i in tqdm(range(n), desc="Verifying"):
        if not np.array_equal(loaded[i], np.asarray(data[i])):
            raise ValueError(f"Cube {i} does not match after round trip")
    print(f"Done: {n} cubes of shape {shape}")


def load_packed(path):
    # Returns a uint8 array of shape (N, *cube_shape) with values in {0, 1}
    with np.load(path) as f:
        packed, shape = f['packed'], tuple(f['shape'])
    n_voxels = int(np.prod(shape))
    cubes = np.unpackbits(packed, axis=1, count=n_voxels)
    return cubes.reshape(len(packed), *shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a pickled list of binary voxel cubes to a bit-packed, compressed .npz")
    parser.add_argument('--src', default='/home/grads/wzhan24/Voxplorer/datacreate/10k_vox.pkl')
    parser.add_argument('--dst', default='10k_vox_packed.npz')
    args = parser.parse_args()
    convert(args.src, args.dst)
