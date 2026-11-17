"""Like-for-like uncompressed NPZ: dense arrays versus FP32/int32 CSR arrays."""
import numpy as np
import torch


def save_storage(model, folder):
    arrays = {n: t.detach().cpu().numpy() for n, t in model.state_dict().items()}
    np.savez(folder / 'dense.npz', **arrays)
    packed = {}
    for name, array in arrays.items():
        if array.ndim < 2:
            packed[name+'__dense'] = array
            continue
        matrix = array.reshape(array.shape[0], -1)
        rows, cols = np.nonzero(matrix)
        packed[name+'__values'] = matrix[rows, cols]
        packed[name+'__columns'] = cols.astype(np.int32)
        packed[name+'__rowptr'] = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(matrix)))].astype(np.int32)
        packed[name+'__shape'] = np.array(array.shape, dtype=np.int32)
    np.savez(folder / 'csr.npz', **packed)
    restored = load_csr(folder / 'csr.npz')
    assert arrays.keys() == restored.keys()
    for name in arrays:
        assert np.array_equal(arrays[name], restored[name]), name
    return {'dense_npz_bytes': (folder/'dense.npz').stat().st_size,
            'csr_npz_bytes': (folder/'csr.npz').stat().st_size,
            'checkpoint_bytes': (folder/'model.pt').stat().st_size}


def load_csr(path):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if key.endswith('__dense'):
                result[key[:-7]] = archive[key]
            elif key.endswith('__shape'):
                name = key[:-7]; shape = tuple(archive[key])
                values = archive[name+'__values']; cols = archive[name+'__columns']
                ptr = archive[name+'__rowptr']
                matrix = np.zeros((shape[0], int(np.prod(shape[1:]))), dtype=np.float32)
                for row in range(shape[0]):
                    matrix[row, cols[ptr[row]:ptr[row+1]]] = values[ptr[row]:ptr[row+1]]
                result[name] = matrix.reshape(shape)
    return result
