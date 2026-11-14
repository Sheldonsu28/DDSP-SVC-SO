"""FAISS feature index construction, adapted from the RVC indexing workflow."""
import os
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def train_index(spk_name, root_dir="data", normalize=False, clustering=True, feat="units",
                train_path=None, valid_path=None):
    roots = (Path(train_path or Path(root_dir) / "train"), Path(valid_path or Path(root_dir) / "val"))
    files = []
    for root in roots:
        directory = root / feat / str(spk_name)
        if not directory.is_dir():
            raise FileNotFoundError(f"Feature directory does not exist: {directory}; run preprocessing first")
        files.extend(directory.rglob("*.npy"))
    if not files:
        raise ValueError(f"No {feat} features found; run preprocessing first")
    arrays = [np.load(path, allow_pickle=False) for path in sorted(files)]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Feature files must have shape (frames, channels)")
    vectors = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    if not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Index features must be nonempty and finite")
    np.random.shuffle(vectors)
    if vectors.shape[0] > 200000 and clustering:
        vectors = np.ascontiguousarray(MiniBatchKMeans(n_clusters=10000,
            batch_size=512 * (os.cpu_count() or 4), compute_labels=False,
            init="random").fit(vectors).cluster_centers_, dtype=np.float32)
    if normalize:
        import torch
        vectors = torch.nn.functional.normalize(torch.from_numpy(vectors), dim=1, eps=1e-12).numpy()
    cells = min(int(16 * np.sqrt(len(vectors))), len(vectors) // 39)
    if cells < 1:
        index = faiss.IndexFlatL2(vectors.shape[1])
    else:
        index = faiss.index_factory(vectors.shape[1], f"IVF{cells},Flat")
        faiss.extract_index_ivf(index).nprobe = 1
        index.train(vectors)
    for start in range(0, len(vectors), 8192):
        index.add(vectors[start:start + 8192])
    return index
