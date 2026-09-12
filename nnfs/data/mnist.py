import gzip
import struct
import urllib.request
from pathlib import Path
import numpy as np
import math

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path

def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST image file")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows*cols)
    return data.astype(np.float64) / 255.0

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST label file")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

def load_mnist(root="data/mnist", limit_train=None, limit_test=None, seed=42):
    root = Path(root)
    paths = {k: _download(url, root/(k+".gz")) for k,url in URLS.items()}
    X_train, y_train = _read_images(paths["train_images"]), _read_labels(paths["train_labels"])
    X_test, y_test = _read_images(paths["test_images"]), _read_labels(paths["test_labels"])
    rng = np.random.default_rng(seed)
    if limit_train:
        idx = rng.choice(len(X_train), min(limit_train,len(X_train)), replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if limit_test:
        idx = rng.choice(len(X_test), min(limit_test,len(X_test)), replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
    return X_train, y_train, X_test, y_test

class DataLoader:
    def __init__(self, X, y, batch_size=128, shuffle=True, seed=42):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(self.X))
        if self.shuffle:
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            b = idx[start:start+self.batch_size]
            yield self.X[b], self.y[b]

    def __len__(self):
        return math.ceil(len(self.X)/self.batch_size)
