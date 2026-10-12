"""Dataset preparation utilities for real texture dataset."""

import io
import os
import random
import shutil
import zipfile
from collections import Counter

import cv2
import numpy as np

SUPPORTED = (".jpg", ".jpeg", ".png")


def is_image(name):
    return name.lower().endswith(SUPPORTED)


def normalize_zip_path(n):
    return n.replace("\\", "/")


def _find_label_from_parts(parts):
    # textures.zip -> parts like ['textures_upload1','banded','file.jpg']
    # nested -> ['banded','file.jpg']
    if len(parts) >= 2:
        # if top is 'textures_upload1' skip
        if parts[0] in ("textures_upload1", "textures", "data"):
            if len(parts) >= 3:
                return parts[1], "/".join(parts[2:])
        return parts[0], "/".join(parts[1:])
    return None, None


def iter_labeled_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            n = normalize_zip_path(info.filename)
            if ".." in n or n.startswith(("/", "C:")):
                continue
            parts = n.split("/")
            label, rel = _find_label_from_parts(parts)
            if label is None or not is_image(rel or n):
                continue
            try:
                data = z.read(info)
            except Exception:
                continue
            yield label, os.path.basename(rel or n), data


def extract_valid_samples(zip_path, max_size=200_000_000):
    samples = []
    seen = set()
    for label, name, data in iter_labeled_from_zip(zip_path):
        h = hash((label, name, len(data)))
        if h in seen:
            continue
        seen.add(h)
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h * w > max_size:
                continue
            samples.append((label, name, data, img))
        except Exception:
            continue
    return samples


def make_stratified_split(samples, train=0.7, val=0.15, test=0.15, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6
    rng = random.Random(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s[0], []).append(s)
    train_list, val_list, test_list = [], [], []
    for c, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        t1 = int(n * train)
        t2 = t1 + int(n * val)
        train_list.extend(items[:t1])
        val_list.extend(items[t1:t2])
        test_list.extend(items[t2:])
    return train_list, val_list, test_list


def describe_split(split):
    c = Counter(s[0] for s in split)
    return dict(c), sum(c.values())


def save_split_to_dir(split, root):
    os.makedirs(root, exist_ok=True)
    for label, name, data, _ in split:
        d = os.path.join(root, label)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(data)
