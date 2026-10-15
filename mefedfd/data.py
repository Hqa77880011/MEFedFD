import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def signal(path, key=None):
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            value = archive[key or "signal"]
    elif path.suffix == ".mat":
        if not key:
            raise ValueError(f"A MAT variable key is required for {path}")
        value = loadmat(path, simplify_cells=True)
        for component in key.split("/"):
            value = value[int(component)] if component.isdigit() else value[component]
    else:
        raise ValueError(f"Unsupported signal format: {path.suffix}")
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"Signal must be a finite one-dimensional vector: {path}")
    return value


def group_split(rows, seed):
    rng = np.random.default_rng(seed)
    assignments = {}
    classes = sorted({int(row["label"]) for row in rows})
    for label in classes:
        groups = sorted({row["group"] for row in rows if int(row["label"]) == label})
        if len(groups) < 3:
            raise ValueError(f"Class {label} needs at least three bearing groups")
        rng.shuffle(groups)
        ideal = np.array([0.6, 0.2, 0.2]) * len(groups)
        counts = np.floor(ideal).astype(int)
        for index in np.argsort(-(ideal - counts), kind="stable")[:len(groups) - counts.sum()]:
            counts[index] += 1
        if np.any(counts == 0):
            raise ValueError(f"Class {label} has insufficient groups for 60/20/20 allocation")
        start = 0
        for split, count in zip(("train", "val", "test"), counts):
            for group in groups[start:start + count]:
                if group in assignments and assignments[group] != split:
                    raise ValueError(f"Bearing {group} has inconsistent class labels")
                assignments[group] = split
            start += count
    return assignments


def prepare(manifest, output, protocol, window=2048, stride=1024, seed=2024):
    manifest = Path(manifest).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"channel1", "channel2", "label", "client", "group", "recording"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    if window < 2 or not 1 <= stride <= window:
        raise ValueError("Invalid window or stride")
    labels = sorted({int(row["label"]) for row in rows})
    if labels != list(range(len(labels))):
        raise ValueError("Labels must be contiguous integers starting at zero")
    identities = [(row["client"], row["recording"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate recording/client rows in manifest")
    assignments = group_split(rows, seed) if protocol == "paderborn" else None
    chunks = []
    for row in rows:
        first = signal(manifest.parent / row["channel1"], row.get("key1"))
        second = signal(manifest.parent / row["channel2"], row.get("key2"))
        if len(first) != len(second):
            raise ValueError(f"Unsynchronized channel lengths: {row['recording']}")
        paired = np.stack((first, second))
        if protocol == "cwru":
            a, b = int(len(first) * 0.6), int(len(first) * 0.8)
            intervals = [("train", 0, a - window), ("val", a + window, b - window),
                         ("test", b + window, len(first))]
        else:
            intervals = [(assignments[row["group"]], 0, len(first))]
        for split, start, end in intervals:
            for offset in range(start, end - window + 1, stride):
                chunks.append((paired[:, offset:offset + window], int(row["label"]),
                               row["client"], split, row["recording"], offset, row["group"]))
    if not chunks:
        raise ValueError("No windows remain after splitting and guard intervals")
    x = np.stack([c[0] for c in chunks])
    y = np.array([c[1] for c in chunks], dtype=np.int64)
    client = np.array([c[2] for c in chunks])
    split = np.array([c[3] for c in chunks])
    means, deviations = {}, {}
    for name in np.unique(client):
        train = (client == name) & (split == "train")
        if not train.any():
            raise ValueError(f"Client {name} has no training windows")
        mean = x[train].mean(axis=(0, 2), keepdims=True)
        deviation = x[train].std(axis=(0, 2), keepdims=True).clip(1e-6)
        selected = client == name
        x[selected] = (x[selected] - mean) / deviation
        means[name], deviations[name] = mean.tolist(), deviation.tolist()
        for part in ("train", "val", "test"):
            if not ((client == name) & (split == part)).any():
                raise ValueError(f"Client {name} has no {part} windows")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x, y=y, client=client, split=split,
                        recording=np.array([c[4] for c in chunks]),
                        offset=np.array([c[5] for c in chunks]),
                        group=np.array([c[6] for c in chunks]),
                        normalization_clients=np.array(list(means)),
                        normalization_mean=np.array(list(means.values())),
                        normalization_std=np.array(list(deviations.values())),
                        protocol=protocol, seed=seed)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in ("x", "y", "client", "split", "recording", "offset")}
    result["x"] = result["x"].astype(np.float32)
    result["y"] = result["y"].astype(np.int64)
    x, y = result["x"], result["y"]
    if x.ndim != 3 or x.shape[1] != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Expected finite windows [N, 2, L] and matching labels")
    if set(np.unique(result["split"])) != {"train", "val", "test"}:
        raise ValueError("Data must contain train, val and test partitions")
    if any(len(value) != len(x) for value in result.values()):
        raise ValueError("All sample metadata must match the window count")
    if sorted(np.unique(y).tolist()) != list(range(int(y.max()) + 1)):
        raise ValueError("Labels must be contiguous integers starting at zero")
    return result
