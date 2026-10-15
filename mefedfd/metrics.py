import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .model import MASKS, sample_mask


def diagnosis_metrics(labels, probability, uncertainty, classes):
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == labels
    edges = np.linspace(0, 1, 16)
    bins = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, 14)
    ece = sum(
        np.mean(bins == index) * abs(correct[bins == index].mean() - confidence[bins == index].mean())
        for index in range(15) if np.any(bins == index)
    )
    error = (~correct).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=range(classes), average="macro", zero_division=0)),
        "ece": float(ece),
        "error_auroc": float(roc_auc_score(error, uncertainty)) if len(np.unique(error)) == 2 else None,
    }


@torch.no_grad()
def predictions(model, x, mask, device, batch_size=128):
    model.eval()
    probabilities, uncertainties = [], []
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start:start + batch_size], device=device)
        mb = torch.as_tensor(mask[start:start + batch_size], dtype=xb.dtype, device=device)
        probability, uncertainty, _ = model(xb, mb)
        probabilities.append(probability.cpu().numpy())
        uncertainties.append(uncertainty.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(uncertainties)


def balanced_average(rows, weights=None):
    if weights is None:
        weights = np.full(len(rows), 1 / len(rows))
    result = {}
    for key in ("accuracy", "macro_f1", "ece", "error_auroc"):
        values = [row[key] for row in rows]
        result[key] = None if any(v is None and w > 0 for v, w in zip(values, weights)) else float(
            sum(v * w for v, w in zip(values, weights) if w > 0))
    return result


def evaluate_models(models, data, split, device, seed, ratios=(0, 0.25, 0.5, 0.75), block=16):
    clients = np.unique(data["client"])
    detailed = []
    mixtures = []
    for client in clients:
        model = models[client] if isinstance(models, dict) else models
        indices = np.flatnonzero((data["client"] == client) & (data["split"] == split))
        if not len(indices):
            raise ValueError(f"Client {client} has no {split} samples")
        x, y = data["x"][indices], data["y"][indices]
        patterns = []
        for pattern in MASKS:
            probability, uncertainty = predictions(model, x, np.tile(pattern, (len(x), 1)), device)
            metrics = diagnosis_metrics(y, probability, uncertainty, model.classes)
            detailed.append(dict(client=str(client), mask=list(pattern), **metrics))
            patterns.append(metrics)
        for ratio in ratios:
            for bias in (0.25, 0.5, 0.75):
                weighted = balanced_average(patterns, [1 - ratio, ratio * (1 - bias), ratio * bias])
                mixtures.append(dict(client=str(client), missing=ratio, drop_sensor1_probability=bias,
                                     protocol="expected_mixture", **weighted))
                generator = torch.Generator().manual_seed(seed)
                mask = sample_mask(len(x), ratio, "cpu", generator, bias).numpy()
                probability, uncertainty = predictions(model, x, mask, device)
                mixtures.append(dict(client=str(client), missing=ratio, drop_sensor1_probability=bias,
                                     protocol="fixed_random_mask", **diagnosis_metrics(y, probability, uncertainty, model.classes)))
        order = np.lexsort((data["offset"][indices], data["recording"][indices]))
        mask = np.ones((len(x), 2), dtype=np.float32)
        for recording in np.unique(data["recording"][indices]):
            record_rows = order[data["recording"][indices][order] == recording]
            for start in range(0, len(record_rows), block * 2):
                mask[record_rows[start:start + block], (start // (block * 2)) % 2] = 0
        probability, uncertainty = predictions(model, x, mask, device)
        mixtures.append(dict(client=str(client), missing=0.5, drop_sensor1_probability=0.5,
                             protocol="contiguous_blocks", **diagnosis_metrics(y, probability, uncertainty, model.classes)))
    aggregate = []
    configurations = {(row["missing"], row["drop_sensor1_probability"], row["protocol"]) for row in mixtures}
    for ratio, bias, protocol in sorted(configurations):
        rows = [r for r in mixtures if (r["missing"], r["drop_sensor1_probability"], r["protocol"]) == (ratio, bias, protocol)]
        aggregate.append(dict(missing=ratio, drop_sensor1_probability=bias, protocol=protocol,
                              worst_client_f1=min(r["macro_f1"] for r in rows), **balanced_average(rows)))
    return {"client_patterns": detailed, "client_conditions": mixtures, "summary": aggregate}


def reliability_audit(history):
    pairs = [(client["reliability_memory"], client["validation_error"])
             for row in history for client in row["clients"]]
    reliability, error = np.asarray(pairs).T
    correlation = spearmanr(reliability, error).statistic if np.std(reliability) and np.std(error) else None
    series = {}
    for row in history:
        for client in row["clients"]:
            series.setdefault(client["client"], []).append(client["reliability_memory"])
    cv = [np.std(values, ddof=1) / np.mean(values) for values in series.values()
          if len(values) > 1 and np.mean(values) > 0]
    result = {"validation_reliability_error_spearman": float(correlation) if correlation is not None else None,
              "reliability_cv": float(np.mean(cv)) if cv else None}
    heldout = [(client["reliability_memory"], client["heldout_masked_error"])
               for row in history for client in row["clients"] if "heldout_masked_error" in client]
    if heldout:
        reliability, error = np.asarray(heldout).T
        result["heldout_reliability_error_spearman"] = float(spearmanr(reliability, error).statistic) if np.std(reliability) and np.std(error) else None
    return result


def compare(paths, output, seed=0, resamples=10000):
    experiments = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_method = {}
    for experiment in experiments:
        key = experiment["method"]
        run_seed = experiment["seed"]
        if run_seed in by_method.setdefault(key, {}):
            raise ValueError(f"Duplicate seed {run_seed} for {key}; compare one dataset/configuration at a time")
        by_method[key][run_seed] = experiment
    reference = by_method.get("fedavg_me")
    if reference is None:
        raise ValueError("Paired analysis requires fedavg_me runs")
    rng = np.random.default_rng(seed)
    records = []
    for method, runs in by_method.items():
        if method == "fedavg_me":
            continue
        common = sorted(set(runs) & set(reference))
        if len(common) < 2:
            raise ValueError(f"At least two matched seeds required for {method}")
        for metric in ("macro_f1", "ece", "error_auroc"):
            def value(run):
                return next(row[metric] for row in run["evaluation"]["summary"]
                            if row["missing"] == 0.5 and row["drop_sensor1_probability"] == 0.5
                            and row["protocol"] == "expected_mixture")
            pairs = [(value(runs[s]), value(reference[s])) for s in common]
            if any(a is None or b is None for a, b in pairs):
                records.append({"method": method, "metric": metric, "reason": "undefined AUROC"})
                continue
            difference = np.array([a - b for a, b in pairs])
            boot = np.mean(rng.choice(difference, (resamples, len(common)), replace=True), axis=1)
            pvalue = float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0
            records.append(dict(method=method, metric=metric, seeds=common,
                                mean_difference=float(difference.mean()),
                                ci95=np.quantile(boot, [0.025, 0.975]).tolist(), pvalue=pvalue))
    valid = [row for row in records if "pvalue" in row]
    ordered = sorted(valid, key=lambda row: row["pvalue"])
    previous = 0
    for index, row in enumerate(ordered):
        previous = max(previous, min(1, (len(ordered) - index) * row["pvalue"]))
        row["holm_pvalue"] = previous
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
