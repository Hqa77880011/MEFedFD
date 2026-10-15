import argparse
import json
from pathlib import Path

import numpy as np

from .data import prepare
from .metrics import compare
from .train import METHODS, train


def plot(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    experiments = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    methods = sorted({run["method"] for run in experiments})
    for metric in ("accuracy", "macro_f1", "ece", "error_auroc", "worst_client_f1"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in methods:
            runs = [run for run in experiments if run["method"] == method]
            ratios = [0, 0.25, 0.5, 0.75]
            means, deviations = [], []
            for ratio in ratios:
                values = [row[metric] for run in runs for row in run["evaluation"]["summary"]
                          if row["protocol"] == "expected_mixture" and row["missing"] == ratio
                          and row["drop_sensor1_probability"] == 0.5 and row[metric] is not None]
                means.append(np.mean(values) if values else np.nan)
                deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            ax.errorbar(ratios, means, yerr=deviations, marker="o", label=method)
        ax.set(xlabel="Missing sensor ratio", ylabel=metric.replace("_", " "))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f"{metric}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for path, experiment in zip(paths, experiments):
        history = json.loads((Path(path).parent / "history.json").read_text(encoding="utf-8"))
        ax.plot([row["communication_bytes"] / 1e6 for row in history],
                [row["validation_macro_f1"] for row in history],
                label=f"{experiment['method']} seed={experiment['seed']}")
    ax.set(xlabel="Cumulative parameter communication (MB)", ylabel="Validation macro-F1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "communication.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m mefedfd")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--manifest", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--protocol", choices=("cwru", "paderborn"), required=True)
    data.add_argument("--window", type=int, default=2048)
    data.add_argument("--stride", type=int, default=1024)
    data.add_argument("--seed", type=int, default=2024)
    run = commands.add_parser("train")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--method", choices=METHODS, default="mefedfd")
    run.add_argument("--rounds", type=int, default=100)
    run.add_argument("--local-epochs", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--lr", type=float, default=0.001)
    run.add_argument("--weight-decay", type=float, default=0.0001)
    run.add_argument("--dropout", type=float, default=0)
    run.add_argument("--seed", type=int, default=2024)
    run.add_argument("--device", default="cpu")
    run.add_argument("--missing", type=float, default=0.5)
    run.add_argument("--validation-fraction", type=float, default=1)
    run.add_argument("--memory", type=float, default=0.8)
    run.add_argument("--floor", type=float, default=0.2)
    run.add_argument("--gamma", type=float, default=2)
    run.add_argument("--mask-balance", type=float, default=0.5)
    run.add_argument("--threshold", type=float, default=0.4)
    run.add_argument("--xi", type=float, default=0.2)
    run.add_argument("--eta", type=float, default=0.5)
    run.add_argument("--kl-weight", type=float, default=1)
    run.add_argument("--proximal", type=float, default=0.001)
    run.add_argument("--server-lr", type=float, default=1)
    run.add_argument("--block", type=int, default=16)
    run.add_argument("--audit-test", action="store_true")
    run.add_argument("--target-f1", type=float, default=0.8)
    stats = commands.add_parser("compare")
    stats.add_argument("--metrics", nargs="+", required=True)
    stats.add_argument("--output", default="outputs/paired.json")
    stats.add_argument("--resamples", type=int, default=10000)
    stats.add_argument("--seed", type=int, default=0)
    visual = commands.add_parser("plot")
    visual.add_argument("--metrics", nargs="+", required=True)
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.protocol, args.window, args.stride, args.seed)
    elif args.command == "train":
        train(args)
    elif args.command == "compare":
        compare(args.metrics, args.output, args.seed, args.resamples)
    else:
        plot(args.metrics, args.output)


if __name__ == "__main__":
    main()
