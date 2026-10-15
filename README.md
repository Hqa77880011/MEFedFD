# MEFedFD

Implementation of **Reliability Propagation for Federated Multi-Sensor Fault Diagnosis under Sensor Loss**.

MEFedFD propagates sensor availability into mask-normalized fusion, Dirichlet evidence, local consistency and reliability-weighted federated aggregation. This project contains recording preparation, training, the paper's federated comparisons and single-component ablations, sensor-loss evaluation, reliability diagnostics, paired statistics and figures.

## Install

Use Python 3.11 or 3.12 and install from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before installing (`source .venv/bin/activate` on Linux/macOS; `.venv\Scripts\Activate.ps1` in PowerShell). For CUDA, install a matching PyTorch build before the remaining requirements. All commands also accept CPU training through the default `--device cpu`.

## 1. Download and describe recordings

Obtain recordings from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) or the [Paderborn KAt Data Center](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter). CWRU uses synchronized 12-kHz drive-end/fan-end channels, grouped by loads 0–3 hp. Paderborn uses synchronized vibration/motor-current channels under `N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, and `N15_M07_F04`. Exclude ambiguous or mixed Paderborn fault labels before creating the manifest.

Create a UTF-8 CSV describing complete recordings:

```csv
channel1,channel2,key1,key2,label,client,group,recording
raw/97.mat,raw/97.mat,X097_DE_time,X097_FE_time,0,load0,normal,record97
```

Each row represents one recording with both synchronized channels. Paths are relative to the CSV. `client` is the operating condition; `group` is physical bearing identity; `recording` is a unique ID within a client. Labels are consecutive integers from zero: CWRU healthy/inner/outer/rolling-element = 0/1/2/3; Paderborn healthy/inner/outer = 0/1/2. Supply records for every class and client.

Inputs may be `.mat`, one-dimensional `.npy`, or `.npz`. For NPZ the default variable is `signal`; `key1`/`key2` can select another variable. MAT keys can traverse simplified nested structures with slash-separated fields and zero-based list indices, for example `record/Y/6/Data`; inspect the actual release's channel names before choosing indices. Every extracted channel must be a finite one-dimensional vector. Channels must have equal lengths and share their native sample grid.

The manifest specifies metadata and channel selection explicitly because file names alone do not identify every release's labels and physical groups.

## 2. Split and prepare windows

```bash
python -m mefedfd prepare --manifest data/cwru.csv --protocol cwru --output data/cwru.npz
python -m mefedfd prepare --manifest data/paderborn.csv --protocol paderborn --output data/paderborn.npz --seed 2024
```

CWRU records are divided chronologically at 60%/80%. The implementation excludes one full 2048-sample window on each side of each boundary before extracting overlapping windows. Paderborn bearing groups are allocated by class to 60/20/20 using seeded largest-remainder allocation, shared across all operating conditions. A physical bearing therefore never crosses Paderborn partitions. A class with too few groups for three nonempty partitions is rejected.

Default window length is 2048 with stride 1024. `--window` and `--stride` change these values; the CWRU guard length follows `--window`. Each client's training-window statistics normalize that client's train, validation and test channels. The NPZ contains `x [N,2,L]`, labels, clients, split, recording, within-record offset, bearing group and normalization statistics. Neither raw records nor prepared windows are included in Git.

## 3. Train

```bash
python -m mefedfd train --data data/paderborn.npz --output outputs/mefedfd_2024 --method mefedfd --seed 2024 --device cuda
```

All clients participate. Defaults follow the paper: 100 rounds, two local epochs, batch size 128, AdamW learning rate 0.001 and weight decay 0.0001. Each sensor has Conv1d blocks `(16,7)`, `(32,5)`, `(64,3)` with batch normalization and ReLU, a 64-dimensional projection and a scalar gate. The classifier consumes the fused vector and two availability bits. With dropout zero, the masked model has `31,042 + 65*K` parameters.

Training samples full and missing views with missing probability 0.5 and equal probability of dropping either sensor; both sensors are never missing. Missing inputs are masked before encoding, and normalized fusion gives them exactly zero weight. Evidence is `softplus(logits)+1`. The loss includes expected Dirichlet cross-entropy, annealed incorrect-evidence KL, uncertainty ordering and confidence-gated complete-to-masked consistency. Local validation estimates confidence/correctness agreement under all three allowed masks. Aggregation applies its mask-balance penalty, exponential memory and positive weight floor.

| Parameter | Default | Use |
|---|---:|---|
| `--rounds`, `--local-epochs` | 100, 2 | Communication and local optimization budgets |
| `--batch-size`, `--lr`, `--weight-decay` | 128, 0.001, 0.0001 | Optimizer settings |
| `--missing` | 0.5 | Training missing-view probability; use 0.25 for the shift experiment |
| `--validation-fraction` | 1.0 | Fraction of the smallest validation class retained per class; use 0.5/0.25 for reliability stress |
| `--memory`, `--floor`, `--gamma` | 0.8, 0.2, 2 | Reliability memory, weight floor and exponent |
| `--mask-balance` | 0.5 | Penalty on reliability differences between masks |
| `--threshold`, `--xi`, `--eta` | 0.4, 0.2, 0.5 | Teacher reliability threshold, consistency and ordering coefficients |
| `--kl-weight` | 1 | Final KL weight, annealed over the first ten rounds |
| `--proximal` | 0.001 | FedProx coefficient |
| `--server-lr` | 1 | Server parameter interpolation; SCAFFOLD grid uses 0.5/1 |
| `--dropout` | 0 | Classifier dropout |
| `--block` | 16 | Consecutive missing-sensor windows in block evaluation |
| `--audit-test` | off | Collect held-out masked errors for the reliability diagnostic only |
| `--target-f1` | 0.8 | Validation threshold for rounds/communication-to-target |

The class-balanced validation subset is fixed for each seed. Every round is scored on client-balanced validation Macro-F1 averaged across the three sensor patterns. The best round is saved to `best.pt` and restored before final test evaluation. `--audit-test` adds diagnostic test observations to history; those observations do not enter optimization, aggregation or checkpoint selection.

## 4. Baselines and ablations

Use the same data, seed, rounds and batch settings across methods:

```bash
python -m mefedfd train --data data/paderborn.npz --output outputs/fedavg_me_2024 --method fedavg_me --seed 2024 --device cuda
python -m mefedfd train --data data/paderborn.npz --output outputs/no_consistency_2024 --method no_consistency --seed 2024 --device cuda
```

Available methods:

| Name | Behavior |
|---|---|
| `mefedfd` | Complete method |
| `fedavg` | Two-channel encoder and softmax classification |
| `fedavg_m` | Mask-normalized fusion and softmax, count-weighted aggregation |
| `fedavg_me` | Mask-normalized evidence model, count-weighted aggregation |
| `fedprox_m` | Mask-aware softmax model with proximal local loss |
| `scaffold_m` | Mask-aware softmax model with server/client control variates |
| `fedbn_m` | Mask-aware softmax model retaining every client's batch-normalization state |
| `local` | Independent client models, without aggregation |
| `centralized` | Pooled training reference |
| `no_consistency` | Removes masked evidential consistency |
| `no_mask_balance` | Removes the mask-balance penalty |
| `no_memory` | Uses current-round reliability directly |
| `no_reliability` | Count-weighted aggregation of the full local objective |
| `no_evidence` | Replaces Dirichlet output with softmax uncertainty |
| `no_mask_training` | Trains on complete views |
| `no_normalization` | Zeroes absent-sensor contributions after an unmasked gate normalization |

SCAFFOLD uses local SGD so its control-variate update matches the local step equation; the other methods use AdamW. Batch-normalization running statistics are averaged for ordinary federated methods and kept client-local for FedBN. Convolution stride is two in each block. Local and centralized references use the same evidential local objective as MEFedFD. These are the concrete architecture/optimizer conventions where the paper does not provide a separate reference configuration.

For the paper's validation search, run the Cartesian product of learning rate `{0.0003,0.001}` and weight decay `{0.00001,0.0001}` with dropout `{0,0.1}`. Replace the third dimension for FedProx with `--proximal {0.001,0.01}`, SCAFFOLD with `--server-lr {0.5,1}`, FedAvg-ME with `--kl-weight {0.5,1}`, and MEFedFD with `--gamma {1,2}`. Pick the highest `best_validation_macro_f1` before comparing test metrics. Repeat seeds 2024 through 2033. No results are supplied in advance.

## 5. Evaluation and outputs

Training ends with all sensor patterns, missing ratios 0/0.25/0.5/0.75, sensor-1/2 biased missingness, fixed random masks and contiguous missing blocks. The same seed fixes the random masks across methods. Primary paper aggregation computes each client's per-pattern metrics first, averages clients equally, and mixes patterns using `(1-rho, rho*(1-bias), rho*bias)`. Nonlinear metrics from an actual sampled missing-mask population are stored separately as `fixed_random_mask`.

`metrics.json` contains:

- Accuracy, Macro-F1, 15-bin ECE, error-detection AUROC and worst-client F1 for every condition.
- Separate full-input, sensor-1-only and sensor-2-only client results.
- Best validation round/score, parameter count, training time and parameter communication.
- Rounds and communication bytes needed to reach the chosen validation target; `null` means the target was not reached.
- Reliability CV and its correlation with validation error; enabling `--audit-test` adds the held-out masked-error Spearman diagnostic.

Error-detection positives are misclassified examples. Evidential models use `K/sum(alpha)`; softmax models use `1-max(p)`. AUROC is `null` when a partition has no errors or no correct predictions. Macro-F1 includes every declared class. `history.json` records validation progression, local losses, reliability, weights and communication per round. Communication counts float parameter transfers and reliability scalars, plus SCAFFOLD controls; transport envelopes and dataset transfer are excluded.

## 6. Paired statistics and plots

Collect one selected configuration per method and seed. Paired analysis requires at least two matched seeds and compares to `fedavg_me` at missing ratio 0.5:

```bash
python -m mefedfd compare --metrics outputs/fedavg_me_2024/metrics.json outputs/fedavg_me_2025/metrics.json outputs/mefedfd_2024/metrics.json outputs/mefedfd_2025/metrics.json --output outputs/paired.json
python -m mefedfd plot --metrics outputs/fedavg_me_2024/metrics.json outputs/mefedfd_2024/metrics.json --output outputs/figures
```

The statistical output reports method-minus-baseline differences, 10,000 seed-level paired bootstrap intervals, Wilcoxon tests and Holm-adjusted p-values across the supplied comparisons. F1 and AUROC favor positive differences; ECE favors negative differences. Plotting creates missingness curves for all primary metrics and validation-versus-communication curves. Error bars across supplied runs are sample standard deviations across seeds.

Source modules separate data preparation, the method/loss, federated training, metrics/statistics and command handling. GitHub Actions installs dependencies and checks source compilation and imports without training.
