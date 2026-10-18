# MEFedFD

Implementation of **Reliability Propagation for Federated Multi-Sensor Fault Diagnosis under Sensor Loss**.

## Installation

Use Python 3.11 or 3.12 for installation. Run from the repository root in an activated environment.

```bash
python -m pip install -r requirements.txt
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain recordings from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) or the [Paderborn KAt Data Center](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter). CWRU uses synchronized 12-kHz drive-end/fan-end channels, grouped by loads 0–3 hp. Paderborn uses synchronized vibration/motor-current channels under `N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, and `N15_M07_F04`. Exclude ambiguous or mixed Paderborn fault labels before creating the manifest.

Create a UTF-8 CSV describing complete recordings:

```csv
channel1,channel2,key1,key2,label,client,group,recording
raw/97.mat,raw/97.mat,X097_DE_time,X097_FE_time,0,load0,normal,record97
```

Each row represents one recording with both synchronized channels. Paths are relative to the CSV. `client` is the operating condition; `group` is physical bearing identity; `recording` is a unique ID within a client. Labels are consecutive integers from zero: CWRU healthy/inner/outer/rolling-element = 0/1/2/3; Paderborn healthy/inner/outer = 0/1/2. Supply records for every class and client.

Inputs may be `.mat`, one-dimensional `.npy`, or `.npz`. For NPZ the default variable is `signal`; `key1`/`key2` can select another variable. MAT keys can traverse simplified nested structures with slash-separated fields and zero-based list indices, for example `record/Y/6/Data`; inspect the actual release's channel names before choosing indices. Every extracted channel must be a finite one-dimensional vector. Channels must have equal lengths and share their native sample grid.

The manifest specifies metadata and channel selection explicitly because file names alone do not identify every release's labels and physical groups.

```bash
python -m mefedfd prepare --manifest data/cwru.csv --protocol cwru --output data/cwru.npz
python -m mefedfd prepare --manifest data/paderborn.csv --protocol paderborn --output data/paderborn.npz --seed 2024
```

CWRU records are divided chronologically at 60%/80%. The implementation excludes one full 2048-sample window on each side of each boundary before extracting overlapping windows. Paderborn bearing groups are allocated by class to 60/20/20 using seeded largest-remainder allocation, shared across all operating conditions. A physical bearing therefore never crosses Paderborn partitions. A class with too few groups for three nonempty partitions is rejected.

Default window length is 2048 with stride 1024. `--window` and `--stride` change these values; the CWRU guard length follows `--window`. Each client's training-window statistics normalize that client's train, validation and test channels. The NPZ contains `x [N,2,L]`, labels, clients, split, recording, within-record offset, bearing group and normalization statistics. Neither raw records nor prepared windows are included in Git.

## Training and inference

```bash
python -m mefedfd train --data data/paderborn.npz --output outputs/mefedfd_2024 --method mefedfd --seed 2024
```

## Evaluation

`train` restores the validation-selected model, then evaluates held-out full, single-missing and block-missing sensor conditions.

## Comparisons and ablations

```bash
python -m mefedfd train --data data/paderborn.npz --output outputs/fedavg_me_2024 --method fedavg_me --seed 2024
python -m mefedfd train --data data/paderborn.npz --output outputs/no_consistency_2024 --method no_consistency --seed 2024
```

Use the same seed, sensor-mask policy and budgets. `--missing`, `--validation-fraction` and `--block` control the supported stress settings.

## Outputs

`best.pt` stores the selected model; `history.json` stores round/reliability records; `metrics.json` contains test metrics and sensor-pattern diagnostics.

```bash
python -m mefedfd plot --metrics outputs/fedavg_me_2024/metrics.json outputs/mefedfd_2024/metrics.json --output outputs/figures
```
