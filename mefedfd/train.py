import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load
from .metrics import evaluate_models, predictions, reliability_audit, diagnosis_metrics
from .model import MASKS, DiagnosisModel, objective, sample_mask


def heldout_masked_error(model, data, client, device):
    selected = (data["client"] == client) & (data["split"] == "test")
    x, y = data["x"][selected], data["y"][selected]
    errors = []
    for pattern in MASKS[1:]:
        probability, _ = predictions(model, x, np.tile(pattern, (len(x), 1)), device)
        errors.append(float(np.mean(probability.argmax(1) != y)))
    return float(np.mean(errors))


METHODS = ("mefedfd", "fedavg", "fedavg_m", "fedavg_me", "fedprox_m", "scaffold_m",
           "fedbn_m", "local", "centralized", "no_consistency", "no_mask_balance",
           "no_memory", "no_reliability", "no_evidence", "no_mask_training", "no_normalization")


def balanced_indices(data, client, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero((data["client"] == client) & (data["split"] == "val"))
    groups = [indices[data["y"][indices] == label] for label in range(int(data["y"].max()) + 1)]
    if any(len(group) == 0 for group in groups):
        raise ValueError(f"Client {client} validation must contain every class")
    count = max(1, int(min(map(len, groups)) * fraction))
    return np.concatenate([rng.choice(group, count, replace=False) for group in groups])


def reliability(model, data, indices, device, balance):
    quality, errors = [], []
    x, y = data["x"][indices], data["y"][indices]
    for pattern in MASKS:
        probability, uncertainty = predictions(model, x, np.tile(pattern, (len(x), 1)), device)
        correct = (probability.argmax(1) == y).astype(float)
        quality.append(1 - np.abs(correct - (1 - uncertainty)).mean())
        errors.append(1 - correct.mean())
    return float(np.clip(np.mean(quality) - balance * np.std(quality), 0, 1)), float(np.mean(errors))


def validation_score(models, data, device):
    scores = []
    for client in np.unique(data["client"]):
        model = models[client] if isinstance(models, dict) else models
        selected = (data["client"] == client) & (data["split"] == "val")
        x, y = data["x"][selected], data["y"][selected]
        for pattern in MASKS:
            p, u = predictions(model, x, np.tile(pattern, (len(x), 1)), device)
            scores.append(diagnosis_metrics(y, p, u, model.classes)["macro_f1"])
    return float(np.mean(scores))


def train(args):
    if not 0 <= args.missing <= 1 or not 0 < args.validation_fraction <= 1:
        raise ValueError("Invalid missing rate or validation fraction")
    if min(args.rounds, args.local_epochs, args.batch_size) < 1:
        raise ValueError("Training lengths must be positive")
    if not 0 <= args.memory < 1 or not 0 < args.floor <= 1 or args.gamma <= 0:
        raise ValueError("Require 0 <= memory < 1, 0 < floor <= 1, gamma > 0")
    if not 0 <= args.threshold < 1 or args.lr <= 0 or args.block < 1:
        raise ValueError("Invalid threshold, learning rate or block length")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load(args.data)
    device = torch.device(args.device)
    classes = int(data["y"].max()) + 1
    softmax = args.method in ("fedavg", "fedavg_m", "fedprox_m", "scaffold_m", "fedbn_m", "no_evidence")
    model = DiagnosisModel(classes, evidential=not softmax,
                           normalized=args.method != "no_normalization",
                           raw=args.method == "fedavg", dropout=args.dropout).to(device)
    clients = np.unique(data["client"]).tolist()
    local_models = {client: copy.deepcopy(model) for client in clients}
    memory = {client: 1.0 for client in clients}
    audit_indices = {client: balanced_indices(data, client, args.validation_fraction, args.seed) for client in clients}
    bn_keys = {name for name in model.state_dict()
               if any(name.startswith(prefix + ".") for prefix, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))}
    server_control = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    client_control = {client: copy.deepcopy(server_control) for client in clients}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -np.inf
    start_time = time.perf_counter()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    communication = 0
    for round_index in range(1, args.rounds + 1):
        states, counts, controls, client_rows = [], [], [], []
        training_clients = ["all"] if args.method == "centralized" else clients
        global_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        for client in training_clients:
            if args.method == "centralized":
                local_model = copy.deepcopy(model)
                selected = data["split"] == "train"
            else:
                local_model = local_models[client]
                if args.method != "local":
                    state = model.state_dict()
                    if args.method == "fedbn_m":
                        state = {key: local_model.state_dict()[key] if key in bn_keys else value
                                 for key, value in state.items()}
                    local_model.load_state_dict(state)
                selected = (data["client"] == client) & (data["split"] == "train")
            initial = {name: p.detach().clone() for name, p in local_model.named_parameters()}
            dataset = TensorDataset(torch.from_numpy(data["x"][selected]), torch.from_numpy(data["y"][selected]))
            generator = torch.Generator().manual_seed(args.seed + round_index * 1009)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
            if args.method == "scaffold_m":
                optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr)
            else:
                optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            local_model.train()
            steps = 0
            total_loss = 0
            own_method = args.method not in ("fedavg", "fedavg_m", "fedavg_me", "fedprox_m", "scaffold_m", "fedbn_m")
            for epoch in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    mask = sample_mask(len(x), 0 if args.method == "no_mask_training" else args.missing,
                                       device, generator)
                    optimizer.zero_grad()
                    loss = objective(local_model, x, y, mask, round_index,
                                     consistency=own_method and args.method != "no_consistency",
                                     margin=own_method and not softmax,
                                     kl_weight=args.kl_weight, xi=args.xi, eta=args.eta,
                                     threshold=args.threshold)
                    if args.method == "fedprox_m":
                        loss = loss + args.proximal / 2 * sum(
                            (p - global_parameters[name]).square().sum() for name, p in local_model.named_parameters())
                    loss.backward()
                    if args.method == "scaffold_m":
                        for name, parameter in local_model.named_parameters():
                            parameter.grad.add_(server_control[name] - client_control[client][name])
                    optimizer.step()
                    total_loss += float(loss.detach())
                    steps += 1
            if args.method == "scaffold_m":
                next_control = {name: client_control[client][name] - server_control[name]
                                + (initial[name] - p.detach()) / (steps * args.lr)
                                for name, p in local_model.named_parameters()}
                controls.append({name: next_control[name] - client_control[client][name]
                                 for name in next_control})
                client_control[client] = next_control
            audit_client = clients[0] if args.method == "centralized" else client
            quality, error = reliability(local_model, data, audit_indices[audit_client], device,
                                         0 if args.method == "no_mask_balance" else args.mask_balance)
            coefficient = 0 if args.method == "no_memory" else args.memory
            memory[audit_client] = coefficient * memory[audit_client] + (1 - coefficient) * quality
            counts.append(len(dataset))
            states.append(copy.deepcopy(local_model.state_dict()))
            if args.method == "centralized":
                model.load_state_dict(local_model.state_dict())
            else:
                local_models[client] = local_model
            client_rows.append(dict(client=str(client), loss=total_loss / steps,
                                    reliability=quality, reliability_memory=memory[audit_client],
                                    validation_error=error))
            if args.audit_test and args.method != "centralized":
                client_rows[-1]["heldout_masked_error"] = heldout_masked_error(local_model, data, client, device)
        weights = np.asarray(counts, dtype=float)
        bounded = args.method not in ("fedavg", "fedavg_m", "fedavg_me", "fedprox_m", "scaffold_m",
                                      "fedbn_m", "local", "centralized", "no_reliability")
        if bounded:
            weights *= np.array([args.floor + (1 - args.floor) * memory[c] for c in clients]) ** args.gamma
        weights /= weights.sum()
        if args.method not in ("local", "centralized"):
            current = model.state_dict()
            for key in current:
                if args.method == "fedbn_m" and key in bn_keys:
                    continue
                if current[key].is_floating_point():
                    averaged = sum(float(w) * state[key] for w, state in zip(weights, states))
                    current[key] = current[key] + args.server_lr * (averaged - current[key])
                else:
                    current[key] = states[0][key]
            model.load_state_dict(current)
            if args.method == "scaffold_m":
                for name in server_control:
                    server_control[name] += sum(delta[name] for delta in controls) / len(clients)
            communication += 2 * len(clients) * parameter_bytes
            if bounded:
                communication += 4 * len(clients)
            if args.method == "scaffold_m":
                communication += 2 * len(clients) * parameter_bytes
        if args.method == "fedbn_m":
            for client in clients:
                state = local_models[client].state_dict()
                state.update({key: value for key, value in model.state_dict().items() if key not in bn_keys})
                local_models[client].load_state_dict(state)
        evaluation_model = local_models if args.method in ("local", "fedbn_m") else model
        score = validation_score(evaluation_model, data, device)
        row = dict(round=round_index, validation_macro_f1=score, communication_bytes=communication,
                   elapsed_seconds=time.perf_counter() - start_time, clients=client_rows,
                   aggregation_weights=weights.tolist())
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            checkpoint = dict(args=vars(args), classes=classes, round=round_index,
                              score=score, model=copy.deepcopy(model.state_dict()),
                              local_models={c: copy.deepcopy(m.state_dict()) for c, m in local_models.items()})
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    for client in clients:
        local_models[client].load_state_dict(checkpoint["local_models"][client])
    evaluation_model = local_models if args.method in ("local", "fedbn_m") else model
    evaluation = evaluate_models(evaluation_model, data, "test", device, args.seed, block=args.block)
    result = dict(method=args.method, seed=args.seed, best_round=checkpoint["round"],
                  best_validation_macro_f1=best_score, parameters=sum(p.numel() for p in model.parameters()),
                  communication_bytes=communication, training_seconds=time.perf_counter() - start_time,
                  evaluation=evaluation, reliability_audit=reliability_audit(history), config=vars(args))
    reaching = [row for row in history if row["validation_macro_f1"] >= args.target_f1]
    result["rounds_to_target"] = reaching[0]["round"] if reaching else None
    result["communication_to_target_bytes"] = reaching[0]["communication_bytes"] if reaching else None
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
