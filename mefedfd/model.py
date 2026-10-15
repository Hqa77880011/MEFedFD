import torch
from torch import nn
from torch.nn import functional as F


MASKS = ((1, 1), (1, 0), (0, 1))


def encoder(channels=1):
    layers = []
    for width, kernel in ((16, 7), (32, 5), (64, 3)):
        layers.extend((nn.Conv1d(channels, width, kernel, stride=2, padding=kernel // 2),
                       nn.BatchNorm1d(width), nn.ReLU()))
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())


class DiagnosisModel(nn.Module):
    def __init__(self, classes, evidential=True, normalized=True, raw=False, dropout=0):
        super().__init__()
        self.classes = classes
        self.evidential = evidential
        self.normalized = normalized
        self.raw = raw
        if raw:
            self.encoder = encoder(2)
            width = 64
        else:
            self.encoders = nn.ModuleList([encoder(), encoder()])
            self.projections = nn.ModuleList([nn.Linear(64, 64), nn.Linear(64, 64)])
            self.gates = nn.ModuleList([nn.Linear(64, 1), nn.Linear(64, 1)])
            width = 66
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, classes))

    def forward(self, x, mask):
        x = x * mask[:, :, None]
        if self.raw:
            features = self.encoder(x)
        else:
            encoded = [self.encoders[sensor](x[:, sensor:sensor + 1]) for sensor in range(2)]
            scores = torch.cat([self.gates[sensor](encoded[sensor]) for sensor in range(2)], 1)
            if self.normalized:
                weights = scores.masked_fill(mask == 0, -torch.inf).softmax(1)
            else:
                weights = scores.softmax(1) * mask
            projected = torch.stack([self.projections[sensor](encoded[sensor]) for sensor in range(2)], 1)
            fused = (weights[:, :, None] * projected).sum(1)
            features = torch.cat((fused, mask), 1)
        logits = self.head(features)
        if self.evidential:
            alpha = F.softplus(logits) + 1
            probability = alpha / alpha.sum(1, keepdim=True)
            uncertainty = self.classes / alpha.sum(1)
        else:
            alpha = None
            probability = logits.softmax(1)
            uncertainty = 1 - probability.max(1).values
        return probability, uncertainty, alpha


def dirichlet_loss(alpha, labels, anneal):
    target = F.one_hot(labels, alpha.shape[1]).to(alpha)
    total = alpha.sum(1)
    nll = (target * (torch.digamma(total[:, None]) - torch.digamma(alpha))).sum(1)
    adjusted = target + (1 - target) * alpha
    prior = torch.ones_like(adjusted)
    kl = torch.distributions.kl_divergence(
        torch.distributions.Dirichlet(adjusted), torch.distributions.Dirichlet(prior))
    return nll + anneal * kl


def objective(model, x, y, mask, round_number, consistency=True, margin=True,
              kl_weight=1.0, xi=0.2, eta=0.5, threshold=0.4):
    complete = torch.ones_like(mask)
    full_p, full_u, full_a = model(x, complete)
    masked_p, masked_u, masked_a = model(x, mask)
    if model.evidential:
        anneal = min(1, round_number / 10) * kl_weight
        loss = dirichlet_loss(full_a, y, anneal) + dirichlet_loss(masked_a, y, anneal)
        if margin:
            loss = loss + eta * F.relu(full_u + 0.05 * (2 - mask.sum(1)) - masked_u)
    else:
        loss = F.nll_loss(full_p.clamp_min(1e-8).log(), y, reduction="none")
        loss = loss + F.nll_loss(masked_p.clamp_min(1e-8).log(), y, reduction="none")
    if consistency:
        teacher = full_p.detach().sqrt()
        teacher = teacher / teacher.sum(1, keepdim=True)
        gate = ((1 - full_u.detach() - threshold) / (1 - threshold)).clamp_min(0)
        kl = (teacher * (teacher.clamp_min(1e-8).log() - masked_p.clamp_min(1e-8).log())).sum(1)
        loss = loss + xi * gate * kl
    return loss.mean()


def sample_mask(count, missing, device, generator=None, bias=0.5):
    draws = torch.rand(count, generator=generator)
    sensors = torch.rand(count, generator=generator)
    mask = torch.ones(count, 2)
    missing_rows = torch.nonzero(draws < missing).flatten()
    dropped = (sensors[missing_rows] >= bias).long()
    mask[missing_rows, dropped] = 0
    return mask.to(device)
