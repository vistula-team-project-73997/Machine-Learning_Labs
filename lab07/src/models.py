"""Small-convolution CNNs with deliberately oversized/compact dense heads."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, c1=16, c2=32, hidden=640):
        super().__init__()
        self.arch = [c1, c2, hidden]
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def weight_parameters(model):
    return {n: p for n, p in model.named_parameters() if p.ndim >= 2}


def magnitude_masks(model, sparsity, scope):
    """Prune exactly round(s*N) coefficients; stable index tie-breaking."""
    params = weight_parameters(model)
    masks = {n: torch.ones_like(p) for n, p in params.items()}
    if scope == 'global':
        flat = torch.cat([p.detach().abs().flatten() for p in params.values()])
        mask = torch.ones_like(flat)
        mask[torch.argsort(flat, stable=True)[:round(sparsity * flat.numel())]] = 0
        offset = 0
        for n, p in params.items():
            masks[n] = mask[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
    elif scope == 'layerwise':
        for n, p in params.items():
            order = torch.argsort(p.detach().abs().flatten(), stable=True)
            masks[n].flatten()[order[:round(sparsity * p.numel())]] = 0
    else:
        raise ValueError(scope)
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    for n, p in model.named_parameters():
        if n in masks:
            p.mul_(masks[n])


def structured_compact(model, fraction):
    """Rank original conv filters independently by L2, slice adjacent tensors."""
    keep = []
    for conv in (model.conv1, model.conv2):
        norms = conv.weight.detach().flatten(1).norm(p=2, dim=1)
        remove = round(fraction * len(norms))
        indices = torch.argsort(norms, stable=True)[remove:].sort().values
        keep.append(indices)
    a, b = keep
    out = CNN(len(a), len(b), model.arch[2])
    with torch.no_grad():
        out.conv1.weight.copy_(model.conv1.weight[a])
        out.conv1.bias.copy_(model.conv1.bias[a])
        out.conv2.weight.copy_(model.conv2.weight[b][:, a])
        out.conv2.bias.copy_(model.conv2.bias[b])
        cols = (b[:, None] * 49 + torch.arange(49)).flatten()
        out.fc1.weight.copy_(model.fc1.weight[:, cols])
        out.fc1.bias.copy_(model.fc1.bias)
        out.fc2.load_state_dict(model.fc2.state_dict())
    # Dense masked reference must agree before any fine-tuning.
    reference = copy.deepcopy(model)
    with torch.no_grad():
        drop_a = torch.ones(model.arch[0], dtype=torch.bool); drop_a[a] = False
        drop_b = torch.ones(model.arch[1], dtype=torch.bool); drop_b[b] = False
        reference.conv1.weight[drop_a] = 0; reference.conv1.bias[drop_a] = 0
        reference.conv2.weight[drop_b] = 0; reference.conv2.bias[drop_b] = 0
    return out, reference, [a.tolist(), b.tolist()]


def distillation_loss(student, teacher, labels, temperature=4., alpha=.5):
    """Literal manual direction KL(p_student || p_teacher), per-example sum."""
    log_s = F.log_softmax(student / temperature, dim=1)
    log_t = F.log_softmax(teacher.detach() / temperature, dim=1)
    kl = (log_s.exp() * (log_s - log_t)).sum(dim=1).mean()
    ce = F.cross_entropy(student, labels)
    return alpha * ce + (1-alpha) * temperature**2 * kl, ce, kl


def compute_counts(model):
    """MAC excludes biases, ReLU, pooling; two FLOPs per multiply-add."""
    c1, c2, h = model.arch
    dense = 28*28*c1*9 + 14*14*c2*c1*9 + 49*c2*h + h*10
    nz = (int(torch.count_nonzero(model.conv1.weight))*28*28 +
          int(torch.count_nonzero(model.conv2.weight))*14*14 +
          int(torch.count_nonzero(model.fc1.weight)) + int(torch.count_nonzero(model.fc2.weight)))
    return dense, nz
