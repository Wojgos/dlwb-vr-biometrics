"""Trzy klasyfikatory: Multi-Input Branch CNN, GRU, Transformer.

Kazdy model zwraca (logits, embedding); embedding to 128-wymiarowa warstwa przedostatnia,
na ktorej liczymy podobienstwo cosinusowe, Rank-1 i RSA.
"""
import math
import torch
import torch.nn as nn

EMB_DIM = 128


class Head(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(in_dim, EMB_DIM), nn.BatchNorm1d(EMB_DIM), nn.ReLU(), nn.Dropout(0.3))
        self.cls = nn.Linear(EMB_DIM, n_classes)

    def forward(self, h):
        e = self.emb(h)
        return self.cls(e), e


class MultiBranchCNN(nn.Module):
    """Osobna galaz konwolucyjna 1D dla kazdego urzadzenia (prawy, helm, lewy), fuzja przez konkatenacje."""

    def __init__(self, groups, n_classes, width=64):
        super().__init__()
        self.groups = groups

        def branch(c_in):
            layers, c = [], c_in
            for k, c_out in ((7, width), (5, width), (3, width * 2)):
                layers += [nn.Conv1d(c, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)]
                c = c_out
            return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())

        self.branches = nn.ModuleList(branch(len(g)) for g in groups)
        self.head = Head(width * 2 * len(groups), n_classes)

    def forward(self, x):                       # x: (B, T, C)
        x = x.transpose(1, 2)
        h = torch.cat([b(x[:, g]) for b, g in zip(self.branches, self.groups)], dim=1)
        return self.head(h)


class GRUNet(nn.Module):
    def __init__(self, groups, n_classes, hidden=96):
        super().__init__()
        c_in = sum(len(g) for g in groups)
        self.gru = nn.GRU(c_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = Head(4 * hidden, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)                    # (B, T, 2H)
        h = torch.cat([out.mean(1), out.max(1).values], dim=1)
        return self.head(h)


class TransformerNet(nn.Module):
    def __init__(self, groups, n_classes, d_model=64, n_layers=3, n_heads=4, patch=5):
        super().__init__()
        c_in = sum(len(g) for g in groups)
        # tokenizacja: nieprzekrywajace sie okna po 5 klatek (135 -> 27 tokenow)
        self.embed = nn.Conv1d(c_in, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, 135 // patch, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = Head(d_model, n_classes)

    def forward(self, x):
        z = self.embed(x.transpose(1, 2)).transpose(1, 2) + self.pos
        z = self.norm(self.enc(z))
        return self.head(z.mean(1))


MODELS = {"MIB-CNN": MultiBranchCNN, "GRU": GRUNet, "Transformer": TransformerNet}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
