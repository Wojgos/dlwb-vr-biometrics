"""Metryki weryfikacji/identyfikacji oparte na podobienstwie cosinusowym."""
import numpy as np
from sklearn.metrics import roc_curve


def l2n(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def cosine_matrix(probe, gallery):
    return l2n(probe) @ l2n(gallery).T


def templates(emb, y):
    classes = np.unique(y)
    return np.stack([l2n(emb[y == c]).mean(0) for c in classes]), classes


def evaluate(train_emb, train_y, test_emb, test_y, max_rank=10):
    """Galeria = dzien 1, proby = dzien 2.

    rank1_nn  - najblizszy pojedynczy wzorzec z galerii (sample-to-sample),
    rank1_tpl - najblizszy szablon uzytkownika (srednia znormalizowanych embeddingow),
    cmc       - krzywa CMC dla szablonow,
    eer       - Equal Error Rate weryfikacji 1:1 (proba vs. szablon deklarowanego uzytkownika).
    """
    s = cosine_matrix(test_emb, train_emb)
    rank1_nn = float((train_y[s.argmax(1)] == test_y).mean())

    tpl, classes = templates(train_emb, train_y)
    st = cosine_matrix(test_emb, tpl)                       # (N_test, 41)
    order = classes[np.argsort(-st, axis=1)]
    hit = order == test_y[:, None]
    cmc = np.cumsum(hit, axis=1).clip(0, 1).mean(0)[:max_rank]

    genuine = st[np.arange(len(test_y)), np.searchsorted(classes, test_y)]
    mask = classes[None, :] != test_y[:, None]
    impostor = st[mask]
    labels = np.r_[np.ones_like(genuine), np.zeros_like(impostor)]
    fpr, tpr, _ = roc_curve(labels, np.r_[genuine, impostor])
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)
    return dict(rank1_nn=rank1_nn, rank1_tpl=float(cmc[0]), rank5_tpl=float(cmc[4]), eer=eer, cmc=cmc.tolist(),
                genuine_mean=float(genuine.mean()), impostor_mean=float(impostor.mean()))
