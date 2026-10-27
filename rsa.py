"""Representational Similarity Analysis na embeddingach zapisanych w train_embedding.npz / test_embedding.npz.

RDM (representational dissimilarity matrix) = 1 - podobienstwo cosinusowe miedzy probkami.
Podobienstwo dwoch reprezentacji = korelacja Spearmana gornych trojkatow ich RDM.
"""
import json
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from data import SYSTEMS, N_USERS, cross_day_split
from metrics import l2n

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["MIB-CNN", "GRU", "Transformer"]
FEATS = ["pos", "rot", "all", "right", "head", "left"]
FEAT_PL = {"pos": "pozycje", "rot": "rotacje", "all": "poz+rot", "right": "prawy", "head": "hełm", "left": "lewy"}
SEEDS = [0, 1, 2]


def load(system, model, feat, seed, split="test"):
    z = np.load(RUNS / system / model / feat / f"seed{seed}" / f"{split}_embedding.npz")
    return z["embedding"], z["label"]


def rdm(e):
    e = l2n(e)
    return 1 - e @ e.T


def centroid_rdm(e, y):
    c = np.stack([l2n(e[y == u]).mean(0) for u in range(N_USERS)])
    return rdm(c)


def upper(m):
    return m[np.triu_indices(len(m), 1)]


def spearman_matrix(vecs):
    r = np.stack([rankdata(v) for v in vecs])
    r = (r - r.mean(1, keepdims=True)) / r.std(1, keepdims=True)
    return r @ r.T / r.shape[1]


def identity_auc(d, different):
    """AUC zgodnosci RDM z idealnym RDM tozsamosci: P(d(rozne osoby) > d(ta sama osoba)).

    Rownowazne (po normalizacji) korelacji rangowej z binarnym RDM, ale bez sufitu wynikajacego
    z silnej nierownowagi par (tylko ~2% par to ta sama osoba).
    """
    r = rankdata(d)
    n_pos, n_neg = different.sum(), (~different).sum()
    return float((r[different].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def raw_embedding(system, feat, split="test"):
    d, _ = cross_day_split(system, feat)
    return d[split]["x"].reshape(len(d[split]["x"]), -1)


def classical_mds(D, k=2):
    """Klasyczne MDS (Torgerson) dla macierzy odleglosci D."""
    n = len(D); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    i = np.argsort(w)[::-1][:k]
    return v[:, i] * np.sqrt(np.maximum(w[i], 0))


def heat(ax, M, labels, title, vmin=0, vmax=1):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=8); ax.grid(False)
    return im


def main():
    out = {}
    configs = [(m, f) for m in MODELS for f in FEATS]
    names = [f"{m}|{FEAT_PL[f]}" for m, f in configs] + ["surowe|poz+rot"]

    # 1. RSA miedzy konfiguracjami (proby testowe, dzien 2), srednia po seedach; osobno dla kazdego systemu
    rsa_sys, ident_corr = {}, {}
    for s in SYSTEMS:
        mats, aucs = [], []
        for seed in SEEDS:
            vecs = []
            for m, f in configs:
                e, y = load(s, m, f, seed)
                vecs.append(upper(rdm(e)))
            vecs.append(upper(rdm(raw_embedding(s, "all"))))
            ideal = upper(y[:, None] != y[None])                            # idealny RDM tozsamosci (binarny)
            mats.append(spearman_matrix(vecs))
            aucs.append([identity_auc(v, ideal) for v in vecs])
        rsa_sys[s] = np.mean(mats, 0)
        ident_corr[s] = dict(zip(names, np.mean(aucs, 0).tolist()))
    out["rsa_between_configs"] = {s: dict(labels=names, matrix=v.tolist()) for s, v in rsa_sys.items()}
    out["rsa_vs_identity"] = ident_corr

    Mavg = np.mean([rsa_sys[s] for s in SYSTEMS], 0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = heat(ax, Mavg, names, "RSA (Spearman RDM, dzień 2), średnia po systemach i seedach")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{Mavg[i, j]:.2f}"[1:] if i != j else "", ha="center", va="center", fontsize=4.5,
                    color="w" if Mavg[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "rsa_configs.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for k, (ax, s) in enumerate(zip(axes, SYSTEMS)):
        im = heat(ax, rsa_sys[s], names, s)
        ax.set_xticks([])
        if k: ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG / "rsa_configs_per_system.pdf", bbox_inches="tight"); plt.close(fig)

    # MDS konfiguracji na podstawie (1 - RSA), usrednione po systemach
    xy = classical_mds(1 - Mavg)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    mk = {"MIB-CNN": "s", "GRU": "o", "Transformer": "^", "surowe": "*"}
    col = {"pozycje": "C0", "rotacje": "C1", "poz+rot": "C2", "prawy": "C3", "hełm": "C4", "lewy": "C5"}
    for (x_, y_), n in zip(xy, names):
        m, f = n.split("|")
        ax.scatter(x_, y_, marker=mk[m], color=col[f], s=60 if m != "surowe" else 160, edgecolor="k", lw=0.4)
    for m, mm in mk.items(): ax.scatter([], [], marker=mm, color="gray", label=m)
    for f, c in col.items(): ax.scatter([], [], marker="s", color=c, label=f)
    ax.legend(fontsize=6, ncol=2); ax.set_title("MDS przestrzeni reprezentacji (odległość = 1 − RSA)")
    fig.tight_layout(); fig.savefig(FIG / "rsa_mds.pdf"); plt.close(fig)

    # 2. Korelacja z idealnym RDM tozsamosci - wykres slupkowy
    fig, ax = plt.subplots(figsize=(8, 2.8))
    w = 0.27
    for j, s in enumerate(SYSTEMS):
        ax.bar(np.arange(len(names)) + (j - 1) * w, [ident_corr[s][n] for n in names], w, label=s)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=70, fontsize=6)
    ax.set_ylabel("AUC zgodności z RDM tożsamości"); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=7)
    ax.set_title("Na ile geometria embeddingów odzwierciedla tożsamość uczestników (dzień 2)")
    fig.tight_layout(); fig.savefig(FIG / "rsa_identity.pdf"); plt.close(fig)

    # 3. Stabilnosc miedzydniowa: RDM centroidow uzytkownikow dzien 1 vs dzien 2
    stab = {}
    for s, (m, f) in product(SYSTEMS, configs):
        vals = []
        for seed in SEEDS:
            etr, ytr = load(s, m, f, seed, "train"); ete, yte = load(s, m, f, seed, "test")
            vals.append(spearman_matrix([upper(centroid_rdm(etr, ytr)), upper(centroid_rdm(ete, yte))])[0, 1])
        stab[f"{s}|{m}|{f}"] = float(np.mean(vals))
    for s in SYSTEMS:
        d, _ = cross_day_split(s, "all")
        stab[f"{s}|surowe|all"] = float(spearman_matrix([upper(centroid_rdm(d["train"]["x"].reshape(410, -1), d["train"]["y"])),
                                                         upper(centroid_rdm(d["test"]["x"].reshape(410, -1), d["test"]["y"]))])[0, 1])
    out["cross_day_stability"] = stab

    # 4. Zgodnosc miedzy systemami VR: RDM centroidow uzytkownikow (ci sami 41 uczestnikow)
    xsys = {}
    pairs = [("Quest", "Vive"), ("Quest", "Cosmos"), ("Vive", "Cosmos")]
    for m, f in configs:
        for a, b in pairs:
            vals = []
            for seed in SEEDS:
                ea, ya = load(a, m, f, seed); eb, yb = load(b, m, f, seed)
                vals.append(spearman_matrix([upper(centroid_rdm(ea, ya)), upper(centroid_rdm(eb, yb))])[0, 1])
            xsys[f"{m}|{f}|{a}-{b}"] = float(np.mean(vals))
    for a, b in pairs:
        ea, eb = raw_embedding(a, "all"), raw_embedding(b, "all")
        y = np.repeat(np.arange(N_USERS), 10)
        xsys[f"surowe|all|{a}-{b}"] = float(spearman_matrix([upper(centroid_rdm(ea, y)), upper(centroid_rdm(eb, y))])[0, 1])
    out["cross_system_rsa"] = xsys

    # 5. Powtarzalnosc reprezentacji miedzy seedami (ten sam model, rozne inicjalizacje)
    seed_rsa = {}
    for s, (m, f) in product(SYSTEMS, configs):
        vecs = [upper(rdm(load(s, m, f, seed)[0])) for seed in SEEDS]
        M = spearman_matrix(vecs)
        seed_rsa[f"{s}|{m}|{f}"] = float(M[np.triu_indices(len(SEEDS), 1)].mean())
    out["seed_rsa"] = seed_rsa

    # 6. Przykładowe RDM (Quest, poz+rot, seed 0) posortowane wg uzytkownika
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, m in zip(axes, MODELS + ["surowe"]):
        e = raw_embedding("Quest", "all") if m == "surowe" else load("Quest", m, "all", 0)[0]
        im = ax.imshow(rdm(e), cmap="magma", vmin=0, vmax=1.2 if m != "surowe" else 1.6)
        ax.set_title(f"RDM: {m}, Quest, dzień 2", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "rsa_rdm_examples.pdf"); plt.close(fig)

    (ROOT / "results" / "rsa_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    print("vs identity:", json.dumps(ident_corr, indent=1, ensure_ascii=False))
    print("cross-day stability:", json.dumps(stab, indent=1))
    print("cross-system:", json.dumps(xsys, indent=1))
    print("seed rsa:", json.dumps(seed_rsa, indent=1))
    print("RSA avg over systems:\n", np.array2string(Mavg, precision=2, max_line_width=250), names)


if __name__ == "__main__":
    main()
