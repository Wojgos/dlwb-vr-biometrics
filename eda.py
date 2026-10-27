"""Eksploracyjna analiza danych (EDA) + bazowy klasyfikator bez uczenia (cosinus na surowych cechach)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import SYSTEMS, DEVICES, POS_IDX, ROT_IDX, FEATURE_SETS, N_USERS, N_REPS, N_T, load_session, cross_day_split
from metrics import evaluate

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
DEV_PL = {"right": "prawy kontroler", "head": "hełm", "left": "lewy kontroler"}
SYS_COL = {"Quest": "#1f77b4", "Vive": "#d62728", "Cosmos": "#2ca02c"}

raw = {(s, k): load_session(s, k) for s in SYSTEMS for k in (1, 2)}
stats = {}

# 1. Podsumowanie korpusu
summary = []
for (s, k), a in raw.items():
    summary.append(dict(session=f"{s}{k}", shape=list(a.shape), n_samples=int(a.shape[0] * a.shape[1]),
                        nan=int(np.isnan(a).sum()), const_trigger_left=bool((a[..., 20] == 0).all()),
                        right_trigger_mean=float(a[..., 6].mean())))
stats["summary"] = summary
prior = np.load(ROOT.parent / "VR-Biometric-Authentication/Authentication/DatasetsNumpy/Prior1.npy")
stats["prior_nonzero_users"] = int((np.abs(prior).sum((1, 2, 3)) > 0).sum())

# 2. Rozklad klas (liczba powtorzen na uzytkownika / sesje)
counts = np.array([[raw[(s, k)].shape[1] for s in SYSTEMS for k in (1, 2)] for _ in range(N_USERS)])
fig, ax = plt.subplots(figsize=(7, 2.2))
ax.bar(np.arange(N_USERS), counts.sum(1), color="#555")
ax.set_xlabel("ID uczestnika"); ax.set_ylabel("liczba nagrań"); ax.set_title("Rozkład klas: 41 uczestników × 3 systemy × 2 dni × 10 rzutów")
ax.set_ylim(0, 70); fig.tight_layout(); fig.savefig(FIG / "eda_class_dist.pdf"); plt.close(fig)

# 3. Przykladowe trajektorie prawego kontrolera (dzien 1 vs dzien 2)
fig, axes = plt.subplots(3, 3, figsize=(8, 6.5), sharex=True)
t = np.arange(N_T) / 45
for r, s in enumerate(SYSTEMS):
    for c, u in enumerate([0, 7, 21]):
        ax = axes[r, c]
        for k, ls in ((1, "-"), (2, "--")):
            for rep in range(N_REPS):
                ax.plot(t, raw[(s, k)][u, rep, :, 1], ls, lw=0.7, color="C0" if k == 1 else "C3", alpha=0.6)
        ax.set_title(f"{s}, uczestnik {u}", fontsize=8)
        if c == 0: ax.set_ylabel("wysokość prawego\nkontrolera y [m]")
        if r == 2: ax.set_xlabel("czas [s]")
axes[0, 0].plot([], [], "-", color="C0", label="dzień 1"); axes[0, 0].plot([], [], "--", color="C3", label="dzień 2")
axes[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "eda_trajectories.pdf"); plt.close(fig)

# 4. Srednia trajektoria 3D (x-y) dla kilku uzytkownikow - widok z boku
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, s in zip(axes, SYSTEMS):
    for u in range(6):
        m = raw[(s, 1)][u, :, :, :].mean(0)
        ax.plot(m[:, 0], m[:, 1], lw=1.2, label=f"u{u}")
    ax.set_title(f"{s} (dzień 1) – prawy kontroler"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
axes[0].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "eda_xy_paths.pdf"); plt.close(fig)

# 5. Rozklady pozycji w zaleznosci od systemu i dnia
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, dev in zip(axes, DEVICES):
    data, labels, cols = [], [], []
    for s in SYSTEMS:
        for k in (1, 2):
            data.append(raw[(s, k)][..., POS_IDX[dev][1]].mean(2).ravel()); labels.append(f"{s[0]}{k}"); cols.append(SYS_COL[s])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], cols): p.set_facecolor(c); p.set_alpha(0.5)
    ax.set_title(f"{DEV_PL[dev]}: średnia wysokość y"); ax.set_xlabel("system/dzień")
axes[0].set_ylabel("y [m]")
fig.tight_layout(); fig.savefig(FIG / "eda_pos_by_system.pdf"); plt.close(fig)

# 6. Rozklad katow Eulera (pokazuje przeskok +-pi -> kodowanie sin/cos)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
for ax, dev in zip(axes, DEVICES):
    for j, n in enumerate("xyz"):
        ax.hist(raw[("Quest", 1)][..., ROT_IDX[dev][j]].ravel(), bins=80, histtype="step", label=f"kąt {n}", density=True)
    ax.set_title(f"Quest1 – {DEV_PL[dev]}"); ax.set_xlabel("kąt [rad]")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "eda_euler_hist.pdf"); plt.close(fig)

# 7. Spust prawego kontrolera - moment chwytu/wypuszczenia pilki
fig, ax = plt.subplots(figsize=(5, 2.6))
release = {}
for s in SYSTEMS:
    trig = np.concatenate([raw[(s, 1)][..., 6], raw[(s, 2)][..., 6]]).reshape(-1, N_T)
    ax.plot(t, trig.mean(0), label=s, color=SYS_COL[s])
    rel = np.array([np.argmax(tr < 0.5) if (tr < 0.5).any() else N_T for tr in trig]) / 45
    release[s] = dict(mean=float(rel.mean()), std=float(rel.std()))
ax.set_xlabel("czas [s]"); ax.set_ylabel("średni nacisk spustu"); ax.set_title("Prawy spust: chwyt i wypuszczenie piłki"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "eda_trigger.pdf"); plt.close(fig)
stats["release_time_s"] = release

# 8. Wspolczynnik Fishera: zmiennosc miedzy- vs wewnatrzosobnicza dla cech usrednionych w czasie
def fisher(a):                          # a: (41,10,135,C) -> (C,)
    m = a.mean(2)                       # srednia w czasie
    between = m.mean(1).var(0)
    within = m.var(1).mean(0)
    return between / (within + 1e-9)

ch_names = [f"{d[0].upper()}-{n}" for d in DEVICES for n in ["px", "py", "pz", "ex", "ey", "ez", "trg"]]
fis = np.stack([fisher(np.concatenate([raw[(s, 1)], raw[(s, 2)]], 1)) for s in SYSTEMS])
keep = [i for i in range(21) if i not in (13, 20)]
fig, ax = plt.subplots(figsize=(8, 2.6))
w = 0.27
for j, s in enumerate(SYSTEMS):
    ax.bar(np.arange(len(keep)) + (j - 1) * w, fis[j, keep], w, label=s, color=SYS_COL[s])
ax.set_xticks(np.arange(len(keep))); ax.set_xticklabels([ch_names[i] for i in keep], rotation=60, fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("F = σ²między / σ²wewnątrz"); ax.legend(fontsize=7)
ax.set_title("Separowalność uczestników w poszczególnych kanałach (cechy uśrednione w czasie, oba dni)")
fig.tight_layout(); fig.savefig(FIG / "eda_fisher.pdf"); plt.close(fig)
stats["fisher"] = {s: {ch_names[i]: float(fis[j, i]) for i in keep} for j, s in enumerate(SYSTEMS)}

# 9. Dryf miedzydniowy: odleglosc srednich pozycji u tego samego uzytkownika vs. miedzy uzytkownikami
drift = {}
fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
for ax, s in zip(axes, SYSTEMS):
    d1, d2 = raw[(s, 1)], raw[(s, 2)]
    for dev in DEVICES:
        p1 = d1[..., POS_IDX[dev]].mean((1, 2)); p2 = d2[..., POS_IDX[dev]].mean((1, 2))
        D = np.linalg.norm(p1[:, None] - p2[None], axis=-1)
        same, diff = np.diag(D), D[~np.eye(N_USERS, dtype=bool)]
        drift[f"{s}-{dev}"] = dict(same_user=float(same.mean()), other_user=float(diff.mean()),
                                   rank1_mean_pos=float((D.argmin(1) == np.arange(N_USERS)).mean()))
        ax.hist(same, bins=20, alpha=0.5, density=True, label=f"{DEV_PL[dev]}: ten sam")
    ax.set_title(f"{s}: dryf średniej pozycji dzień1→dzień2"); ax.set_xlabel("odległość [m]")
axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG / "eda_drift.pdf"); plt.close(fig)
stats["drift"] = drift

# 10. PCA surowych sekwencji (dzien 1 i 2) dla 8 uzytkownikow
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, s in zip(axes, SYSTEMS):
    data, _ = cross_day_split(s, "all")
    X = np.concatenate([data["train"]["x"], data["test"]["x"]]).reshape(2 * N_USERS * N_REPS, -1)
    y = np.r_[data["train"]["y"], data["test"]["y"]]; day = np.r_[np.ones(410), 2 * np.ones(410)]
    Z = PCA(2).fit_transform(X)
    for u in range(8):
        for dd, mk in ((1, "o"), (2, "x")):
            m = (y == u) & (day == dd)
            ax.scatter(Z[m, 0], Z[m, 1], s=12, marker=mk, color=f"C{u}")
    ax.set_title(f"{s}: PCA (o = dzień 1, x = dzień 2)", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "eda_pca.pdf"); plt.close(fig)

# 11. Baseline bez uczenia: cosinus na spłaszczonych, standaryzowanych sekwencjach
baseline = {}
for s in SYSTEMS:
    for fs in FEATURE_SETS:
        data, _ = cross_day_split(s, fs)
        m = evaluate(data["train"]["x"].reshape(410, -1), data["train"]["y"], data["test"]["x"].reshape(410, -1), data["test"]["y"])
        baseline[f"{s}|{fs}"] = {k: m[k] for k in ("rank1_nn", "rank1_tpl", "eer")}
stats["baseline_raw_cosine"] = baseline

(ROOT / "results" / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf8")
print(json.dumps({k: v for k, v in stats.items() if k != "fisher"}, indent=1, ensure_ascii=False))
