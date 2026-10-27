"""Agregacja wynikow eksperymentow: tabele (CSV + LaTeX) i wykresy."""
import json
from pathlib import Path
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import SYSTEMS
from metrics import l2n, templates

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"; TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
MODELS = ["MIB-CNN", "GRU", "Transformer"]
FEAT_PL = {"pos": "pozycje", "rot": "rotacje", "all": "poz.+rot.", "right": "prawy kontr.", "head": "hełm", "left": "lewy kontr."}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

rows = [json.loads(p.read_text()) for p in RUNS.glob("*/*/*/*/metrics.json")]
base = json.loads((ROOT / "results" / "eda_stats.json").read_text(encoding="utf8"))["baseline_raw_cosine"]
print(f"{len(rows)} przebiegow")
keys = [k for k in rows[0] if k != "cmc"]
with open(ROOT / "results" / "all_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


class R(dict):
    __getattr__ = dict.__getitem__


def runs(s, m, f):
    return [r for r in rows if r["system"] == s and r["model"] == m and r["features"] == f]


AGG = {}
for s in SYSTEMS:
    for m in MODELS:
        for f in FEAT_PL:
            rr = runs(s, m, f)
            v = lambda k: np.array([r[k] for r in rr])
            AGG[(s, m, f)] = R(r1=v("rank1_nn").mean(), r1s=v("rank1_nn").std(ddof=1), r1t=v("rank1_tpl").mean(),
                               r1ts=v("rank1_tpl").std(ddof=1), r5=v("rank5_tpl").mean(), eer=v("eer").mean(),
                               eers=v("eer").std(ddof=1), sm=v("softmax_acc").mean(), gen=v("genuine_mean").mean(),
                               imp=v("impostor_mean").mean(), time=v("train_time_s").mean(), params=int(rr[0]["n_params"]),
                               cmc=np.mean([r["cmc"] for r in rr], 0))
with open(ROOT / "results" / "summary_metrics.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["system", "model", "features", "rank1_nn", "rank1_nn_std", "rank1_tpl", "rank5_tpl", "eer", "eer_std", "softmax_acc", "time_s", "params"])
    for (s, m, f), a in AGG.items():
        w.writerow([s, m, f, a.r1, a.r1s, a.r1t, a.r5, a.eer, a.eers, a.sm, a.time, a.params])


def cell(m, s, pct=True, bold=False):
    t = f"{100 * m:.1f}$\\pm${100 * s:.1f}" if pct else f"{m:.3f}"
    return f"\\textbf{{{t}}}" if bold else t


def latex_table(feats, fname, caption, label):
    lines = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\begin{tabular}{ll" + "ccc" * 1 + "ccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Rank-1 [\%] (NN, cosinus)} & \multicolumn{3}{c}{EER [\%]} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             "Cechy & Model & " + " & ".join(SYSTEMS) + " & " + " & ".join(SYSTEMS) + r" \\", r"\midrule"]
    for f in feats:
        best_r1 = {s: max(AGG[(s, m, f)].r1 for m in MODELS) for s in SYSTEMS}
        best_eer = {s: min(AGG[(s, m, f)].eer for m in MODELS) for s in SYSTEMS}
        for i, m in enumerate(MODELS + ["surowe"]):
            name = FEAT_PL[f] if i == 0 else ""
            c1, c2 = [], []
            for s in SYSTEMS:
                if m == "surowe":
                    b = base[f"{s}|{f}"]
                    c1.append(f"{100 * b['rank1_nn']:.1f}"); c2.append(f"{100 * b['eer']:.1f}")
                else:
                    r = AGG[(s, m, f)]
                    c1.append(cell(r.r1, r.r1s, bold=r.r1 == best_r1[s])); c2.append(cell(r.eer, r.eers, bold=r.eer == best_eer[s]))
            mname = r"\textit{surowe cechy}" if m == "surowe" else m
            lines.append(f"{name} & {mname} & " + " & ".join(c1 + c2) + r" \\")
        lines.append(r"\midrule" if f != feats[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    (TAB / fname).write_text("\n".join(lines), encoding="utf8")


latex_table(["pos", "rot", "all"], "ablation_modality.tex",
            r"Eksperyment ablacyjny (modalności), weryfikacja cross-day: trening dzień 1, test dzień 2. Średnia $\pm$ odch. std. z 3 seedów. Pogrubiono najlepszy model w danej kolumnie.",
            "tab:abl_mod")
latex_table(["right", "head", "left"], "ablation_devices.tex",
            r"Dodatkowa ablacja urządzeń (pozycje + rotacje pojedynczego urządzenia), cross-day, 3 seedy.",
            "tab:abl_dev")

# tabela zbiorcza: Rank-1 NN vs szablon, Rank-5, softmax, czas, parametry (poz.+rot.)
lines = [r"\begin{table}[H]\centering\small",
         r"\caption{Pełny zestaw modalności (poz.+rot.): porównanie wariantów metryki i kosztu trenowania (średnia z 3 seedów).}\label{tab:full}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{llcccccr}", r"\toprule",
         r"System & Model & Rank-1 NN & Rank-1 szablon & Rank-5 szablon & Acc. softmax & $\bar{s}_{gen}$ / $\bar{s}_{imp}$ & Czas [s] \\", r"\midrule"]
for s in SYSTEMS:
    for m in MODELS:
        r = AGG[(s, m, "all")]
        lines.append(f"{s if m == MODELS[0] else ''} & {m} & {100 * r.r1:.1f} & {100 * r.r1t:.1f} & {100 * r.r5:.1f} & {100 * r.sm:.1f} & "
                     f"{r.gen:.2f} / {r.imp:.2f} & {r.time:.0f} \\\\")
    lines.append(r"\midrule" if s != SYSTEMS[-1] else r"\bottomrule")
lines += [r"\end{tabular}}", r"\end{table}"]
(TAB / "full_features.tex").write_text("\n".join(lines), encoding="utf8")

params = {m: {f: AGG[("Quest", m, f)].params for f in FEAT_PL} for m in MODELS}
(TAB / "params.json").write_text(json.dumps(params, indent=1))
print(params)

# wykresy ablacji
for feats, fname, title in ((["pos", "rot", "all"], "ablation_modality.pdf", "Ablacja modalności"),
                            (["right", "head", "left", "all"], "ablation_devices.pdf", "Ablacja urządzeń")):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, s in zip(axes, SYSTEMS):
        w = 0.2
        for j, m in enumerate(MODELS + ["surowe"]):
            if m == "surowe":
                vals = [base[f"{s}|{f}"]["rank1_nn"] for f in feats]; errs = None
            else:
                vals = [AGG[(s, m, f)].r1 for f in feats]; errs = [AGG[(s, m, f)].r1s for f in feats]
            ax.bar(np.arange(len(feats)) + (j - 1.5) * w, vals, w, yerr=errs, capsize=2, label=m,
                   color=f"C{j}" if m != "surowe" else "#aaa")
        ax.set_xticks(range(len(feats))); ax.set_xticklabels([FEAT_PL[f] for f in feats])
        ax.set_title(f"{s}"); ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Rank-1 (cross-day)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, fontsize=7)
    fig.suptitle(title, x=0.1, ha="left"); fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)

# krzywe CMC (poz.+rot.)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
for ax, s in zip(axes, SYSTEMS):
    for j, m in enumerate(MODELS):
        c = AGG[(s, m, "all")].cmc
        ax.plot(np.arange(1, len(c) + 1), c, "o-", ms=3, label=m)
    ax.set_title(s); ax.set_xlabel("ranga k"); ax.set_xticks(range(1, 11))
axes[0].set_ylabel("CMC (szablony)"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "cmc.pdf"); plt.close(fig)

# krzywe uczenia
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
for j, m in enumerate(MODELS):
    for s, ls in zip(SYSTEMS, ["-", "--", ":"]):
        h = json.loads((RUNS / s / m / "all" / "seed0" / "history.json").read_text())
        h = R(epoch=[x["epoch"] for x in h], loss=[x["loss"] for x in h], train_acc=[x["train_acc"] for x in h])
        axes[0].plot(h.epoch, h.loss, ls, color=f"C{j}", lw=1, label=f"{m}" if s == "Quest" else None)
        axes[1].plot(h.epoch, h.train_acc, ls, color=f"C{j}", lw=1)
axes[0].set_title("Strata treningowa (linia: Quest –, Vive --, Cosmos :)"); axes[0].set_xlabel("epoka"); axes[0].legend(fontsize=7)
axes[1].set_title("Dokładność treningowa (z augmentacją)"); axes[1].set_xlabel("epoka")
fig.tight_layout(); fig.savefig(FIG / "training_curves.pdf"); plt.close(fig)

# rozklady podobienstw genuine/impostor + t-SNE (Quest, poz.+rot., seed 0)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
for ax, m in zip(axes, MODELS):
    tr = np.load(RUNS / "Quest" / m / "all" / "seed0" / "train_embedding.npz")
    te = np.load(RUNS / "Quest" / m / "all" / "seed0" / "test_embedding.npz")
    tpl, cls = templates(tr["embedding"], tr["label"])
    st = l2n(te["embedding"]) @ l2n(tpl).T
    g = st[np.arange(len(st)), te["label"]]; mask = np.ones_like(st, bool); mask[np.arange(len(st)), te["label"]] = False
    ax.hist(st[mask], bins=60, density=True, alpha=0.6, label="impostor"); ax.hist(g, bins=40, density=True, alpha=0.6, label="genuine")
    ax.set_title(f"{m} (Quest, dzień 2 vs szablon dnia 1)", fontsize=8); ax.set_xlabel("podobieństwo cosinusowe")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "score_hist.pdf"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, m in zip(axes, MODELS):
    tr = np.load(RUNS / "Quest" / m / "all" / "seed0" / "train_embedding.npz")
    te = np.load(RUNS / "Quest" / m / "all" / "seed0" / "test_embedding.npz")
    E = l2n(np.concatenate([tr["embedding"], te["embedding"]])); y = np.r_[tr["label"], te["label"]]; d = np.r_[np.ones(410), 2 * np.ones(410)]
    Z = PCA(2).fit_transform(E)
    for u in range(12):
        for dd, mk in ((1, "o"), (2, "x")):
            sel = (y == u) & (d == dd)
            ax.scatter(Z[sel, 0], Z[sel, 1], s=10, marker=mk, color=plt.cm.tab20(u))
    sel = y >= 12
    ax.scatter(Z[sel, 0], Z[sel, 1], s=3, color="#ddd", zorder=0)
    ax.set_title(f"PCA embeddingów: {m} (o = dzień 1, x = dzień 2)", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(FIG / "emb_pca.pdf"); plt.close(fig)

# per-user Rank-1 (czy bledy skupiaja sie na kilku osobach?)
per_user = {}
for s in SYSTEMS:
    accs = []
    for m in MODELS:
        tr = np.load(RUNS / s / m / "all" / "seed0" / "train_embedding.npz"); te = np.load(RUNS / s / m / "all" / "seed0" / "test_embedding.npz")
        sim = l2n(te["embedding"]) @ l2n(tr["embedding"]).T
        ok = tr["label"][sim.argmax(1)] == te["label"]
        accs.append([ok[te["label"] == u].mean() for u in range(41)])
    per_user[s] = np.mean(accs, 0)
fig, ax = plt.subplots(figsize=(9, 2.5))
for j, s in enumerate(SYSTEMS):
    ax.bar(np.arange(41) + (j - 1) * 0.27, per_user[s], 0.27, label=s)
ax.set_xlabel("ID uczestnika"); ax.set_ylabel("Rank-1 (śr. 3 modeli)"); ax.legend(fontsize=7)
ax.set_title("Rank-1 per uczestnik (poz.+rot., seed 0)"); ax.set_xlim(-1, 41)
fig.tight_layout(); fig.savefig(FIG / "per_user.pdf"); plt.close(fig)
worst = {s: [int(i) for i in np.argsort(per_user[s])[:5]] for s in SYSTEMS}
worst_acc = {s: [float(per_user[s][i]) for i in worst[s]] for s in SYSTEMS}

print("worst users:", worst, worst_acc)
for (sy, m, f), a in AGG.items():
    print(f"{sy:7s} {m:12s} {f:6s} r1={a.r1:.3f}+-{a.r1s:.3f} r1t={a.r1t:.3f} r5={a.r5:.3f} eer={a.eer:.3f} sm={a.sm:.3f} t={a.time:.0f} p={a.params}")
for f in FEAT_PL:
    print(f, {m: round(np.mean([AGG[(sy, m, f)].r1 for sy in SYSTEMS]), 3) for m in MODELS},
          {sy: round(np.mean([AGG[(sy, m, f)].r1 for m in MODELS]), 3) for sy in SYSTEMS})
