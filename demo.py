"""Demo na zajecia.

1) Bez treningu (kilka sekund): przelicza Rank-1 i EER z zapisanych embeddingow .npz
   python demo.py
2) Trening na zywo jednej konfiguracji (~15 s na GPU, ~1-2 min na CPU) + przykladowe identyfikacje
   python demo.py --train --system Vive --model GRU --features all
"""
import argparse
from pathlib import Path
import numpy as np

from data import SYSTEMS
from metrics import evaluate, l2n

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
MODELS = ["MIB-CNN", "GRU", "Transformer"]


def table_from_saved(features=("pos", "rot", "all")):
    print("Rank-1 [%] / EER [%] przeliczone z train_embedding.npz i test_embedding.npz (srednia z 3 seedow)\n")
    print(f"{'cechy':6s} {'model':12s}" + "".join(f"{s:>16s}" for s in SYSTEMS))
    for f in features:
        for m in MODELS:
            cells = []
            for s in SYSTEMS:
                r1, eer = [], []
                for d in sorted((RUNS / s / m / f).glob("seed*")):
                    tr, te = np.load(d / "train_embedding.npz"), np.load(d / "test_embedding.npz")
                    res = evaluate(tr["embedding"], tr["label"], te["embedding"], te["label"])
                    r1.append(res["rank1_nn"]); eer.append(res["eer"])
                cells.append(f"{100 * np.mean(r1):6.1f} / {100 * np.mean(eer):4.1f}")
            print(f"{f:6s} {m:12s}" + "".join(f"{c:>16s}" for c in cells))
        print()


def live(system, model, features, epochs, seed):
    import train
    train.HP["epochs"] = epochs
    out = ROOT / "results" / "demo" / system / model / features / f"seed{seed}"
    print(f"Trening: {system}, {model}, cechy={features}, {epochs} epok, urzadzenie={train.DEV}")
    m = train.run(system, model, features, seed, out)
    print(f"\nRank-1 (NN) = {100 * m['rank1_nn']:.1f}%   Rank-1 (szablon) = {100 * m['rank1_tpl']:.1f}%   "
          f"EER = {100 * m['eer']:.1f}%   czas = {m['train_time_s']:.0f} s")
    print(f"Embeddingi zapisane w: {out}")

    tr, te = np.load(out / "train_embedding.npz"), np.load(out / "test_embedding.npz")
    sim = l2n(te["embedding"]) @ l2n(tr["embedding"]).T
    rng = np.random.default_rng(0)
    print("\nPrzykladowe identyfikacje (proba z dnia 2 -> najblizsza proba z dnia 1):")
    for i in rng.choice(len(te["label"]), 8, replace=False):
        j = sim[i].argmax()
        ok = "OK " if tr["label"][j] == te["label"][i] else "ZLE"
        print(f"  [{ok}] uczestnik {te['label'][i]:2d}, rzut {te['rep'][i]} -> uczestnik {tr['label'][j]:2d} "
              f"(cos = {sim[i, j]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--system", default="Vive", choices=SYSTEMS)
    ap.add_argument("--model", default="GRU", choices=MODELS)
    ap.add_argument("--features", default="all", choices=["pos", "rot", "all", "right", "head", "left"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.train:
        live(a.system, a.model, a.features, a.epochs, a.seed)
    else:
        table_from_saved()
