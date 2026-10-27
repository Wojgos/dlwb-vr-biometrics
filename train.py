"""Eksperyment cross-day: trening na dniu 1, weryfikacja na dniu 2.

Uruchomienie (pelna siatka):
    python train.py
Pojedyncza konfiguracja:
    python train.py --systems Quest --models GRU --features all --seeds 0
Dla kazdej konfiguracji zapisywane sa train_embedding.npz, test_embedding.npz, metrics.json, history.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import SYSTEMS, FEATURE_SETS, N_USERS, cross_day_split
from models import MODELS, n_params
from metrics import evaluate

ROOT = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HP = dict(epochs=150, batch_size=32, lr=1e-3, weight_decay=1e-2, label_smoothing=0.1,
          noise_std=0.05, max_shift=10, scale_jitter=0.1)


def augment(x):
    """Lekka augmentacja: losowe przesuniecie w czasie, skalowanie amplitudy, szum gaussowski."""
    b, t, c = x.shape
    shifts = torch.randint(-HP["max_shift"], HP["max_shift"] + 1, (b,), device=x.device)
    idx = (torch.arange(t, device=x.device)[None] - shifts[:, None]).clamp(0, t - 1)
    x = torch.gather(x, 1, idx[..., None].expand(-1, -1, c))
    x = x * (1 + HP["scale_jitter"] * (torch.rand(b, 1, c, device=x.device) * 2 - 1))
    return x + HP["noise_std"] * torch.randn_like(x)


@torch.no_grad()
def embed(model, x, bs=256):
    model.eval()
    out = [model(torch.from_numpy(x[i:i + bs]).to(DEV))[1].cpu().numpy() for i in range(0, len(x), bs)]
    return np.concatenate(out)


def run(system, model_name, feature_set, seed, out_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    data, groups = cross_day_split(system, feature_set)
    xtr = torch.from_numpy(data["train"]["x"]).to(DEV)
    ytr = torch.from_numpy(data["train"]["y"]).to(DEV)
    model = MODELS[model_name](groups, N_USERS).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    steps_per_epoch = int(np.ceil(len(xtr) / HP["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, HP["lr"], total_steps=HP["epochs"] * steps_per_epoch)

    hist, t0 = [], time.time()
    for ep in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(len(xtr), device=DEV)
        tot, correct = 0.0, 0
        for i in range(0, len(xtr), HP["batch_size"]):
            b = perm[i:i + HP["batch_size"]]
            logits, _ = model(augment(xtr[b]))
            loss = F.cross_entropy(logits, ytr[b], label_smoothing=HP["label_smoothing"])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b); correct += (logits.argmax(1) == ytr[b]).sum().item()
        hist.append(dict(epoch=ep + 1, loss=tot / len(xtr), train_acc=correct / len(xtr)))

    e_tr, e_te = embed(model, data["train"]["x"]), embed(model, data["test"]["x"])
    m = evaluate(e_tr, data["train"]["y"], e_te, data["test"]["y"])
    with torch.no_grad():
        logits = model(torch.from_numpy(data["test"]["x"]).to(DEV))[0].argmax(1).cpu().numpy()
    m.update(softmax_acc=float((logits == data["test"]["y"]).mean()), train_time_s=time.time() - t0,
             n_params=n_params(model), system=system, model=model_name, features=feature_set, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train_embedding.npz", embedding=e_tr, label=data["train"]["y"], rep=data["train"]["rep"], day=1)
    np.savez(out_dir / "test_embedding.npz", embedding=e_te, label=data["test"]["y"], rep=data["test"]["rep"], day=2)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    (out_dir / "history.json").write_text(json.dumps(hist))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--features", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "hyperparams.json").write_text(json.dumps(HP, indent=2))
    for seed in a.seeds:
        for system in a.systems:
            for feat in a.features:
                for model in a.models:
                    d = Path(a.out) / system / model / feat / f"seed{seed}"
                    if a.skip_existing and (d / "metrics.json").exists():
                        continue
                    m = run(system, model, feat, seed, d)
                    print(f"{system:7s} {model:12s} {feat:6s} s{seed}  rank1_nn={m['rank1_nn']:.3f} "
                          f"rank1_tpl={m['rank1_tpl']:.3f} eer={m['eer']:.3f} ({m['train_time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
