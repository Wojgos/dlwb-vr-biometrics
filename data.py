"""Wczytywanie korpusu VR-Biometric-Authentication i wybor cech."""
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "VR-Biometric-Authentication" / "Authentication" / "DatasetsNumpy"
SYSTEMS = ["Quest", "Vive", "Cosmos"]
DEVICES = ["right", "head", "left"]          # kolejnosc w tablicy (po 7 kolumn na urzadzenie)
N_USERS, N_REPS, N_T = 41, 10, 135

# indeksy w ostatnim wymiarze: [px,py,pz, ex,ey,ez, trigger] x 3 urzadzenia
POS_IDX = {d: [7 * i + 0, 7 * i + 1, 7 * i + 2] for i, d in enumerate(DEVICES)}
ROT_IDX = {d: [7 * i + 3, 7 * i + 4, 7 * i + 5] for i, d in enumerate(DEVICES)}

# Zestawy cech: modalnosci (ablacja z polecenia) + podzbiory urzadzen (dodatkowa ablacja)
FEATURE_SETS = {
    "pos": dict(mods=("pos",), devices=DEVICES),
    "rot": dict(mods=("rot",), devices=DEVICES),
    "all": dict(mods=("pos", "rot"), devices=DEVICES),
    "right": dict(mods=("pos", "rot"), devices=["right"]),
    "head": dict(mods=("pos", "rot"), devices=["head"]),
    "left": dict(mods=("pos", "rot"), devices=["left"]),
}


def load_session(system, session):
    return np.load(DATA_DIR / f"{system}{session}.npy").astype(np.float32)


def build_features(raw, feature_set):
    """raw: (..., T, 21) -> (..., T, C) oraz lista grup kanalow per urzadzenie.

    Pozycje zostaja w metrach, katy Eulera (rad, przeskok +-pi) kodujemy jako sin/cos,
    zeby uniknac nieciaglosci. Nacisk spustu pomijamy - nie nalezy do zadnej modalnosci.
    """
    cfg = FEATURE_SETS[feature_set]
    cols, groups, c = [], [], 0
    for dev in cfg["devices"]:
        start = c
        if "pos" in cfg["mods"]:
            cols.append(raw[..., POS_IDX[dev]]); c += 3
        if "rot" in cfg["mods"]:
            ang = raw[..., ROT_IDX[dev]]
            cols += [np.sin(ang), np.cos(ang)]; c += 6
        groups.append(list(range(start, c)))
    return np.concatenate(cols, axis=-1).astype(np.float32), groups


def cross_day_split(system, feature_set):
    """Dzien 1 (sesja 1) -> trening/galeria, dzien 2 (sesja 2) -> test/proby."""
    out = {}
    for split, sess in (("train", 1), ("test", 2)):
        raw = load_session(system, sess)                      # (41,10,135,21)
        x, groups = build_features(raw, feature_set)
        out[split] = dict(
            x=x.reshape(-1, N_T, x.shape[-1]),
            y=np.repeat(np.arange(N_USERS), N_REPS),
            rep=np.tile(np.arange(N_REPS), N_USERS),
        )
    # standaryzacja wylacznie statystykami ze zbioru treningowego
    mu = out["train"]["x"].mean((0, 1), keepdims=True)
    sd = out["train"]["x"].std((0, 1), keepdims=True) + 1e-6
    for s in out.values():
        s["x"] = (s["x"] - mu) / sd
    return out, groups
