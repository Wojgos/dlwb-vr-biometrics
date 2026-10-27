# Deep Learning w Biometrii – spotkania #1,2,3

Weryfikacja biometryczna cross-day na korpusie **VR-Biometric-Authentication** (41 uczestników, Quest / Vive / Cosmos).

## Struktura

| Plik | Zawartość |
| --- | --- |
| `data.py` | wczytywanie `DatasetsNumpy/*.npy`, zestawy cech (`pos`, `rot`, `all`, `right`, `head`, `left`), podział dzień 1 → dzień 2, standaryzacja |
| `models.py` | Multi-Input Branch CNN, GRU (dwukierunkowy), Transformer |
| `metrics.py` | podobieństwo cosinusowe, Rank-1 (NN i szablon), CMC, EER |
| `train.py` | trening wszystkich konfiguracji; zapis `train_embedding.npz`, `test_embedding.npz`, `metrics.json`, `history.json` |
| `eda.py` | eksploracyjna analiza danych + baseline bez uczenia |
| `analyze.py` | agregacja wyników, tabele LaTeX, wykresy |
| `rsa.py` | Representational Similarity Analysis |
| `report/raport.tex` | raport (PDF: `report/raport.pdf`) |

## Odtworzenie

Repozytorium danych musi leżeć obok katalogu projektu: `../VR-Biometric-Authentication/Authentication/DatasetsNumpy/`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
python eda.py        # results/figures/eda_*.pdf, results/eda_stats.json
python train.py      # 3 systemy x 3 modele x 6 zestawów cech x 3 seedy = 162 przebiegi (~1 h na RTX 3050)
python analyze.py    # results/summary_metrics.csv, results/tables/*.tex, wykresy
python rsa.py        # results/rsa_results.json, results/figures/rsa_*.pdf
cd report && pdflatex raport.tex && pdflatex raport.tex
```

Embeddingi: `results/runs/<system>/<model>/<cechy>/seed<k>/{train,test}_embedding.npz`
(klucze: `embedding` [N×128], `label`, `rep`, `day`).
