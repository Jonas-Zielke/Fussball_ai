# WM 2026 Predictor

Ein vollstaendiges KI-System, das Ausgaenge von Fussball-Laenderspielen vorhersagt -
speziell trainiert fuer die **FIFA Fussball-Weltmeisterschaft 2026**.

Gebaut mit:
- **PyTorch 2.11** (CUDA 12.8) - trainiert auf NVIDIA RTX 4070 Ti Super
- **49.378 historische Laenderspiele** (1872-2026) als Datenbasis
- **Elo-Rating + Form + Head-to-Head + Heimvorteil** als Features
- **MLP-Klassifikator** (96-64-32 Hidden, BatchNorm + Dropout)

## Quickstart

```bash
# 1) venv erstellen + CUDA-PyTorch installieren (einmalig)
python -m venv venv
.\venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
.\venv\Scripts\pip install pandas scikit-learn matplotlib tabulate tqdm requests pyarrow

# 2) Daten herunterladen (einmalig)
.\venv\Scripts\python -m src.data_download

# 3) Features berechnen (einmalig, ~30s)
.\venv\Scripts\python -m src.features

# 4) Modell trainieren (einmalig, ~15s auf 4070 Ti Super)
.\venv\Scripts\python -m src.train

# 5) Prognose fuer ein Spiel
.\venv\Scripts\python -m src.predict "Germany" "Brazil"
.\venv\Scripts\python -m src.predict "Spanien" "Frankreich"   # Aliase funktionieren
.\venv\Scripts\python -m src.predict "Germany" "Brazil" --json

# 6) Mehrere Modi
.\venv\Scripts\python -m src.predict --list-top 20
.\venv\Scripts\python -m src.predict --sweep "Germany,Brazil,Argentina,France,Spain"
.\venv\Scripts\python -m src.predict --simulate-wm
.\venv\Scripts\python -m src.predict   # interaktiver Modus

# 7) Tests
.\venv\Scripts\python -m tests.test_smoke
```

## Architektur

```
Fussball_ai/
├── venv/                          <- Isolierte Python-Umgebung (CUDA)
├── data/
│   ├── raw/
│   │   ├── results.csv            <- 49.378 Spiele (Martj42 Datensatz)
│   │   └── shootouts.csv          <- Elfmeterschiessen-Historie
│   └── processed/
│       ├── features.parquet       <- 49.306 Spiele x 15 Features
│       ├── features.npz           <- Kompaktes Numpy-Format
│       └── features_meta.json     <- Statistiken
├── models/
│   ├── latest.pt                  <- Aktuelles Modell (~54 KB)
│   ├── latest_meta.json           <- Architektur + Performance
│   └── history.json               <- Training-History pro Epoche
├── src/
│   ├── data_download.py           <- Laedt Martj42 Daten
│   ├── features.py                <- Feature-Engineering (Elo, Form, H2H)
│   ├── train.py                   <- PyTorch Modell + Training
│   ├── predict.py                 <- CLI fuer Inference
│   └── team_normalize.py          <- Alias-Mapping (DE, EN, FR, ...)
├── scripts/
│   └── check_elo.py               <- Sanity-Check fuer Elo-Ratings
├── tests/
│   └── test_smoke.py              <- 13 End-to-End Tests
└── requirements.txt
```

## Features (15 Dimensionen)

| # | Feature          | Bedeutung                                                  |
|---|------------------|------------------------------------------------------------|
| 1 | `elo_home`       | Elo-Rating Team A VOR dem Spiel                            |
| 2 | `elo_away`       | Elo-Rating Team B VOR dem Spiel                            |
| 3 | `elo_diff`       | Differenz (mit +80 Heimvorteil)                            |
| 4 | `form5_home`     | Punkte pro Spiel, letzte 5 Spiele von A                    |
| 5 | `form5_away`     | Punkte pro Spiel, letzte 5 Spiele von B                    |
| 6 | `gf5_home`       | Tore/Schnitt, letzte 5 Spiele A                            |
| 7 | `ga5_home`       | Gegentore/Schnitt, letzte 5 Spiele A                       |
| 8 | `gf5_away`       | ... B                                                      |
| 9 | `ga5_away`       | ... B                                                      |
| 10| `rest_home`      | Tage seit letztem Spiel A                                  |
| 11| `rest_away`      | Tage seit letztem Spiel B                                  |
| 12| `neutral`        | 0/1 (1 = neutraler Boden wie bei Turnieren)                |
| 13| `tournament_w`   | Wichtigkeit (K-Faktor): WM=60, EM=50, Friendly=20          |
| 14| `h2h_home`       | Win-Rate A in den letzten direkten Duellen                 |
| 15| `h2h_away`       | Win-Rate B in den letzten direkten Duellen                 |

## Modell

```
Input (15) -> Linear(96) -> BatchNorm -> ReLU -> Dropout(0.25)
            -> Linear(64) -> BatchNorm -> ReLU -> Dropout(0.25)
            -> Linear(32) -> BatchNorm -> ReLU -> Dropout(0.25)
            -> Linear(3)   # 0=Draw, 1=HomeWin, 2=AwayWin
```

Training:
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Loss: CrossEntropyLoss mit Class-Weighting + Label Smoothing 0.05
- Early Stopping: Patience 10
- Split: Train (2000-2023) | Val (2024-2025)

## Performance

Auf Validation (2.446 Spiele, 2024-2025):
- **Accuracy: 54.6%** (Zufall waere 33%, "immer Heimsieg" waere ~49%)
- **Log-Loss: 0.92**
- **Brier-Score: 0.55**
- Per-Class: Draw=41%, HomeWin=62%, AwayWin=53%

Das Modell hat ~4.700 freie Parameter und ist auf der 4070 Ti Super
in **14 Sekunden** trainiert (Early Stopping nach 13 Epochen).

## Limitations & moegliche Erweiterungen

- Keine Spieler-Level-Daten (Kader, Verletzungen, Marktwert)
- Keine xG-Daten oder advanced stats
- Keine Wetter-/Austragungsort-Features
- Trainiert nur auf historischen Ergebnissen, nicht auf Quoten
- Moegliche Erweiterungen: API-Football-Integration, FiveThirtyEight-SpiL Ratings,
  socceraction/SOFA-Scraping, LSTM auf Elo-Zeitreihen, Transformer mit Attention

## Datenquellen

- **Martj42/international_results** auf GitHub (CC0-1.0) - 49.016 Spiele,
  ueber 150 Jahre Fussball-Geschichte, taeglich aktualisiert.
- Keine API-Keys, kein Login, keine Rate-Limits.
