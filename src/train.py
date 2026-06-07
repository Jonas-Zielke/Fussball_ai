"""
PyTorch-Modell + Training fuer den WM 2026 Predictor.

Architektur:
    - Input: 15 Features (elo, form, gf/ga, rest days, neutral, tournament, h2h)
    - Hidden: 2-3 Layer MLP mit BatchNorm + Dropout
    - Output: 3 Klassen (Draw, HomeWin, AwayWin)

Training:
    - Loss: CrossEntropyLoss mit Class-Weighting (Unentschieden ist selten)
    - Optimizer: AdamW mit Cosine LR Schedule
    - Early Stopping auf Validation Accuracy
    - Device: CUDA wenn verfuegbar, sonst CPU
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features import load_features, PROCESSED_DIR, get_current_team_ratings
from .team_normalize import tournament_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Model --------------------

class MatchPredictor(nn.Module):
    """MLP fuer Fussball-Match-Prediction.

    15 numerische Input-Features -> 3 Klassen-Logits (Draw/HomeWin/AwayWin).
    """

    def __init__(self, in_dim: int = 15, hidden: tuple[int, ...] = (64, 64, 32), dropout: float = 0.25, num_classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.hidden = hidden
        self.dropout = dropout
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------- Training --------------------

def _class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency Gewichte fuer CrossEntropy."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * counts + 1e-9)
    # Unentschieden soll etwas staerker gewichtet werden (sind seltene Events)
    weights[0] *= 1.15
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _split_train_val(X: np.ndarray, y: np.ndarray, dates: np.ndarray, val_start: str = "2024-01-01"):
    import pandas as pd
    d = pd.to_datetime(dates)
    mask = d < pd.Timestamp(val_start)
    return (X[mask], y[mask], X[~mask], y[~mask])


def _normalize_features(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-Feature z-score Normalisierung. Stat aus Trainingsset."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return X_train_n.astype(np.float32), X_val_n.astype(np.float32), stats


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    correct = 0
    n = 0
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += yb.size(0)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    pred = logits.argmax(axis=1)

    # Brier Score (multiclass)
    probs = _softmax_np(logits)
    brier = float(((probs - _onehot(y_true, 3)) ** 2).sum(axis=1).mean())
    # Log-loss
    eps = 1e-9
    log_loss = float(-np.log(probs[np.arange(len(y_true)), y_true] + eps).mean())
    # Top-1 Accuracy
    acc = correct / n

    # Accuracy pro Klasse
    per_class = {}
    for c in range(3):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[int(c)] = float((pred[mask] == c).mean())
    return {"accuracy": acc, "log_loss": log_loss, "brier": brier, "per_class_acc": per_class, "n": int(n)}


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def train_model(
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: tuple[int, ...] = (96, 64, 32),
    dropout: float = 0.25,
    patience: int = 10,
    train_start: str = "2000-01-01",
    val_start: str = "2024-01-01",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Trainiert das Modell und gibt Metriken + Pfad zum Modell zurueck."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(" Training")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    X_all, y_all, dates_all, _, _, feat_names = load_features("all", train_start, val_start)
    # Falls zu wenig Daten, nimm alles ab 1990
    if len(X_all) < 1000:
        X_all, y_all, dates_all, _, _, feat_names = load_features("all", "1990-01-01", val_start)

    X_tr, y_tr, X_va, y_va = _split_train_val(X_all, y_all, dates_all, val_start)
    X_tr, X_va, norm_stats = _normalize_features(X_tr, X_va)
    print(f"   Train: {len(X_tr):,} | Val: {len(X_va):,}")
    print(f"   Features: {len(feat_names)} ({', '.join(feat_names[:3])}, ...)")

    # Class weights
    cw = _class_weights(y_tr, num_classes=3)
    print(f"   Class-weights: {cw.cpu().numpy().tolist()}  (0=Draw, 1=HomeWin, 2=AwayWin)")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())

    # pin_memory nur bei CUDA
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=pin)

    model = MatchPredictor(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = []

    print(f"   Starte Training: epochs={epochs}, batch={batch_size}, lr={lr}")
    print("-" * 70)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            epoch_loss += loss.item() * yb.size(0)
            epoch_correct += (logits.argmax(dim=1) == yb).sum().item()
            epoch_n += yb.size(0)

        train_loss = epoch_loss / epoch_n
        train_acc = epoch_correct / epoch_n
        val_metrics = evaluate(model, val_loader)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_loss": val_metrics["log_loss"],
            "val_brier": val_metrics["brier"],
        })
        if verbose and (epoch <= 5 or epoch % 5 == 0 or epoch == epochs):
            print(f"   Ep {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  val_logloss={val_metrics['log_loss']:.4f}  "
                  f"brier={val_metrics['brier']:.4f}  perClass={val_metrics['per_class_acc']}")
        if val_metrics["accuracy"] > best_val_acc + 1e-5:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"   Early stopping nach {epoch} Epochen (kein Improvement seit {patience}).")
                break

    train_time = time.time() - t0
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader)

    # Speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"model_{timestamp}.pt"
    latest_path = MODELS_DIR / "latest.pt"
    meta = {
        "in_dim": X_tr.shape[1],
        "hidden": list(hidden),
        "dropout": dropout,
        "num_classes": 3,
        "feature_names": list(feat_names),
        "norm_stats": norm_stats,
        "train_start": train_start,
        "val_start": val_start,
        "best_val_acc": best_val_acc,
        "final_val": final_val,
        "epochs_run": len(history),
        "train_time_sec": train_time,
        "device": str(DEVICE),
        "class_weights": cw.cpu().numpy().tolist(),
        "history": history,
    }
    torch.save({"state_dict": best_state, "meta": meta, "model_class": "MatchPredictor"}, model_path)
    # 'latest.pt' ueberschreiben (atomar via tmp)
    tmp_latest = latest_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": best_state, "meta": meta, "model_class": "MatchPredictor"}, tmp_latest)
    tmp_latest.replace(latest_path)
    # Meta als JSON
    with open(MODELS_DIR / "latest_meta.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in meta.items() if k != "history"}, fh, indent=2, ensure_ascii=False)
    with open(MODELS_DIR / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print("-" * 70)
    print(f"   Beste Val-Accuracy: {best_val_acc:.4f}")
    print(f"   Final Val-Accuracy: {final_val['accuracy']:.4f}")
    print(f"   Final Val-LogLoss:  {final_val['log_loss']:.4f}")
    print(f"   Final Val-Brier:    {final_val['brier']:.4f}")
    print(f"   Per-Class-Acc:      {final_val['per_class_acc']}")
    print(f"   Trainingsdauer:     {train_time:.1f}s")
    print(f"   Modell gespeichert: {latest_path}")
    print("=" * 70)
    return {"model_path": str(latest_path), "meta": meta}


# -------------------- Inference helpers --------------------

def _load_latest() -> tuple[nn.Module, dict]:
    """Laedt das aktuellste Modell + Normalisierungs-Statistik."""
    bundle = torch.load(MODELS_DIR / "latest.pt", map_location=DEVICE, weights_only=False)
    meta = bundle["meta"]
    model = MatchPredictor(
        in_dim=meta["in_dim"],
        hidden=tuple(meta["hidden"]),
        dropout=meta["dropout"],
        num_classes=meta["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, meta


def _build_inference_vector(
    home: str, away: str, neutral: bool, tournament: str, today: datetime
) -> np.ndarray:
    """Baut den 15-dim Feature-Vektor fuer ein hypothetisches Spiel."""
    from .team_normalize import normalize_team_name
    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)
    state = get_current_team_ratings()
    h = state.get(home_n)
    a = state.get(away_n)
    if h is None or a is None:
        missing = []
        if h is None:
            missing.append(home_n)
        if a is None:
            missing.append(away_n)
        raise ValueError(f"Unbekanntes Team: {missing}. Verfuegbar (Top 20 nach Elo): "
                         f"{', '.join(sorted(state.keys(), key=lambda t: -state[t]['elo'])[:20])}")

    feat = {
        "elo_home": h["elo"],
        "elo_away": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "form5_home": h["form5"],
        "form5_away": a["form5"],
        "gf5_home": h["gf5"],
        "ga5_home": h["ga5"],
        "gf5_away": a["gf5"],
        "ga5_away": a["ga5"],
        "rest_home": 30,  # Annahme: Standard, koennte man aus `last_match` ableiten
        "rest_away": 30,
        "neutral": int(neutral),
        "tournament_w": tournament_weight(tournament),
        "h2h_home": 0.5,
        "h2h_away": 0.5,
    }
    # Wenn wir das letzte Match-Datum haben, koennen wir Restdays berechnen
    if h["last_match"]:
        last = datetime.fromisoformat(h["last_match"])
        feat["rest_home"] = min((today - last).days, 365)
    if a["last_match"]:
        last = datetime.fromisoformat(a["last_match"])
        feat["rest_away"] = min((today - last).days, 365)
    # H2H aus den Daten berechnen
    import pandas as pd
    raw = pd.read_csv(REPO_ROOT / "data" / "raw" / "results.csv", parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"])
    pair = raw[((raw["home_team"] == home) & (raw["away_team"] == away)) |
               ((raw["home_team"] == away) & (raw["away_team"] == home))].tail(10)
    if len(pair) > 0:
        wins_as_home = 0
        wins_as_away = 0
        for _, r in pair.iterrows():
            if r["home_team"] == home:
                if r["home_score"] > r["away_score"]:
                    wins_as_home += 1
                elif r["home_score"] < r["away_score"]:
                    wins_as_away += 1
            else:  # home war away
                if r["away_score"] > r["home_score"]:
                    wins_as_home += 1
                elif r["away_score"] < r["home_score"]:
                    wins_as_away += 1
        n = len(pair)
        feat["h2h_home"] = wins_as_home / n
        feat["h2h_away"] = wins_as_away / n

    feature_names = [
        "elo_home", "elo_away", "elo_diff",
        "form5_home", "form5_away",
        "gf5_home", "ga5_home", "gf5_away", "ga5_away",
        "rest_home", "rest_away",
        "neutral", "tournament_w",
        "h2h_home", "h2h_away",
    ]
    vec = np.array([feat[k] for k in feature_names], dtype=np.float32)
    return vec


def predict_match(
    home: str,
    away: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
    today: datetime | None = None,
) -> dict:
    """Prognose fuer ein einzelnes Spiel. Liefert Klassen-Wahrscheinlichkeiten."""
    if today is None:
        today = datetime.now()
    model, meta = _load_latest()
    norm = meta["norm_stats"]
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    from .team_normalize import normalize_team_name
    home_n = normalize_team_name(home)
    away_n = normalize_team_name(away)

    vec = _build_inference_vector(home, away, neutral, tournament, today)
    vec_n = (vec - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(vec_n).to(DEVICE).unsqueeze(0))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    # 0=Draw, 1=HomeWin, 2=AwayWin
    label_map = {0: "Unentschieden", 1: f"Sieg {home_n}", 2: f"Sieg {away_n}"}
    out = {
        "home": home_n,
        "away": away_n,
        "neutral": neutral,
        "tournament": tournament,
        "as_of": today.isoformat(),
        "probabilities": {
            "draw": float(probs[0]),
            "home_win": float(probs[1]),
            "away_win": float(probs[2]),
        },
        "labels": {
            "draw": "Unentschieden",
            "home_win": f"Sieg {home_n}",
            "away_win": f"Sieg {away_n}",
        },
        "argmax_label": label_map[int(probs.argmax())],
        "elo_home": float(vec[0]),
        "elo_away": float(vec[1]),
    }
    return out


def main() -> int:
    train_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
