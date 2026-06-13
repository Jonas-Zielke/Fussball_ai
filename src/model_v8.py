"""
E8Net — End-to-End Football Prediction Transformer (Era 8).

Architecture (three modalities):
  (A) Static encoder  — full 57-feature combined vector (V6 features, already
      encodes both teams' relative state). → tower_hidden-dim representation.
  (B) Seq towers      — shared SeqEncoder applied to home / away sequences
      separately. Cross-attention then fuses the two team views.
  (C) Squad towers    — shared SquadEncoder applied per team, fused similarly.

All three are concatenated with a match context vector and fed to heads.

Heads:
  1. Poisson    → (log_λ_home, log_λ_away)   — score distribution
  2. KO         → (P_et, P_pen_given_et)      — extra-time / penalty prob
  3. Penalty    → P(home wins penalty shootout)

Training note: Swap-Augmentation (swap home/away and invert labels) is applied
in train_v8.py to enforce symmetry. The static _a/_b indices are defined in
STATIC_HOME_IDX / STATIC_AWAY_IDX for easy reordering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── V6 feature index maps for swap augmentation ─────────────────────────────
# Features with _a (home) counterpart at _b (away) — ordered pairs (a_idx, b_idx)
SWAP_PAIRS: list[tuple[int, int]] = [
    (2, 3),    # elo_a, elo_b
    (5, 6),    # vr_elo_a, vr_elo_b
    (8, 9),    # re_elo_a, re_elo_b
    (11, 12),  # form1_a, form1_b
    (13, 14),  # form2
    (15, 16),  # form3
    (17, 18),  # form5
    (19, 20),  # form10
    (21, 22),  # gf5
    (23, 24),  # ga5
    (25, 26),  # gd5
    (27, 28),  # h2h_a, h2h_b
    (29, 30),  # rest_a, rest_b
    (31, 32),  # win_streak_a, win_streak_b
    (34, 35),  # continent_a, continent_b
    (36, 37),  # oppo_elo5
    (38, 39),  # w_form
    (40, 41),  # momentum
    (42, 43),  # wins_top10
    (44, 45),  # wins_top20
    (46, 47),  # stability
    (48, 49),  # sq_ovr
    (50, 51),  # sq_att
    (52, 53),  # sq_def
    (55, 56),  # sq_age
]
# Scalar diff features that flip sign on swap
DIFF_IDX: list[int] = [4, 7, 10, 54]  # elo_diff, vr_elo_diff, re_elo_diff, sq_diff


def swap_home_away(X: torch.Tensor) -> torch.Tensor:
    """Swap home/away features in static vector for augmentation."""
    X2 = X.clone()
    for a, b in SWAP_PAIRS:
        X2[:, a], X2[:, b] = X[:, b].clone(), X[:, a].clone()
    for d in DIFF_IDX:
        X2[:, d] = -X[:, d]
    return X2


# ── Hyperparameters ──────────────────────────────────────────────────────────

@dataclass
class E8Config:
    # Input dimensions
    static_dim: int = 57
    seq_len: int = 10
    seq_dim: int = 7
    n_players: int = 15
    player_dim: int = 3
    context_dim: int = 2        # (is_neutral, tournament_weight_norm)

    # Encoder widths
    tower_hidden: int = 192     # static encoder output
    seq_proj: int = 128         # sequence encoder output
    squad_proj: int = 64        # squad encoder output

    # Transformer config
    n_seq_heads: int = 4
    n_seq_layers: int = 2
    n_squad_heads: int = 4
    n_cross_heads: int = 4

    # Head
    head_hidden: int = 256
    dropout: float = 0.20


# ── Helpers ──────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class SeqEncoder(nn.Module):
    """Encodes last-K-games sequence for one team via Transformer."""
    def __init__(self, seq_dim: int, proj: int, seq_len: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(seq_dim, proj)
        self.pe = PositionalEncoding(proj, max_len=seq_len + 4, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj, nhead=n_heads, dim_feedforward=proj * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(proj)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, K, seq_dim) → (B, proj)"""
        pad = (seq.abs().sum(-1) == 0)          # (B, K) — zero rows are padding
        # PyTorch Attention produces NaN when ALL tokens are masked; always keep last
        # Use cat instead of in-place assignment for ONNX compatibility
        pad = torch.cat([pad[:, :-1], torch.zeros_like(pad[:, :1])], dim=1)
        x = F.gelu(self.proj(seq))
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).float()
        return self.norm((x * valid).sum(1) / valid.sum(1).clamp(min=1))


class SquadEncoder(nn.Module):
    """Encodes a set of player vectors (padded) via self-attention + mean pool."""
    def __init__(self, player_dim: int, proj: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(player_dim, proj)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj, nhead=n_heads, dim_feedforward=proj * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(proj)

    def forward(self, squad: torch.Tensor) -> torch.Tensor:
        """squad: (B, N, player_dim) → (B, proj)"""
        pad = (squad.abs().sum(-1) == 0)
        # Use cat instead of in-place assignment for ONNX compatibility
        pad = torch.cat([pad[:, :-1], torch.zeros_like(pad[:, :1])], dim=1)
        x = F.gelu(self.proj(squad))
        x = self.encoder(x, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).float()
        return self.norm((x * valid).sum(1) / valid.sum(1).clamp(min=1))


class CrossAttn(nn.Module):
    """Bidirectional cross-attention: each team queries the other."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_h = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """h, a: (B, d) → (B, 2d)"""
        h_q = h.unsqueeze(1)
        a_q = a.unsqueeze(1)
        h2, _ = self.attn_h(h_q, a_q, a_q)
        a2, _ = self.attn_a(a_q, h_q, h_q)
        h_out = self.norm_h(h + h2.squeeze(1))
        a_out = self.norm_a(a + a2.squeeze(1))
        return torch.cat([h_out, a_out], dim=-1)


# ── E8Net ────────────────────────────────────────────────────────────────────

@dataclass
class E8Output:
    log_lam_home: torch.Tensor    # (B,)
    log_lam_away: torch.Tensor    # (B,)
    p_et: torch.Tensor            # (B,) P(extra time)
    p_pen_given_et: torch.Tensor  # (B,) P(penalties | ET)
    p_home_pen: torch.Tensor      # (B,) P(home wins shootout)


class E8Net(nn.Module):
    def __init__(self, cfg: E8Config | None = None):
        super().__init__()
        if cfg is None:
            cfg = E8Config()
        self.cfg = cfg

        # (A) Static encoder
        self.static_enc = nn.Sequential(
            nn.Linear(cfg.static_dim, cfg.tower_hidden),
            nn.BatchNorm1d(cfg.tower_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.tower_hidden, cfg.tower_hidden),
            nn.BatchNorm1d(cfg.tower_hidden),
            nn.GELU(),
        )

        # (B) Sequence encoder — shared weights for home/away
        self.seq_enc = SeqEncoder(
            cfg.seq_dim, cfg.seq_proj, cfg.seq_len,
            cfg.n_seq_heads, cfg.n_seq_layers, cfg.dropout,
        )

        # (C) Squad encoder — shared weights for home/away
        self.squad_enc = SquadEncoder(
            cfg.player_dim, cfg.squad_proj, cfg.n_squad_heads, cfg.dropout,
        )

        # Cross-attention over per-team (seq + squad) vectors
        team_dim = cfg.seq_proj + cfg.squad_proj
        self.cross_attn = CrossAttn(team_dim, cfg.n_cross_heads, cfg.dropout)

        # Final combination: static + fused_teams + context
        fused = cfg.tower_hidden + 2 * team_dim + cfg.context_dim
        self.pre_head = nn.Sequential(
            nn.Linear(fused, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.GELU(),
        )

        self.poisson_head = nn.Linear(cfg.head_hidden, 2)   # log λ home/away
        self.ko_head = nn.Linear(cfg.head_hidden, 2)        # P(ET), P(pen|ET)
        self.pen_head = nn.Linear(cfg.head_hidden, 1)       # P(home wins pens)

    def forward(
        self,
        static_X: torch.Tensor,   # (B, static_dim) — combined V6 features
        seq_home: torch.Tensor,    # (B, K, seq_dim)
        squad_home: torch.Tensor,  # (B, N, player_dim)
        seq_away: torch.Tensor,
        squad_away: torch.Tensor,
        context: torch.Tensor,     # (B, context_dim) — [is_neutral, tourn_norm]
    ) -> E8Output:
        # (A) static
        s = self.static_enc(static_X)                          # (B, tower_hidden)

        # (B+C) per-team encode
        h_seq = self.seq_enc(seq_home)                         # (B, seq_proj)
        a_seq = self.seq_enc(seq_away)
        h_sq = self.squad_enc(squad_home)                      # (B, squad_proj)
        a_sq = self.squad_enc(squad_away)

        h_team = torch.cat([h_seq, h_sq], dim=-1)              # (B, team_dim)
        a_team = torch.cat([a_seq, a_sq], dim=-1)
        fused_teams = self.cross_attn(h_team, a_team)          # (B, 2*team_dim)

        combined = torch.cat([s, fused_teams, context], dim=-1)
        feat = self.pre_head(combined)

        # Clamp log-lambda: lambda ∈ [exp(-2.5), exp(3.0)] = [0.08, 20.1]
        log_lam = torch.clamp(self.poisson_head(feat), min=-2.5, max=3.0)
        ko = self.ko_head(feat)
        pen = self.pen_head(feat).squeeze(-1)

        return E8Output(
            log_lam_home=log_lam[:, 0],
            log_lam_away=log_lam[:, 1],
            p_et=torch.sigmoid(ko[:, 0]),
            p_pen_given_et=torch.sigmoid(ko[:, 1]),
            p_home_pen=torch.sigmoid(pen),
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: E8Config | None = None) -> E8Net:
    return E8Net(cfg or E8Config())
