"""Fusion-module variants for the AlignEMO fusion ablation (CMC revision, M4).

All variants share the HMF/ASF interface:
    forward(text_embedding, visual, acoustic) -> (fused_hidden[768], logits[clsNum])
so they are drop-in replacements for the final HMF fusion in fusion.py.

Variants
  concat      : concat(text, visual, acoustic) -> MLP
  crossattn   : text as query attends to [visual, acoustic] (key/value)
  hmf_nogate  : HMF/ASF but WITHOUT the beta-shift alpha gating (displacement
                added directly) -> isolates the gating contribution
  hmf         : the full HMF (imports IEMOCAP/model.py:ASF)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_D = 768


class PairwiseModalityAttention(nn.Module):
    """Attend over two modality tokens independently for every sample."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

    def forward(self, first, second):
        if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
            raise ValueError(
                "Expected matching [batch, hidden] tensors; got "
                f"{tuple(first.shape)} and {tuple(second.shape)}"
            )
        tokens = torch.stack((first, second), dim=1)  # [B, 2, D]
        attended, _ = self.multihead_attn(
            tokens, tokens, tokens, need_weights=False
        )
        return (attended + tokens).flatten(start_dim=1)  # [B, 2D]


class ConcatMLP(nn.Module):
    def __init__(self, clsNum, hidden_size=_D, dropout_prob=0.2, **kw):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * _D, _D), nn.ReLU(), nn.Dropout(dropout_prob))
        self.W = nn.Linear(_D, clsNum)

    def forward(self, text_embedding, *, visual, acoustic):
        h = self.mlp(torch.cat([text_embedding, visual, acoustic], dim=-1))
        return h, self.W(h)


class CrossAttnFusion(nn.Module):
    def __init__(self, clsNum, hidden_size=_D, dropout_prob=0.2, num_head=4, **kw):
        super().__init__()
        self.attn = nn.MultiheadAttention(_D, num_head, batch_first=True)
        self.ln = nn.LayerNorm(_D)
        self.drop = nn.Dropout(dropout_prob)
        self.W = nn.Linear(_D, clsNum)

    def forward(self, text_embedding, *, visual, acoustic):
        q = text_embedding.unsqueeze(1)                # [B,1,D]
        kv = torch.stack([visual, acoustic], dim=1)    # [B,2,D]
        a, _ = self.attn(q, kv, kv)                     # [B,1,D]
        h = self.ln(text_embedding + self.drop(a.squeeze(1)))
        return h, self.W(h)


class HMFNoGate(nn.Module):
    """ASF/HMF without the beta-shift alpha gating (add displacement directly)."""
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super().__init__()
        self.pairwise_attn = PairwiseModalityAttention(_D, num_head)
        self.W_hav = nn.Linear(3 * _D, _D)
        self.W_av = nn.Linear(2 * _D, _D)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(2 * _D)
        self.dropout = nn.Dropout(dropout_prob)
        self.W = nn.Linear(_D, clsNum)

    def forward(self, text_embedding, *, visual, acoustic):
        new_nv = self.pairwise_attn(visual, acoustic)
        av = self.dropout(self.AV_LayerNorm(new_nv))
        weight_av = F.relu(self.W_hav(torch.cat((av, text_embedding), dim=-1)))
        h_m = weight_av * self.W_av(av)
        # --- no gating: displacement added directly (vs alpha*h_m in ASF) ---
        out = self.dropout(self.LayerNorm(h_m + text_embedding))
        return out, self.W(out)


class ShiftAwareFusion(nn.Module):
    """HMF displacement with a *disagreement gate* (CMC revision, novel).

    Motivation (from our emotion-shift analysis: shift w-F1 58.7 vs no-shift 84.4):
    at emotion-shift utterances the text carries conversational context that leans
    toward the *previous* emotion, while prosody/expression (non-verbal) reflect the
    *current* one. So when text and the non-verbal stream disagree, a shift is likely
    and the non-verbal displacement should be trusted more. We keep the HMF base gate
    `alpha` and multiply it by (1 + gamma * disagreement), boosting non-verbal cues
    exactly when they contradict the (context-laden) text.
    """
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super().__init__()
        self.beta_shift = beta_shift
        self.pairwise_attn = PairwiseModalityAttention(_D, num_head)
        self.W_hav = nn.Linear(3 * _D, _D)
        self.W_av = nn.Linear(2 * _D, _D)
        self.AV_LayerNorm = nn.LayerNorm(2 * _D)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        # disagreement head: |proj(text) - proj(av)| -> scalar in [0,1]
        self.proj_t = nn.Linear(_D, _D)
        self.proj_av = nn.Linear(_D, _D)
        self.disagree = nn.Sequential(nn.Linear(_D, _D // 2), nn.ReLU(), nn.Linear(_D // 2, 1))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.W = nn.Linear(_D, clsNum)

    def forward(self, text_embedding, *, visual, acoustic):
        eps = 1e-6
        new_nv = self.pairwise_attn(visual, acoustic)
        av = self.dropout(self.AV_LayerNorm(new_nv))
        av_d = self.W_av(av)
        h_m = F.relu(self.W_hav(torch.cat((av, text_embedding), dim=-1))) * av_d

        # HMF base gate
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        ones = torch.ones(em_norm.shape, device=text_embedding.device)
        alpha = torch.min((em_norm / (hm_norm + eps)) * self.beta_shift, ones).unsqueeze(-1)

        # disagreement between text and non-verbal -> shift signal
        disagree = torch.sigmoid(self.disagree(torch.abs(self.proj_t(text_embedding) - self.proj_av(av_d))))
        gate = alpha * (1.0 + self.gamma * disagree)

        out = self.dropout(self.LayerNorm(gate * h_m + text_embedding))
        return out, self.W(out)


def build_fusion(variant, clsNum, hidden_size=_D, beta_shift=0.2, dropout_prob=0.2, num_head=4):
    if variant == "concat":
        return ConcatMLP(clsNum, hidden_size, dropout_prob)
    if variant == "crossattn":
        return CrossAttnFusion(clsNum, hidden_size, dropout_prob, num_head)
    if variant == "hmf_nogate":
        return HMFNoGate(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    if variant == "shiftaware":
        return ShiftAwareFusion(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    if variant == "hmf":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "IEMOCAP"))
        from model import ASF
        return ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    raise ValueError(f"unknown fusion variant: {variant}")
