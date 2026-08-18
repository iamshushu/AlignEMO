"""ASF with the beta-shift alpha rescaling removed, on the CORRECTED attention.

Completes the 2x2 that separates the two changes made to HMF:

                        alpha gate ON            alpha gate OFF
    legacy attention    70.83 (job 10954)        job 10975
    fixed  attention    job 10965                this variant

The legacy-side counterpart lives at
EACL2TelME/EACL/TelME/IEMOCAP/asf_nogate.py and removes the same term, so the two
rows differ only in attention and the two columns only in the gate.

Relative to model.py:ASF a single line changes:
    ASF         embedding_output = LayerNorm(alpha * h_m + text),  alpha = min(||text||/||h_m|| * beta, 1)
    ASF_NoGate  embedding_output = LayerNorm(        h_m + text)
The ReLU gate `weight_av` is kept, matching `hmf_nogate` in
scripts/ablation/fusion_variants.py -- "nogate" there names the norm-based
rescaling, not that ReLU.
"""
import torch
import torch.nn.functional as F

from model import ASF


class ASF_NoGate(ASF):
    def forward(self, text_embedding, *, visual, acoustic):
        new_nv = self.pairwise_attn(visual, acoustic)

        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        # --- gate ablation: the alpha rescaling of ASF is dropped here ---
        embedding_output = self.dropout(
            self.LayerNorm(h_m + text_embedding)
        )

        logits = self.W(embedding_output)
        return embedding_output, logits
