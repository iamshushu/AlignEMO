"""
3개의 t-SNE PNG를 가로로 합쳐 (a)(b)(c) 라벨 붙이고 PDF/PNG 로 저장.
원본 t-SNE PNG를 키워 그린 뒤 (make_anchor_tsne_final.py) 합칩니다.

출력: confusion_figs/tsne_all.pdf , .png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

OUTDIR = 'confusion_figs'
os.makedirs(OUTDIR, exist_ok=True)

FILES = [
    'anchor_tsne/iemocap/anchor_tsne_iemocap_final.png',
    'anchor_tsne/meld/anchor_tsne_meld_final.png',
    'anchor_tsne/74.61/anchor_tsne_kmind_final.png',
]
LABELS = ['(a) IEMOCAP', '(b) MELD', '(c) K-MIND']

for f in FILES:
    if not os.path.exists(f):
        raise FileNotFoundError(f"missing: {f}")

fig, axes = plt.subplots(1, 3, figsize=(42, 16))
for ax, f, lab in zip(axes, FILES, LABELS):
    ax.imshow(mpimg.imread(f))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(lab, fontsize=48, fontweight='bold', labelpad=20)

fig.tight_layout()
out_pdf = os.path.join(OUTDIR, 'tsne_all.pdf')
out_png = os.path.join(OUTDIR, 'tsne_all.png')
fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
fig.savefig(out_png, bbox_inches='tight', dpi=300)
print(f"saved -> {out_pdf}")
print(f"saved -> {out_png}")
