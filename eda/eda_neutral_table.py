"""
Neutral focal 의 사용자 요청 표 (Sad table 과 같은 format) 생성.
diversity_detail_*.csv 와 density 계산을 합성.
"""
import os
import pandas as pd

OUT = os.environ.get("ALIGNEMO_OUT_DIR", "eda/outputs")

# density bin 결과 (이미 출력된 eda_sad_density.py 의 Neutral 결과 hard-coded)
DENSITY = {
    "IEMOCAP": dict(in_dlg_pct=89.4, bin_0_5=8.9, bin_50plus=14.8, mean_ratio=24.67, median_ratio=18.60),
    "MELD":    dict(in_dlg_pct=92.6, bin_0_5=0.0, bin_50plus=56.3, mean_ratio=52.09, median_ratio=50.00),
    "K-MIND":  dict(in_dlg_pct=100.0, bin_0_5=0.0, bin_50plus=85.1, mean_ratio=65.22, median_ratio=66.48),
}

rows = []
for ds in ["IEMOCAP", "MELD", "K-MIND"]:
    # diversity_detail 에서 neutral-dominant 정확 계산
    d = pd.read_csv(f"{OUT}/diversity_detail_{ds}.csv")
    neu_dom = (d['dominant'] == 'neutral').sum()
    total = len(d)
    neu_dom_pct = neu_dom / total * 100

    rows.append(dict(
        Dataset=ds,
        Neutral_in_dlg_pct=DENSITY[ds]["in_dlg_pct"],
        Neutral_dominant_pct=round(neu_dom_pct, 1),
        density_0_5pct=DENSITY[ds]["bin_0_5"],
        density_50plus_pct=DENSITY[ds]["bin_50plus"],
        mean_ratio=DENSITY[ds]["mean_ratio"],
        median_ratio=DENSITY[ds]["median_ratio"],
    ))

df = pd.DataFrame(rows)
print("\n=== Neutral table (Sad 와 같은 format) ===\n")
print(df.to_string(index=False))
print()
df.to_csv(f"{OUT}/neutral_table.csv", index=False)
print(f"[OK] saved: {OUT}/neutral_table.csv")
