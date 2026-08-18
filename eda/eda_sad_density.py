"""
Per-dialogue Sad density (length-normalized).

"Dominant" metric 은 K-MIND 같은 긴 dialogue (avg 88턴) 에서 sparse class 에
구조적으로 불리. 대신:
  - sad_ratio = (sad 발화 수) / (전체 발화 수)  → length-normalized
  - 분포: 0%, 0-5%, 5-10%, 10-25%, 25-50%, 50%+
  - median / mean / max ratio

각 데이터셋에서 sad 가 dialogue 의 몇 % 를 차지하는지를 length-normalized 로 비교.

실행:
  python eda_sad_density.py
"""
import pandas as pd
from eda_compare_datasets import load_meld, load_iemocap, load_kmind, KMIND_EN, IEMO_FULL, OUT

DATASETS = ["IEMOCAP", "MELD", "K-MIND"]

# 분석할 focal class 리스트. label naming 이 dataset 마다 다른 경우 dict 사용.
FOCAL_GROUPS = {
    "Sad":     {"IEMOCAP": "sad",     "MELD": "sadness", "K-MIND": "sad"},
    "Neutral": {"IEMOCAP": "neutral", "MELD": "neutral", "K-MIND": "neutral"},
    "Joy/Happy": {"IEMOCAP": "happy", "MELD": "joy",     "K-MIND": "joy"},
    "Anger":   {"IEMOCAP": "anger",   "MELD": "anger",   "K-MIND": "anger"},
}


def en(ds, emo):
    if ds == "K-MIND":
        return KMIND_EN.get(emo, emo)
    if ds == "IEMOCAP":
        return IEMO_FULL.get(emo, emo)
    return emo


def get_df(all_rows, ds):
    df = pd.DataFrame([r for r in all_rows if r["dataset"] == ds]).copy()
    df = df[df["emotion"].astype(str).str.strip() != ""]
    df = df[df["emotion"].notna()]
    df["emo_en"] = df["emotion"].map(lambda e: en(ds, e))
    return df


def analyze_density(df, ds_name, focal):
    print(f"\n{'='*70}")
    print(f"[{ds_name}]  focal = '{focal}'")
    print(f"{'='*70}")

    # 각 dialogue 의 (total turns, sad count, sad ratio)
    rows = []
    for dlg_id, g in df.groupby("dialogue_id"):
        N = len(g)
        n_sad = (g["emo_en"] == focal).sum()
        sad_ratio = n_sad / N if N > 0 else 0
        rows.append(dict(dialogue_id=dlg_id, N=N, n_sad=n_sad, sad_ratio=sad_ratio))
    d = pd.DataFrame(rows)

    # 전체 dialogue 통계
    print(f"\n[전체 {len(d)} dialogue 의 통계]")
    print(f"  평균 dialogue 길이 N = {d['N'].mean():.1f}")
    print(f"  Sad 가 1번 이상 등장한 dialogue = {(d['n_sad'] > 0).sum()} ({(d['n_sad']>0).mean()*100:.1f}%)")

    # Sad ratio 분포 (sad-containing dialogue 만)
    sad_d = d[d['n_sad'] > 0].copy()
    if len(sad_d) > 0:
        print(f"\n[Sad-containing dialogue (n={len(sad_d)}) 의 sad ratio 분포]")
        print(f"  평균   sad_ratio = {sad_d['sad_ratio'].mean()*100:>5.2f}%")
        print(f"  중앙   sad_ratio = {sad_d['sad_ratio'].median()*100:>5.2f}%")
        print(f"  최대   sad_ratio = {sad_d['sad_ratio'].max()*100:>5.2f}%")
        print(f"  평균   Sad 발화 수/대화 = {sad_d['n_sad'].mean():>5.2f}")
        print(f"  중앙   Sad 발화 수/대화 = {sad_d['n_sad'].median():>5.0f}")
        print(f"  최대   Sad 발화 수/대화 = {sad_d['n_sad'].max()}")

        # bin 분포
        print(f"\n  Sad ratio 구간별 분포:")
        bins = [0, 0.05, 0.10, 0.25, 0.50, 1.01]
        labels = ["0–5%", "5–10%", "10–25%", "25–50%", "50%+"]
        cuts = pd.cut(sad_d['sad_ratio'], bins, labels=labels, right=False, include_lowest=True)
        dist = cuts.value_counts().sort_index()
        total = len(sad_d)
        for lab, cnt in dist.items():
            bar = "█" * int(cnt / max(dist.max(),1) * 30)
            print(f"    {lab:<8s}: {cnt:>4d}  ({cnt/total*100:>5.1f}%)  {bar}")

    return d


def main():
    print("로딩...")
    all_rows = load_meld() + load_iemocap() + load_kmind()
    dfs = {ds: get_df(all_rows, ds) for ds in DATASETS}

    all_summaries = []
    for focal_name, ds_to_label in FOCAL_GROUPS.items():
        print(f"\n\n{'#'*70}")
        print(f"# Focal class group: {focal_name}")
        print(f"{'#'*70}")

        summary = []
        for ds in DATASETS:
            focal = ds_to_label[ds]
            d = analyze_density(dfs[ds], ds, focal)
            d['dataset'] = ds
            d.to_csv(f"{OUT}/density_{focal_name.replace('/','_')}_{ds}.csv", index=False)

            sad_d = d[d['n_sad'] > 0]
            summary.append(dict(
                focal=focal_name,
                dataset=ds,
                avg_N=round(d['N'].mean(), 1),
                n_dlg_with_focal=len(sad_d),
                pct_dlg_with_focal=round((d['n_sad']>0).mean()*100, 1),
                mean_ratio_pct=round(sad_d['sad_ratio'].mean()*100, 2) if len(sad_d) else 0,
                median_ratio_pct=round(sad_d['sad_ratio'].median()*100, 2) if len(sad_d) else 0,
                max_ratio_pct=round(sad_d['sad_ratio'].max()*100, 2) if len(sad_d) else 0,
                mean_n_per_dlg=round(sad_d['n_sad'].mean(), 2) if len(sad_d) else 0,
            ))
        all_summaries.extend(summary)
        print(f"\n[Summary — {focal_name}]")
        print(pd.DataFrame(summary).to_string(index=False))

    sdf = pd.DataFrame(all_summaries)
    sdf.to_csv(f"{OUT}/density_summary_all.csv", index=False)
    print(f"\n\n{'='*70}\nFULL SUMMARY (all focal classes)\n{'='*70}")
    print(sdf.to_string(index=False))
    print(f"\n[OK] saved: {OUT}/density_summary_all.csv")


if __name__ == "__main__":
    main()
