"""
대화 내 emotion diversity 분석.

각 dialogue 에 대해:
  - distinct emotion 개수 (1=단일 감정 대화, 7=모든 감정 등장)
  - 분포: 몇 개 emotion 짜리 대화가 얼마나 있나
  - dominant emotion + secondary emotion
  - Co-occurrence matrix (어떤 emotion 쌍이 자주 같이 등장하나)

특히 focal class 분석:
  - Sad 가 포함된 dialogue 에서 함께 등장하는 다른 emotion 분포
  - 이게 K-MIND Sad 의 "isolated minority" 가설 검증

실행:
  cd <repo-root>
  python eda_emotion_diversity.py
"""
import pandas as pd
from collections import Counter
from itertools import combinations
from eda_compare_datasets import load_meld, load_iemocap, load_kmind, KMIND_EN, IEMO_FULL, OUT

DATASETS = ["IEMOCAP", "MELD", "K-MIND"]
# focal class per dataset (label naming 다름)
FOCAL_PER_DS = {"IEMOCAP": "sad", "MELD": "sadness", "K-MIND": "sad"}


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


def analyze_diversity(df, ds_name):
    """Per-dialogue emotion diversity 분석."""
    print(f"\n{'='*70}")
    print(f"[{ds_name}]  #dialogues={df['dialogue_id'].nunique()}")
    print(f"{'='*70}")
    focal = FOCAL_PER_DS.get(ds_name, "sad")

    # 1) 대화당 distinct emotion 개수
    dlg_diversity = df.groupby("dialogue_id")["emo_en"].agg(
        n_distinct="nunique",
        n_utterances="size",
        emotions=lambda s: tuple(sorted(set(s))),
        dominant=lambda s: s.value_counts().idxmax(),
        dominant_count=lambda s: s.value_counts().iloc[0],
    ).reset_index()

    print(f"\n[Distinct emotion count per dialogue]")
    print(f"  평균   = {dlg_diversity['n_distinct'].mean():.2f}")
    print(f"  중앙   = {dlg_diversity['n_distinct'].median():.0f}")
    print(f"  최소/최대 = {dlg_diversity['n_distinct'].min()} / {dlg_diversity['n_distinct'].max()}")

    print(f"\n  분포:")
    dist = dlg_diversity['n_distinct'].value_counts().sort_index()
    total = len(dlg_diversity)
    for n_emo, cnt in dist.items():
        bar = "█" * int(cnt / max(dist.max(),1) * 30)
        print(f"    {n_emo} emotion  : {cnt:>5d}  ({cnt/total*100:>5.1f}%)  {bar}")

    # 2) Dominant emotion 비율
    print(f"\n[Dominant emotion share per dialogue]")
    dlg_diversity['dom_pct'] = dlg_diversity['dominant_count'] / dlg_diversity['n_utterances']
    print(f"  평균 dominant 비율 = {dlg_diversity['dom_pct'].mean()*100:.1f}%")
    print(f"  (전체 dialogue 의 평균: dominant emotion 이 차지하는 발화 비율)")

    # 단일 감정 대화 (dominant 100%)
    single_emo = (dlg_diversity['n_distinct'] == 1).sum()
    print(f"  단일 감정 대화 (dominant 100%) = {single_emo} ({single_emo/total*100:.1f}%)")
    # neutral-only / sad-only / etc.
    if single_emo > 0:
        single_dist = dlg_diversity[dlg_diversity['n_distinct'] == 1]['dominant'].value_counts()
        for emo, cnt in single_dist.items():
            print(f"     {emo}: {cnt}")

    # 3) Focal emotion (Sad) 분석
    print(f"\n[Focal class = '{focal}' co-occurrence]")
    sad_dlgs = dlg_diversity[dlg_diversity['emotions'].apply(lambda t: focal in t)]
    n_sad_dlgs = len(sad_dlgs)
    print(f"  Sad 가 포함된 dialogue 개수 = {n_sad_dlgs} ({n_sad_dlgs/total*100:.1f}%)")

    if n_sad_dlgs > 0:
        # 이 dialogue 들에서 Sad 가 dominant 인 비율
        sad_dominant = (sad_dlgs['dominant'] == focal).sum()
        print(f"  그 중 Sad 가 dominant emotion 인 비율 = {sad_dominant}/{n_sad_dlgs} ({sad_dominant/n_sad_dlgs*100:.1f}%)")

        # Sad 와 함께 등장하는 다른 emotion 빈도
        co_emos = Counter()
        for emos_tuple in sad_dlgs['emotions']:
            for e in emos_tuple:
                if e != focal:
                    co_emos[e] += 1
        print(f"\n  Sad 와 함께 등장하는 emotion (frequency, 같은 대화에 등장한 횟수):")
        for emo, cnt in co_emos.most_common():
            pct = cnt / n_sad_dlgs * 100
            bar = "█" * int(pct / 3)
            print(f"     {emo:<12s} {cnt:>4d}  ({pct:>5.1f}% of sad-dialogues)  {bar}")

    # 4) Co-occurrence (모든 emotion pair)
    print(f"\n[All-pair co-occurrence — top 10]")
    pair_count = Counter()
    for emos_tuple in dlg_diversity['emotions']:
        for a, b in combinations(emos_tuple, 2):
            pair_count[tuple(sorted([a, b]))] += 1
    for (a, b), cnt in pair_count.most_common(10):
        print(f"     {a:<12s} + {b:<12s} {cnt:>5d}  ({cnt/total*100:>5.1f}% of dialogues)")

    return dlg_diversity


def main():
    print("로딩...")
    all_rows = load_meld() + load_iemocap() + load_kmind()
    dfs = {ds: get_df(all_rows, ds) for ds in DATASETS}

    summary = []
    for ds in DATASETS:
        d = analyze_diversity(dfs[ds], ds)
        d['dataset'] = ds
        d.to_csv(f"{OUT}/diversity_detail_{ds}.csv", index=False)

        # focal Sad (label name 이 dataset 마다 다름)
        focal = FOCAL_PER_DS.get(ds, "sad")
        sad_dlgs = d[d['emotions'].apply(lambda t: focal in t)]
        n_sad = len(sad_dlgs)
        n_total = len(d)
        sad_dominant = (sad_dlgs['dominant'] == focal).sum() if n_sad > 0 else 0

        summary.append(dict(
            dataset=ds,
            n_dlg=n_total,
            avg_distinct_emo=round(d['n_distinct'].mean(), 2),
            median_distinct_emo=int(d['n_distinct'].median()),
            single_emotion_pct=round((d['n_distinct'] == 1).sum() / n_total * 100, 1),
            avg_dominant_share=round(d['dom_pct'].mean() * 100, 1),
            sad_in_dlg_pct=round(n_sad / n_total * 100, 1),
            sad_dominant_pct=round(sad_dominant / max(n_sad,1) * 100, 1),
        ))

    print(f"\n\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    sdf.to_csv(f"{OUT}/diversity_summary.csv", index=False)
    print(f"\n[OK] saved: {OUT}/diversity_summary.csv, {OUT}/diversity_detail_<DS>.csv")


if __name__ == "__main__":
    main()
