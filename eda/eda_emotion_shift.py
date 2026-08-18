"""
세 데이터셋의 dialogue-level emotion shift 통계.

각 dialogue 에 대해 계산:
  - N (총 발화 수)
  - total_shifts (인접 두 발화의 emotion 다른 경우)
  - shift_rate = total_shifts / (N-1)
  - non_neu_shifts (neutral↔neutral 제외, 즉 실제 정서 변화만)
  - same_spk_shifts (같은 화자의 인접 turn 만 — 화자 자신의 mood 변화)

출력: 데이터셋별 요약 + per-dialogue distribution.

실행:
  cd <repo-root>
  python eda_emotion_shift.py
"""
import pandas as pd
from eda_compare_datasets import load_meld, load_iemocap, load_kmind, KMIND_EN, IEMO_FULL, OUT

DATASETS = ["IEMOCAP", "MELD", "K-MIND"]


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
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["dialogue_id", "turn"]).reset_index(drop=True)
    df["emo_en"] = df["emotion"].map(lambda e: en(ds, e))
    return df


def analyze_shifts(df, ds_name):
    """Per-dialogue emotion shift 통계."""
    rows = []
    for dlg_id, g in df.groupby("dialogue_id"):
        g = g.sort_values("turn").reset_index(drop=True)
        N = len(g)
        if N < 2:
            continue
        emos = g["emo_en"].tolist()
        speakers = g["speaker"].tolist()

        # 1) Total shifts (mixed, 화자 무관)
        shifts = sum(1 for i in range(N - 1) if emos[i] != emos[i + 1])

        # 2a) Inter-affective shifts (양쪽 모두 non-neutral, 즉 진짜 emotion ↔ emotion)
        inter_affective_shifts = sum(
            1 for i in range(N - 1)
            if emos[i] != emos[i + 1] and emos[i] != "neutral" and emos[i + 1] != "neutral"
        )
        # 2b) Neutral-involved shifts (한쪽이 neutral 인 transition)
        neutral_shifts = sum(
            1 for i in range(N - 1)
            if emos[i] != emos[i + 1] and (emos[i] == "neutral" or emos[i + 1] == "neutral")
        )
        non_neu_shifts = inter_affective_shifts  # 호환 위해 유지

        # 3) Same-speaker shifts (같은 화자가 다음 turn 에서 emotion 변경)
        same_spk_pairs = sum(1 for i in range(N - 1) if speakers[i] == speakers[i + 1])
        same_spk_shifts = sum(
            1 for i in range(N - 1)
            if speakers[i] == speakers[i + 1] and emos[i] != emos[i + 1]
        )

        # 4) Cross-speaker shifts (서로 다른 화자 사이의 emotion 변화)
        cross_spk_pairs = (N - 1) - same_spk_pairs
        cross_spk_shifts = sum(
            1 for i in range(N - 1)
            if speakers[i] != speakers[i + 1] and emos[i] != emos[i + 1]
        )

        rows.append(dict(
            dialogue_id=dlg_id, N=N,
            total_shifts=shifts,
            inter_affective_shifts=inter_affective_shifts,
            neutral_shifts=neutral_shifts,
            non_neu_shifts=non_neu_shifts,
            same_spk_pairs=same_spk_pairs,
            same_spk_shifts=same_spk_shifts,
            cross_spk_pairs=cross_spk_pairs,
            cross_spk_shifts=cross_spk_shifts,
            shift_rate=shifts / (N - 1) if N > 1 else 0,
        ))

    df_out = pd.DataFrame(rows)
    return df_out


def summarize(df_out, ds_name):
    """데이터셋 요약 print."""
    print(f"\n{'='*70}")
    print(f"[{ds_name}]  #dialogues={len(df_out)},  avg N/dlg={df_out['N'].mean():.1f}")
    print(f"{'='*70}")

    print(f"\n  Total shifts (mixed, 화자 무관):")
    print(f"    avg shifts/dlg     = {df_out['total_shifts'].mean():>6.2f}")
    print(f"    median shifts/dlg  = {df_out['total_shifts'].median():>6.0f}")
    print(f"    avg shift rate     = {df_out['shift_rate'].mean()*100:>6.1f}%  (shifts / (N-1))")

    print(f"\n  Shift type 분해:")
    ia = df_out['inter_affective_shifts'].sum()
    nv = df_out['neutral_shifts'].sum()
    tot = df_out['total_shifts'].sum()
    print(f"    Inter-affective shifts (emotion↔emotion, 양쪽 non-neutral):")
    print(f"      avg/dlg = {df_out['inter_affective_shifts'].mean():>6.2f}  ({ia}, 전체 shift 의 {ia/max(tot,1)*100:.1f}%)")
    print(f"    Neutral-involved shifts (한쪽이 neutral):")
    print(f"      avg/dlg = {df_out['neutral_shifts'].mean():>6.2f}  ({nv}, 전체 shift 의 {nv/max(tot,1)*100:.1f}%)")

    same_pairs = df_out['same_spk_pairs'].sum()
    same_shifts = df_out['same_spk_shifts'].sum()
    cross_pairs = df_out['cross_spk_pairs'].sum()
    cross_shifts = df_out['cross_spk_shifts'].sum()
    print(f"\n  Same-speaker shifts (화자 자신의 mood 변화):")
    print(f"    avg same-spk shifts/dlg = {df_out['same_spk_shifts'].mean():>6.2f}")
    print(f"    same-spk shift rate     = {same_shifts / max(same_pairs,1) * 100:>5.1f}%  ({same_shifts}/{same_pairs} pairs)")

    print(f"\n  Cross-speaker shifts (다른 화자 응답에서 emotion 변경):")
    print(f"    avg cross-spk shifts/dlg = {df_out['cross_spk_shifts'].mean():>6.2f}")
    print(f"    cross-spk shift rate     = {cross_shifts / max(cross_pairs,1) * 100:>5.1f}%  ({cross_shifts}/{cross_pairs} pairs)")

    # Distribution
    print(f"\n  Shift count 분포 (per dialogue):")
    bins = [0, 1, 5, 10, 20, 50, 1000]
    labels = ["0", "1-4", "5-9", "10-19", "20-49", "50+"]
    cuts = pd.cut(df_out['total_shifts'], bins, labels=labels, right=False)
    dist = cuts.value_counts().sort_index()
    for lab, cnt in dist.items():
        bar = "█" * int(cnt / max(dist.max(),1) * 30)
        print(f"    {lab:<8s}: {cnt:>4d}  {bar}")

    return df_out


def main():
    print("로딩...")
    all_rows = load_meld() + load_iemocap() + load_kmind()
    dfs = {ds: get_df(all_rows, ds) for ds in DATASETS}

    summary_rows = []
    for ds in DATASETS:
        df_out = analyze_shifts(dfs[ds], ds)
        summarize(df_out, ds)
        summary_rows.append(dict(
            dataset=ds,
            n_dialogues=len(df_out),
            avg_N_per_dlg=round(df_out['N'].mean(), 1),
            avg_total_shifts=round(df_out['total_shifts'].mean(), 2),
            avg_shift_rate_pct=round(df_out['shift_rate'].mean() * 100, 1),
            avg_non_neu_shifts=round(df_out['non_neu_shifts'].mean(), 2),
            avg_same_spk_shifts=round(df_out['same_spk_shifts'].mean(), 2),
            avg_cross_spk_shifts=round(df_out['cross_spk_shifts'].mean(), 2),
        ))
        # detail CSV
        df_out.to_csv(f"{OUT}/shift_detail_{ds}.csv", index=False)

    # 요약 표
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUT}/shift_summary.csv", index=False)
    print(f"\n\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\n[OK] saved: {OUT}/shift_summary.csv, {OUT}/shift_detail_<DS>.csv")


if __name__ == "__main__":
    main()
