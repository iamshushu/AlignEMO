"""
K-MIND test set 의 모든 슬픔 (Sad) 발화에 대해 prev/next turn 의 emotion 분포를
자동 집계.

추가 분석:
  - prev turn emotion 분포 (모든 화자 mixed)
  - next turn emotion 분포 (모든 화자 mixed)
  - prev turn (same-speaker only) — 앞서 같은 화자가 말한 가장 가까운 turn
  - next turn (same-speaker only) — 이후 같은 화자가 말한 가장 가까운 turn
  - "Sad 가 emotion-consistent context 에 있는가" 정량화
    (인접 turn 이 비중립 affective 인 비율)

실행:
  cd <repo-root>
  python eda_kmind_sad_context.py
출력: ./eda_outputs/kmind_sad_context.csv + 콘솔
"""
import os
import pandas as pd
from collections import Counter

ROOT = os.environ.get("ALIGNEMO_DATA_ROOT", ".")
OUT = os.environ.get("ALIGNEMO_OUT_DIR", "eda/outputs"); os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(ROOT, "dataset/KMIND/splits/test_ekman_0210.csv")

# 한국어 emotion → 영어 매핑
KO2EN = {
    "중립": "neutral", "기쁨": "joy", "슬픔": "sad", "화남": "anger",
    "놀람": "surprise", "무서움": "fear", "혐오": "disgust",
}

def main():
    df = pd.read_csv(CSV)
    print(f"전체 K-MIND test utterance: {len(df)}")
    print(f"컬럼: {list(df.columns)}")

    df["emo"] = df["Affective_State"].map(lambda e: KO2EN.get(str(e).strip(), str(e).strip()))
    print(f"\n전체 emotion 분포:")
    for emo, cnt in df["emo"].value_counts().items():
        print(f"  {emo:<10s} {cnt:>6d} ({100*cnt/len(df):>5.1f}%)")

    # dialogue 단위로 정렬
    df = df.sort_values(["Dialogue_ID", "Start_Time"]).reset_index(drop=True)

    # 각 utterance 에 대화 내 turn index 부여
    df["turn_idx"] = df.groupby("Dialogue_ID").cumcount()

    sad_mask = df["emo"] == "sad"
    n_sad = sad_mask.sum()
    print(f"\nSad utterance 총 개수: {n_sad}")

    prev_emos_mixed = []          # 직전 turn (화자 무관)
    next_emos_mixed = []          # 직후 turn (화자 무관)
    prev_emos_same_spk = []       # 같은 화자의 가장 가까운 이전 turn
    next_emos_same_spk = []       # 같은 화자의 가장 가까운 이후 turn
    prev_emos_other_spk = []      # 다른 화자의 가장 가까운 이전 turn (보통 직전과 동일)
    next_emos_other_spk = []      # 다른 화자의 가장 가까운 이후 turn

    rows_detail = []

    for idx in df[sad_mask].index:
        dlg_id = df.at[idx, "Dialogue_ID"]
        speaker = df.at[idx, "Speaker"]
        t = df.at[idx, "turn_idx"]
        utt = df.at[idx, "Utterance"]

        # 같은 dialogue 의 모든 turn
        same_dlg = df[df["Dialogue_ID"] == dlg_id].sort_values("turn_idx")

        # (1) 직전 turn (mixed)
        prev_rows = same_dlg[same_dlg["turn_idx"] < t]
        prev_emo = prev_rows.iloc[-1]["emo"] if len(prev_rows) > 0 else None
        prev_spk = prev_rows.iloc[-1]["Speaker"] if len(prev_rows) > 0 else None
        if prev_emo: prev_emos_mixed.append(prev_emo)

        # (2) 직후 turn (mixed)
        next_rows = same_dlg[same_dlg["turn_idx"] > t]
        next_emo = next_rows.iloc[0]["emo"] if len(next_rows) > 0 else None
        next_spk = next_rows.iloc[0]["Speaker"] if len(next_rows) > 0 else None
        if next_emo: next_emos_mixed.append(next_emo)

        # (3) 같은 화자 이전 turn
        prev_same = prev_rows[prev_rows["Speaker"] == speaker]
        prev_same_emo = prev_same.iloc[-1]["emo"] if len(prev_same) > 0 else None
        if prev_same_emo: prev_emos_same_spk.append(prev_same_emo)

        # (4) 같은 화자 이후 turn
        next_same = next_rows[next_rows["Speaker"] == speaker]
        next_same_emo = next_same.iloc[0]["emo"] if len(next_same) > 0 else None
        if next_same_emo: next_emos_same_spk.append(next_same_emo)

        # (5) 다른 화자 이전 turn
        prev_other = prev_rows[prev_rows["Speaker"] != speaker]
        prev_other_emo = prev_other.iloc[-1]["emo"] if len(prev_other) > 0 else None
        if prev_other_emo: prev_emos_other_spk.append(prev_other_emo)

        # (6) 다른 화자 이후 turn
        next_other = next_rows[next_rows["Speaker"] != speaker]
        next_other_emo = next_other.iloc[0]["emo"] if len(next_other) > 0 else None
        if next_other_emo: next_emos_other_spk.append(next_other_emo)

        rows_detail.append(dict(
            dialogue_id=dlg_id, turn_idx=t, speaker=speaker,
            utterance=utt[:60],
            prev_mixed=prev_emo, next_mixed=next_emo,
            prev_same_spk=prev_same_emo, next_same_spk=next_same_emo,
            prev_other_spk=prev_other_emo, next_other_spk=next_other_emo,
        ))

    # ============================
    # 분포 집계 + print
    # ============================
    def show_dist(label, lst):
        c = Counter(lst)
        n = sum(c.values())
        print(f"\n=== {label} (n={n}) ===")
        for emo, cnt in c.most_common():
            print(f"  {emo:<10s} {cnt:>4d}  ({100*cnt/n:>5.1f}%)")
        return c

    print("\n" + "="*60)
    print("Sad utterance 의 인접 turn emotion 분포")
    print("="*60)
    c1 = show_dist("PREV turn (mixed, 화자 무관)", prev_emos_mixed)
    c2 = show_dist("NEXT turn (mixed, 화자 무관)", next_emos_mixed)
    c3 = show_dist("PREV turn (same-speaker only)", prev_emos_same_spk)
    c4 = show_dist("NEXT turn (same-speaker only)", next_emos_same_spk)
    c5 = show_dist("PREV turn (other-speaker only)", prev_emos_other_spk)
    c6 = show_dist("NEXT turn (other-speaker only)", next_emos_other_spk)

    # ============================
    # "Emotion-consistent context" 정량화
    # ============================
    affective = {"joy", "sad", "anger", "surprise", "fear", "disgust"}

    def affective_ratio(lst):
        n = len(lst)
        if n == 0: return 0
        return sum(1 for e in lst if e in affective) / n * 100

    def sad_ratio(lst):
        n = len(lst)
        if n == 0: return 0
        return sum(1 for e in lst if e == "sad") / n * 100

    def neutral_ratio(lst):
        n = len(lst)
        if n == 0: return 0
        return sum(1 for e in lst if e == "neutral") / n * 100

    print("\n" + "="*60)
    print("Sad 가 emotion-consistent context 에 있는지 (%)")
    print("="*60)
    print(f"{'context':<35s} {'Neutral%':>10s} {'Sad%':>8s} {'Affective%':>12s}")
    print("-"*65)
    for label, lst in [
        ("PREV (mixed)", prev_emos_mixed),
        ("NEXT (mixed)", next_emos_mixed),
        ("PREV (same-spk)", prev_emos_same_spk),
        ("NEXT (same-spk)", next_emos_same_spk),
        ("PREV (other-spk)", prev_emos_other_spk),
        ("NEXT (other-spk)", next_emos_other_spk),
    ]:
        print(f"  {label:<33s} {neutral_ratio(lst):>9.1f} {sad_ratio(lst):>7.1f} {affective_ratio(lst):>11.1f}")

    # CSV 저장
    pd.DataFrame(rows_detail).to_csv(f"{OUT}/kmind_sad_context_detail.csv", index=False)
    print(f"\n[OK] detail saved: {OUT}/kmind_sad_context_detail.csv")

    # 요약 CSV 저장
    summary = []
    for label, lst in [
        ("prev_mixed", prev_emos_mixed),
        ("next_mixed", next_emos_mixed),
        ("prev_same_spk", prev_emos_same_spk),
        ("next_same_spk", next_emos_same_spk),
        ("prev_other_spk", prev_emos_other_spk),
        ("next_other_spk", next_emos_other_spk),
    ]:
        c = Counter(lst)
        n = sum(c.values())
        for emo, cnt in c.items():
            summary.append(dict(context=label, emotion=emo, count=cnt, pct=round(100*cnt/n, 2)))
    pd.DataFrame(summary).to_csv(f"{OUT}/kmind_sad_context_summary.csv", index=False)
    print(f"[OK] summary saved: {OUT}/kmind_sad_context_summary.csv")


if __name__ == "__main__":
    main()
