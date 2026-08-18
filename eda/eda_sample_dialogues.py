"""
각 데이터셋에서 sad density 가 representative 한 dialogue 샘플 추출.

각 데이터셋 × 3 case:
  (A) HIGH density: sad ratio 가 큰 dialogue 1개 (dramatic case)
  (B) MEDIAN density: sad ratio 가 median 근처인 dialogue 1개 (typical case)
  (C) LOW density: sad ratio 가 작은 (but > 0) dialogue 1개 (sparse case)

각 dialogue 의 emotion sequence + sad utterance 의 앞뒤 context 출력.

PPT 에 그대로 paste 가능한 format.

실행:
  python eda_sample_dialogues.py > eda_outputs/sample_sad_dialogues.txt
"""
import pandas as pd
from eda_compare_datasets import load_meld, load_iemocap, load_kmind, KMIND_EN, IEMO_FULL, OUT

DATASETS = ["IEMOCAP", "MELD", "K-MIND"]
FOCAL_PER_DS = {"IEMOCAP": "sad", "MELD": "sadness", "K-MIND": "sad"}

# Emotion → short code for sequence visualization
SHORT = {
    "neutral": "N", "sad": "S", "sadness": "S", "anger": "A", "angry": "A",
    "joy": "J", "happy": "J", "excited": "E", "frustrated": "F",
    "surprise": "U", "fear": "X", "disgust": "D",
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


def find_representative_dialogues(df, focal):
    """sad density 가 high/median/low 인 dialogue 1개씩 선택."""
    rows = []
    for dlg_id, g in df.groupby("dialogue_id"):
        N = len(g)
        if N < 4:
            continue
        n_sad = (g["emo_en"] == focal).sum()
        if n_sad == 0:
            continue
        rows.append((dlg_id, N, n_sad, n_sad / N))

    if not rows:
        return {}
    r_df = pd.DataFrame(rows, columns=["dlg_id", "N", "n_sad", "ratio"])
    r_df = r_df.sort_values("ratio").reset_index(drop=True)

    # 3 case 추출
    low = r_df.iloc[len(r_df) // 8]                # 하위 12% 근처
    median = r_df.iloc[len(r_df) // 2]              # 정중앙
    # high: median 의 2배 이상이거나 가장 큰 것
    high = r_df.iloc[-1]
    # high 가 너무 극단 (100%) 이면 75퍼센타일 사용
    if high['ratio'] > 0.8 and len(r_df) > 10:
        high = r_df.iloc[int(len(r_df) * 0.75)]
    return {"LOW": low, "MEDIAN": median, "HIGH": high}


def print_dialogue(df, dlg_id, focal, label):
    g = df[df["dialogue_id"] == dlg_id].copy()
    g = g.sort_values("turn" if "turn" in g.columns else g.index.name or "Start_Time") if "turn" in g.columns else g
    if "turn" in g.columns:
        g = g.sort_values("turn")

    N = len(g)
    n_sad = (g["emo_en"] == focal).sum()
    ratio = n_sad / N * 100
    print(f"\n  [{label}]  dlg_id={dlg_id}  N={N}  sad={n_sad}  ratio={ratio:.1f}%")

    # Emotion sequence (단일 line)
    seq = "".join(SHORT.get(e, "?") for e in g["emo_en"].tolist())
    print(f"    sequence:  {seq}")
    print(f"    legend:    S=sad/sadness  N=neutral  J=joy/happy  A=anger  "
          f"E=excited  F=frustrated  U=surprise  X=fear  D=disgust")

    # Sad utterance 의 앞뒤 1턴씩 context 출력
    sad_indices = [i for i, e in enumerate(g["emo_en"].tolist()) if e == focal]
    if not sad_indices:
        return
    print(f"    Sad utterances + context (앞뒤 1턴):")

    utts = g["text"].tolist() if "text" in g.columns else g["utterance"].tolist() if "utterance" in g.columns else None
    if utts is None:
        # 다른 column 시도
        for col in ["Utterance", "utterance", "text"]:
            if col in g.columns:
                utts = g[col].tolist()
                break
    speakers = g["speaker"].tolist() if "speaker" in g.columns else [""] * N
    emos_seq = g["emo_en"].tolist()

    # 한 dialogue 에 sad 가 많으면 최대 5개 까지만 출력
    for sad_i in sad_indices[:5]:
        start = max(0, sad_i - 1)
        end = min(N, sad_i + 2)
        for k in range(start, end):
            marker = "▶" if k == sad_i else " "
            txt = str(utts[k])[:80] if utts else ""
            print(f"     {marker} [{emos_seq[k]:>10s}] (spk={speakers[k]}) {txt}")
        if sad_i != sad_indices[:5][-1]:
            print(f"      ...")


def main():
    print("로딩...")
    all_rows = load_meld() + load_iemocap() + load_kmind()
    dfs = {ds: get_df(all_rows, ds) for ds in DATASETS}

    for ds in DATASETS:
        focal = FOCAL_PER_DS[ds]
        print(f"\n{'='*70}")
        print(f"[{ds}]  focal = '{focal}'")
        print(f"{'='*70}")

        df = dfs[ds]
        cases = find_representative_dialogues(df, focal)
        if not cases:
            print("  (no sad-containing dialogue)")
            continue

        for label, row in cases.items():
            print_dialogue(df, row["dlg_id"], focal, label)


if __name__ == "__main__":
    main()
