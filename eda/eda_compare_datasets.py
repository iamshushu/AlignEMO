"""
세 ERC 데이터셋(IEMOCAP / MELD / K-MIND-Ekman) 종합 데이터 분석(EDA).

GPU 불필요. 순수 pandas/json/matplotlib. 로그인 노드에서 바로:
    cd <repo-root>
    python eda_compare_datasets.py

분석 항목 (데이터셋 x split):
  1) 감정 클래스 체계 + 교차 데이터셋 매핑
  2) 클래스 분포 (개수/비율) + 불균형비(IR) + 다수클래스 비중 + 엔트로피
  3) 대화 구조 (#대화, #발화, 대화당 턴 수, 화자 수, dyadic/multi-party)
  4) 발화 길이 (문자수/공백토큰수) 전체 + 감정별
  5) 감정별 예시 발화 (각 5개) -> 직접 눈으로 확인
  6) 모달리티 가용성 (text/audio/video)

출력:
  ./eda_outputs/
     - dist_<dataset>_<split>.csv         (클래스 분포)
     - summary_comparison.md              (교차 비교 표)
     - sample_utterances.txt              (감정별 예시)
     - fig_class_dist_norm.png            (정규화 분포 비교)
     - fig_class_counts_<dataset>.png     (개수 막대)
     - fig_uttlen_box_<dataset>.png       (감정별 발화길이)
  + 콘솔에 전체 리포트 출력
"""
import os
import re
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.environ.get("ALIGNEMO_DATA_ROOT", ".")
OUT = os.environ.get("ALIGNEMO_OUT_DIR", "eda/outputs")
os.makedirs(OUT, exist_ok=True)
random.seed(42)

# ------------------------------------------------------------------
# 경로
# ------------------------------------------------------------------
MELD_CSV = {s: f"{ROOT}/EACL/data/MELD/{s}_data.csv" for s in ["train", "dev", "test"]}
IEMOCAP_JSON = {s: f"{ROOT}/EACL/data/IEMOCAP/{s}_data.json" for s in ["train", "dev", "test"]}
KMIND_JSON = {s: f"{ROOT}/EACL/data/KMIND_ekman/{s}_data.json" for s in ["train", "dev", "test"]}
KMIND_CSV = f"{ROOT}/EACL/data/KMIND_ekman/KMIND_with_split_ekman.csv"

# 한국어 -> 영어 (그림 라벨용; 텍스트/CSV는 원문 유지)
KMIND_EN = {"기쁨": "joy", "슬픔": "sad", "화남": "anger", "놀람": "surprise",
            "무서움": "fear", "혐오": "disgust", "중립": "neutral"}
IEMO_FULL = {"ang": "anger", "exc": "excited", "fru": "frustrated",
             "hap": "happy", "neu": "neutral", "sad": "sad"}

# 교차 데이터셋 공통 감정 매핑 (대략적 정렬)
COMMON = ["joy/happy", "sadness", "anger", "surprise", "fear", "disgust",
          "neutral", "excited", "frustrated"]


# ------------------------------------------------------------------
# 로더: 공통 스키마 -> rows(dict): dataset, split, dialogue_id, turn, text, speaker, emotion
# ------------------------------------------------------------------
def load_meld():
    rows = []
    for split, path in MELD_CSV.items():
        if not os.path.exists(path):
            print(f"[WARN] MELD missing: {path}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(dict(
                dataset="MELD", split=split,
                dialogue_id=f"{split}_{r['Dialogue_ID']}",
                turn=int(r["Utterance_ID"]),
                text=str(r["Utterance"]),
                speaker=str(r["Speaker"]),
                emotion=str(r["Emotion"]).strip().lower(),
            ))
    return rows


def _load_erc_json(path, dataset, split):
    rows = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for di, dialogue in enumerate(data):
        for ti, turn in enumerate(dialogue):
            rows.append(dict(
                dataset=dataset, split=split,
                dialogue_id=f"{split}_{di}",
                turn=ti,
                text=str(turn.get("text", "")),
                speaker=str(turn.get("speaker", "")),
                emotion=str(turn.get("label", "")).strip(),
            ))
    return rows


def load_iemocap():
    rows = []
    for split, path in IEMOCAP_JSON.items():
        if not os.path.exists(path):
            print(f"[WARN] IEMOCAP missing: {path}")
            continue
        rows += _load_erc_json(path, "IEMOCAP", split)
    return rows


def load_kmind():
    rows = []
    for split, path in KMIND_JSON.items():
        if not os.path.exists(path):
            print(f"[WARN] K-MIND missing: {path}")
            continue
        rows += _load_erc_json(path, "K-MIND", split)
    return rows


# ------------------------------------------------------------------
# 분석 유틸
# ------------------------------------------------------------------
def tok_len(s):
    return len(s.split())


def char_len(s):
    return len(s)


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in ps)


def class_distribution(rows, dataset):
    """split별 + 전체 클래스 분포 표 반환 + 콘솔 출력."""
    df = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
    if df.empty:
        return None
    print(f"\n{'='*70}\n[{dataset}] 클래스 분포\n{'='*70}")
    out_tables = {}
    for split in ["train", "dev", "test", "ALL"]:
        sub = df if split == "ALL" else df[df["split"] == split]
        if sub.empty:
            continue
        cnt = sub["emotion"].value_counts()
        total = cnt.sum()
        tab = pd.DataFrame({
            "emotion": cnt.index,
            "count": cnt.values,
            "pct": (cnt.values / total * 100).round(2),
        })
        ir = cnt.max() / cnt.min() if cnt.min() > 0 else float("inf")
        H = entropy(list(cnt.values))
        Hmax = math.log2(len(cnt)) if len(cnt) > 1 else 1
        print(f"\n  --- {split} (N={total}, #class={len(cnt)}, "
              f"IR={ir:.1f}x, majority={cnt.max()/total*100:.1f}%, "
              f"entropy={H:.2f}/{Hmax:.2f}) ---")
        for _, rr in tab.iterrows():
            bar = "█" * int(rr["pct"] / 2)
            print(f"    {str(rr['emotion'])[:10]:<12} {int(rr['count']):>7d}  "
                  f"{rr['pct']:>5.1f}%  {bar}")
        if split == "ALL":
            tab.to_csv(f"{OUT}/dist_{dataset}_ALL.csv", index=False)
        out_tables[split] = tab
    return out_tables


def structure_stats(rows, dataset):
    df = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
    if df.empty:
        return None
    n_utt = len(df)
    n_dlg = df["dialogue_id"].nunique()
    turns = df.groupby("dialogue_id").size()
    # 화자: MELD=인물명, IEMOCAP/K-MIND=화자ID(역할 포함)
    spk_per_dlg = df.groupby("dialogue_id")["speaker"].nunique()
    n_spk_global = df["speaker"].nunique()
    party = "dyadic(2인)" if spk_per_dlg.mean() <= 2.3 else "multi-party"
    print(f"\n{'='*70}\n[{dataset}] 대화 구조\n{'='*70}")
    print(f"  #대화(dialogue)   : {n_dlg}")
    print(f"  #발화(utterance)  : {n_utt}")
    print(f"  대화당 턴 수      : 평균 {turns.mean():.1f} / 중앙 {int(turns.median())} / 최대 {turns.max()}")
    print(f"  대화당 화자 수    : 평균 {spk_per_dlg.mean():.2f}  -> {party}")
    print(f"  전체 고유 화자 수 : {n_spk_global}")
    return dict(dataset=dataset, n_dialogue=n_dlg, n_utt=n_utt,
                turns_mean=round(turns.mean(), 1), turns_max=int(turns.max()),
                spk_per_dlg=round(spk_per_dlg.mean(), 2), party=party,
                n_speaker=n_spk_global)


def length_stats(rows, dataset):
    df = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
    if df.empty:
        return None
    df["toks"] = df["text"].map(tok_len)
    df["chars"] = df["text"].map(char_len)
    print(f"\n{'='*70}\n[{dataset}] 발화 길이\n{'='*70}")
    print(f"  토큰(공백) : 평균 {df['toks'].mean():.1f} / 중앙 {int(df['toks'].median())} / 최대 {df['toks'].max()}")
    print(f"  문자       : 평균 {df['chars'].mean():.1f} / 중앙 {int(df['chars'].median())}")
    print(f"  감정별 평균 토큰 수:")
    g = df.groupby("emotion")["toks"].mean().sort_values(ascending=False)
    for emo, v in g.items():
        print(f"    {str(emo)[:10]:<12} {v:>5.1f}")
    return df


def sample_utterances(rows, dataset, n=5, fh=None):
    df = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
    if df.empty:
        return
    header = f"\n{'#'*70}\n# [{dataset}] 감정별 예시 발화 (각 {n}개)\n{'#'*70}"
    print(header)
    if fh:
        fh.write(header + "\n")
    for emo in sorted(df["emotion"].unique()):
        sub = df[df["emotion"] == emo]
        picks = sub.sample(min(n, len(sub)), random_state=42)
        line = f"\n  ── {emo}  (n={len(sub)}) ──"
        print(line)
        if fh:
            fh.write(line + "\n")
        for _, r in picks.iterrows():
            txt = r["text"].replace("\n", " ")[:120]
            out = f"     [{r['speaker'][:18]:<18}] {txt}"
            print(out)
            if fh:
                fh.write(out + "\n")


def modality_note(dataset):
    notes = {
        "MELD": "텍스트(O) / 오디오·비디오: 원본 MELD.Raw 에 존재(현재 EACL csv엔 경로 없음). 다자 대화(Friends 시트콤), StartTime/EndTime 보유.",
        "IEMOCAP": "텍스트(O) / 오디오·비디오: 원본에 존재(현재 EACL json엔 경로 없음). 2인 연기(acted) 대화, 즉흥(impro)/대본(script) 혼합.",
        "K-MIND": "텍스트(O) / 오디오(Wav_Path) + 영상프레임(Video_Path) 경로 CSV에 존재. 한국어 2인 대화.",
    }
    print(f"\n[{dataset}] 모달리티: {notes.get(dataset, '?')}")
    return notes.get(dataset, "")


# ------------------------------------------------------------------
# 그림
# ------------------------------------------------------------------
def make_plots(all_rows, dist_by_ds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — 그림 생략")
        return

    def en_label(ds, emo):
        if ds == "K-MIND":
            return KMIND_EN.get(emo, emo)
        if ds == "IEMOCAP":
            return IEMO_FULL.get(emo, emo)
        return emo

    # 1) 정규화 분포 비교 (datasets 옆으로)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, ds in zip(axes, ["IEMOCAP", "MELD", "K-MIND"]):
        tab = dist_by_ds.get(ds, {}).get("ALL")
        if tab is None:
            continue
        labels = [en_label(ds, e) for e in tab["emotion"]]
        ax.bar(labels, tab["pct"], color="steelblue")
        ax.set_title(f"{ds}  (N={int(tab['count'].sum())})")
        ax.set_ylabel("% of utterances")
        ax.tick_params(axis="x", rotation=45)
        for i, p in enumerate(tab["pct"]):
            ax.text(i, p + 0.5, f"{p:.0f}", ha="center", fontsize=8)
    plt.suptitle("Class distribution comparison (normalized)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_class_dist_norm.png", dpi=150)
    plt.close()
    print(f"[OK] {OUT}/fig_class_dist_norm.png")

    # 2) 감정별 발화길이 박스플롯
    for ds in ["IEMOCAP", "MELD", "K-MIND"]:
        df = pd.DataFrame([r for r in all_rows if r["dataset"] == ds])
        if df.empty:
            continue
        df["toks"] = df["text"].map(tok_len)
        emos = sorted(df["emotion"].unique())
        data = [df[df["emotion"] == e]["toks"].values for e in emos]
        labels = [en_label(ds, e) for e in emos]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{ds}: utterance length (whitespace tokens) by emotion")
        ax.set_ylabel("tokens")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUT}/fig_uttlen_box_{ds}.png", dpi=150)
        plt.close()
        print(f"[OK] {OUT}/fig_uttlen_box_{ds}.png")


# ------------------------------------------------------------------
# 교차 비교 요약 (markdown)
# ------------------------------------------------------------------
def write_summary_md(struct_rows, dist_by_ds):
    path = f"{OUT}/summary_comparison.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# IEMOCAP vs MELD vs K-MIND : 데이터 비교\n\n")

        f.write("## 1. 감정 클래스 체계\n\n")
        f.write("| Dataset | #class | classes |\n|---|---|---|\n")
        f.write("| IEMOCAP | 6 | ang, exc(excited), fru(frustrated), hap, neu, sad |\n")
        f.write("| MELD | 7 | anger, disgust, fear, joy, neutral, sadness, surprise |\n")
        f.write("| K-MIND(Ekman) | 7 | 기쁨, 슬픔, 화남, 놀람, 무서움, 혐오, 중립 |\n\n")
        f.write("- MELD·K-MIND 는 **Ekman 6 + neutral** 체계로 동일 정렬 가능.\n")
        f.write("- IEMOCAP 은 surprise/disgust/fear 없음, 대신 **excited·frustrated** 포함 (감정 축이 다름).\n\n")

        f.write("## 2. 구조 / 규모\n\n")
        f.write("| Dataset | #대화 | #발화 | 턴/대화(avg) | 화자/대화 | 형태 | 고유화자 |\n|---|---|---|---|---|---|---|\n")
        for s in struct_rows:
            if s:
                f.write(f"| {s['dataset']} | {s['n_dialogue']} | {s['n_utt']} | "
                        f"{s['turns_mean']} | {s['spk_per_dlg']} | {s['party']} | {s['n_speaker']} |\n")
        f.write("\n")

        f.write("## 3. 클래스 불균형 (전체 기준)\n\n")
        f.write("| Dataset | majority class % | IR(max/min) | entropy/max |\n|---|---|---|---|\n")
        for ds in ["IEMOCAP", "MELD", "K-MIND"]:
            tab = dist_by_ds.get(ds, {}).get("ALL")
            if tab is None:
                continue
            cnts = tab["count"].values
            total = cnts.sum()
            ir = cnts.max() / cnts.min()
            H = entropy(list(cnts))
            Hmax = math.log2(len(cnts))
            f.write(f"| {ds} | {cnts.max()/total*100:.1f}% | {ir:.1f}x | {H:.2f}/{Hmax:.2f} |\n")
        f.write("\n불균형이 심할수록 IR↑, entropy↓.\n")
    print(f"[OK] {path}")


# ------------------------------------------------------------------
def main():
    print("데이터 로딩 중...")
    meld = load_meld()
    iemo = load_iemocap()
    kmind = load_kmind()
    all_rows = meld + iemo + kmind
    print(f"로드 완료: MELD={len(meld)}, IEMOCAP={len(iemo)}, K-MIND={len(kmind)} 발화")

    dist_by_ds, struct_rows = {}, []
    fh = open(f"{OUT}/sample_utterances.txt", "w", encoding="utf-8")
    for ds in ["IEMOCAP", "MELD", "K-MIND"]:
        dist_by_ds[ds] = class_distribution(all_rows, ds) or {}
        struct_rows.append(structure_stats(all_rows, ds))
        length_stats(all_rows, ds)
        modality_note(ds)
        sample_utterances(all_rows, ds, n=5, fh=fh)
    fh.close()
    print(f"\n[OK] {OUT}/sample_utterances.txt")

    write_summary_md(struct_rows, dist_by_ds)
    make_plots(all_rows, dist_by_ds)
    print("\n=== EDA 완료. ./eda_outputs/ 확인 ===")


if __name__ == "__main__":
    main()
