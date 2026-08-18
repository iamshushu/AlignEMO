"""
Sentiment-level evaluation utility.

기존 emotion-level inference 결과 (pred_emotion, true_emotion) 를 받아서
sentiment-level (negative/positive/neutral) 로 collapse 후 w-F1, macro-F1 계산.

사용법:
  1. 각 데이터셋의 Stage 1B / Stage 2 ELACL inference 스크립트에서
     pred_labels, true_labels 를 numpy array 또는 list 로 저장
  2. 이 스크립트의 evaluate_both_levels(pred, true, dataset) 호출
  3. 또는 CSV 로 저장된 (pred, true) 를 load 해서 분석

출력:
  - Emotion-level w-F1 / macro-F1
  - Sentiment-level w-F1 / macro-F1
  - Per-sentiment F1 breakdown
  - Gap analysis (emotion vs sentiment)
"""
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import argparse, sys


# =========================================================
# Emotion → Sentiment mapping (per dataset)
# =========================================================
# IEMOCAP 6-class: happy, sad, neutral, anger, excited, frustrated
IEMOCAP_E2S = {
    "happy": "positive", "excited": "positive",
    "sad": "negative", "anger": "negative", "angry": "negative",
    "frustrated": "negative", "frustration": "negative",
    "neutral": "neutral",
}

# MELD 7-class: anger, disgust, fear, joy, neutral, sadness, surprise
MELD_E2S = {
    "joy": "positive",
    "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
    "neutral": "neutral",
    "surprise": "neutral",  # MELD 논문: surprise 는 binary 가능. 여기선 단순화로 neutral
                            # 또는 dataset 의 sentiment label 따로 사용 가능
}

# K-MIND 7-class (Ekman): joy, sad, anger, surprise, fear, disgust, neutral
KMIND_E2S = {
    "joy": "positive", "기쁨": "positive",
    "sad": "negative", "슬픔": "negative",
    "anger": "negative", "화남": "negative",
    "disgust": "negative", "혐오": "negative",
    "fear": "negative", "무서움": "negative",
    "neutral": "neutral", "중립": "neutral",
    "surprise": "neutral", "놀람": "neutral",
}

DATASET_MAPS = {
    "iemocap": IEMOCAP_E2S,
    "meld": MELD_E2S,
    "kmind": KMIND_E2S,
    "k-mind": KMIND_E2S,
}


# =========================================================
# 평가 함수
# =========================================================
def evaluate_both_levels(pred_emotions, true_emotions, dataset="meld", verbose=True):
    """
    Args:
      pred_emotions: list[str] of predicted emotion labels
      true_emotions: list[str] of ground truth emotion labels
      dataset: "iemocap" / "meld" / "kmind"
    Returns:
      dict with emotion_wf1, emotion_macro, sentiment_wf1, sentiment_macro
    """
    e2s = DATASET_MAPS[dataset.lower()]

    pred_emo = [str(e).strip().lower() for e in pred_emotions]
    true_emo = [str(e).strip().lower() for e in true_emotions]

    # Emotion-level
    emo_wf1 = f1_score(true_emo, pred_emo, average="weighted", zero_division=0) * 100
    emo_macro = f1_score(true_emo, pred_emo, average="macro", zero_division=0) * 100

    # Sentiment-level (collapse)
    def map_to_sent(e):
        return e2s.get(e, e2s.get(e.lower(), "unknown"))

    pred_sent = [map_to_sent(e) for e in pred_emo]
    true_sent = [map_to_sent(e) for e in true_emo]

    # filter unknown
    valid = [(p, t) for p, t in zip(pred_sent, true_sent) if p != "unknown" and t != "unknown"]
    if len(valid) < len(pred_sent):
        n_unk = len(pred_sent) - len(valid)
        if verbose:
            print(f"  [WARN] {n_unk} 발화의 emotion 이 mapping 에 없음 (제외됨)")
    pred_sent = [p for p, t in valid]
    true_sent = [t for p, t in valid]

    sent_wf1 = f1_score(true_sent, pred_sent, average="weighted", zero_division=0) * 100
    sent_macro = f1_score(true_sent, pred_sent, average="macro", zero_division=0) * 100

    if verbose:
        print(f"\n=== {dataset.upper()} ===")
        print(f"N = {len(pred_emo)} (emotion-level) / {len(pred_sent)} (sentiment-level)")
        print(f"\nEmotion-level:")
        print(f"  weighted-F1 = {emo_wf1:.2f}")
        print(f"  macro-F1    = {emo_macro:.2f}")
        print(f"\nSentiment-level:")
        print(f"  weighted-F1 = {sent_wf1:.2f}")
        print(f"  macro-F1    = {sent_macro:.2f}")
        print(f"\nGap (sentiment - emotion):")
        print(f"  weighted-F1 gap = {sent_wf1 - emo_wf1:+.2f}")
        print(f"  macro-F1 gap    = {sent_macro - emo_macro:+.2f}")
        print(f"\nPer-sentiment classification report:")
        print(classification_report(true_sent, pred_sent, digits=3, zero_division=0))

    return dict(
        dataset=dataset,
        n=len(pred_emo),
        emotion_wf1=round(emo_wf1, 2),
        emotion_macro=round(emo_macro, 2),
        sentiment_wf1=round(sent_wf1, 2),
        sentiment_macro=round(sent_macro, 2),
        wf1_gap=round(sent_wf1 - emo_wf1, 2),
        macro_gap=round(sent_macro - emo_macro, 2),
    )


# =========================================================
# CSV 로부터 load 해서 평가 (기존 inference 결과가 CSV 로 있을 경우)
# =========================================================
def evaluate_from_csv(csv_path, dataset, pred_col="pred", true_col="true"):
    """CSV 에서 (pred, true) 컬럼을 load 해서 평가."""
    df = pd.read_csv(csv_path)
    return evaluate_both_levels(
        pred_emotions=df[pred_col].tolist(),
        true_emotions=df[true_col].tolist(),
        dataset=dataset,
    )


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="CSV path with pred, true columns")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "meld", "kmind", "k-mind"])
    parser.add_argument("--pred_col", type=str, default="pred")
    parser.add_argument("--true_col", type=str, default="true")
    args = parser.parse_args()

    if args.csv:
        evaluate_from_csv(args.csv, args.dataset, args.pred_col, args.true_col)
    else:
        print("Usage: python eda_sentiment_level_eval.py --csv <path> --dataset <meld|iemocap|kmind>")
        print("Or import evaluate_both_levels() and pass pred/true lists directly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
