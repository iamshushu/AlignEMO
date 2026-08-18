"""
Confusion matrix 기반 sentiment-level evaluation.

기존 emotion-level confusion matrix 를 sentiment 그룹으로 collapse 해서:
  - Sentiment-level weighted F1, macro F1
  - Sentiment-level per-class F1
  - Within-sentiment vs cross-sentiment confusion 비율
계산.

사용법:
  1. 각 데이터셋의 Stage 2 ELACL inference 끝부분에 confusion matrix 한 번만 저장
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(true_labels, pred_labels, labels=emoList)
     np.save("cm_iemocap_stage2.npy", cm)

  2. 이 스크립트로 collapse:
     python eda_cm_sentiment_collapse.py --cm cm_iemocap_stage2.npy --dataset iemocap

  또는 import 해서 직접 호출:
     from eda_cm_sentiment_collapse import analyze_cm
     analyze_cm(cm, emo_labels=['ang','exc','fru','hap','neu','sad'], dataset='iemocap')
"""
import numpy as np
import argparse


# =========================================================
# Emotion → Sentiment mapping
# =========================================================
SENT_MAPS = {
    "iemocap": {
        "hap": "positive", "happy": "positive",
        "exc": "positive", "excited": "positive",
        "sad": "negative", "sadness": "negative",
        "ang": "negative", "anger": "negative", "angry": "negative",
        "fru": "negative", "frustrated": "negative", "frustration": "negative",
        "neu": "neutral", "neutral": "neutral",
    },
    "meld": {
        "joy": "positive",
        "anger": "negative", "disgust": "negative",
        "fear": "negative", "sadness": "negative",
        "neutral": "neutral",
        "surprise": "neutral",  # 단순화 (MELD 원본 sentiment label 따로 있음)
    },
    "kmind": {
        "joy": "positive", "기쁨": "positive",
        "sad": "negative", "슬픔": "negative",
        "anger": "negative", "화남": "negative",
        "disgust": "negative", "혐오": "negative",
        "fear": "negative", "무서움": "negative",
        "neutral": "neutral", "중립": "neutral",
        "surprise": "neutral", "놀람": "neutral",
    },
}


# =========================================================
# Core function
# =========================================================
def collapse_cm(cm, emo_labels, e2s_map):
    """
    Emotion-level CM (N×N) → Sentiment-level CM (M×M).

    cm[i,j] = #(true=emo_labels[i], pred=emo_labels[j])
    return: cm_sent[a,b] = #(true=sent_a, pred=sent_b)
    """
    sent_labels = sorted(set(e2s_map.values()))
    sent_idx = {s: i for i, s in enumerate(sent_labels)}

    n_sent = len(sent_labels)
    cm_sent = np.zeros((n_sent, n_sent), dtype=int)

    for i, ei in enumerate(emo_labels):
        for j, ej in enumerate(emo_labels):
            si = sent_idx[e2s_map[ei.lower()]]
            sj = sent_idx[e2s_map[ej.lower()]]
            cm_sent[si, sj] += cm[i, j]

    return cm_sent, sent_labels


def metrics_from_cm(cm, labels):
    """CM 에서 per-class P/R/F1 + weighted-F1 + macro-F1 계산."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    support = cm.sum(axis=1).astype(float)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)

    total = support.sum()
    weighted_f1 = (f1 * support).sum() / total * 100 if total > 0 else 0
    macro_f1 = f1.mean() * 100

    return dict(
        per_class_f1={lab: round(f * 100, 2) for lab, f in zip(labels, f1)},
        per_class_precision={lab: round(p * 100, 2) for lab, p in zip(labels, precision)},
        per_class_recall={lab: round(r * 100, 2) for lab, r in zip(labels, recall)},
        per_class_support={lab: int(s) for lab, s in zip(labels, support)},
        weighted_f1=round(weighted_f1, 2),
        macro_f1=round(macro_f1, 2),
    )


def analyze_within_sentiment_confusion(cm, emo_labels, e2s_map):
    """
    Within-sentiment vs cross-sentiment 오류 비율.
    예: negative 끼리 헷갈리는 비율 vs neutral/positive 로 leak 하는 비율.
    """
    sent_groups = {}
    for emo in emo_labels:
        s = e2s_map[emo.lower()]
        sent_groups.setdefault(s, []).append(emo_labels.index(emo))

    result = {}
    for sent, idxs in sent_groups.items():
        # true label 이 이 sentiment 그룹인 utterance 의 분류 결과
        block = cm[np.ix_(idxs, range(cm.shape[0]))]
        total = block.sum()
        if total == 0:
            continue

        # 같은 sentiment 그룹 안에서의 분류 (correct + within-sentiment confusion)
        within = block[:, idxs].sum()
        # 다른 sentiment 그룹으로 leak
        cross = total - within

        # 정확한 emotion 까지 맞춘 비율
        correct = sum(cm[i, i] for i in idxs)
        within_confusion = within - correct
        cross_confusion = cross

        result[sent] = dict(
            total=int(total),
            correct=int(correct),
            within_sent_confusion=int(within_confusion),
            cross_sent_leak=int(cross_confusion),
            within_pct=round(within_confusion / total * 100, 1),
            cross_pct=round(cross_confusion / total * 100, 1),
            correct_pct=round(correct / total * 100, 1),
        )
    return result


# =========================================================
# Main analysis
# =========================================================
def analyze_cm(cm, emo_labels, dataset, name="model"):
    """전체 분석: emotion-level + sentiment-level + within-sent confusion."""
    print(f"\n{'='*60}")
    print(f"{name} on {dataset.upper()}")
    print(f"{'='*60}")

    cm = np.asarray(cm, dtype=int)
    e2s_map = SENT_MAPS[dataset.lower().replace("-", "")]

    # 1. Emotion-level
    emo_m = metrics_from_cm(cm, emo_labels)
    print(f"\n[Emotion-level] ({len(emo_labels)}-class)")
    print(f"  weighted-F1 = {emo_m['weighted_f1']:.2f}")
    print(f"  macro-F1    = {emo_m['macro_f1']:.2f}")
    print(f"  per-class F1:")
    for emo, f1 in emo_m["per_class_f1"].items():
        print(f"    {emo:<12s} F1={f1:>6.2f}  support={emo_m['per_class_support'][emo]:>5d}")

    # 2. Sentiment-level
    cm_sent, sent_labels = collapse_cm(cm, emo_labels, e2s_map)
    sent_m = metrics_from_cm(cm_sent, sent_labels)
    print(f"\n[Sentiment-level] ({len(sent_labels)}-class collapse)")
    print(f"  weighted-F1 = {sent_m['weighted_f1']:.2f}")
    print(f"  macro-F1    = {sent_m['macro_f1']:.2f}")
    print(f"  per-sentiment F1:")
    for sent, f1 in sent_m["per_class_f1"].items():
        prec = sent_m["per_class_precision"][sent]
        rec = sent_m["per_class_recall"][sent]
        sup = sent_m["per_class_support"][sent]
        print(f"    {sent:<10s} F1={f1:>6.2f}  P={prec:>5.2f}  R={rec:>5.2f}  N={sup:>5d}")

    # 3. Gap
    print(f"\n[Gap: sentiment − emotion]")
    print(f"  weighted-F1 gap = {sent_m['weighted_f1'] - emo_m['weighted_f1']:+.2f}")
    print(f"  macro-F1    gap = {sent_m['macro_f1'] - emo_m['macro_f1']:+.2f}")

    # 4. Within-sentiment confusion
    print(f"\n[Within-sentiment confusion breakdown]")
    print(f"  (각 true sentiment group 의 분류 결과)")
    print(f"  {'sentiment':<12s} {'correct':>8s}  {'within-sent':>12s}  {'cross-sent':>12s}")
    print(f"  {'-'*55}")
    wsc = analyze_within_sentiment_confusion(cm, emo_labels, e2s_map)
    for sent, stats in wsc.items():
        print(f"  {sent:<12s} {stats['correct_pct']:>7.1f}%  "
              f"{stats['within_pct']:>11.1f}%  {stats['cross_pct']:>11.1f}%  (N={stats['total']})")

    # 5. Sentiment-level confusion matrix print
    print(f"\n[Sentiment-level confusion matrix]")
    print(f"  rows=true, cols=pred (row-normalized %)")
    rowsum = cm_sent.sum(axis=1, keepdims=True)
    cm_sent_norm = np.divide(cm_sent, rowsum, out=np.zeros_like(cm_sent, dtype=float),
                              where=rowsum > 0) * 100
    header = "  " + " "*10 + "  ".join(f"{s:>9s}" for s in sent_labels)
    print(header)
    for i, sent_true in enumerate(sent_labels):
        row = "  ".join(f"{cm_sent_norm[i, j]:>8.1f}%" for j in range(len(sent_labels)))
        print(f"  {sent_true:<10s} {row}")

    return dict(emotion=emo_m, sentiment=sent_m, within=wsc, cm_sent=cm_sent_norm)


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm", type=str, required=True,
                        help="Confusion matrix path (.npy or .csv)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Comma-separated emotion labels in CM order, e.g., 'ang,exc,fru,hap,neu,sad'")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "meld", "kmind", "k-mind"])
    parser.add_argument("--name", type=str, default="Stage 2 ELACL")
    args = parser.parse_args()

    if args.cm.endswith(".npy"):
        cm = np.load(args.cm)
    elif args.cm.endswith(".csv"):
        import pandas as pd
        cm = pd.read_csv(args.cm, index_col=0).values
    else:
        raise ValueError("CM file must be .npy or .csv")

    emo_labels = [l.strip() for l in args.labels.split(",")]
    assert cm.shape == (len(emo_labels), len(emo_labels)), \
        f"CM shape {cm.shape} != labels count {len(emo_labels)}"

    analyze_cm(cm, emo_labels, args.dataset, args.name)


if __name__ == "__main__":
    main()
