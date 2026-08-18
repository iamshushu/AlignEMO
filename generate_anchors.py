"""Generate per-emotion text anchor embeddings (RoBERTa label embeddings) for IEMOCAP and MELD."""
import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import argparse
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', type=str, default='roberta-large')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = AutoModel.from_pretrained(args.text_model)

    save_path = args.text_model.split("/")[-1]
    iemocap_emos = [
        "neutral",
        "excited",
        "frustrated",
        "sad",
        "happy",
        "angry"
    ]

    meld_emos = [
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'surprise',
        'neutral'
    ]

    embeddings = []
    feature_extractor = pipeline("feature-extraction",framework="pt",model=args.text_model)

    iemocap_dir = f"./IEMOCAP/emo_anchors/{save_path}"
    meld_dir = f"./MELD/emo_anchors/{save_path}"
    os.makedirs(iemocap_dir, exist_ok=True)
    os.makedirs(meld_dir, exist_ok=True)

    with torch.no_grad():
        for emo in iemocap_emos:
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, f"{iemocap_dir}/iemocap_emo.pt")
        print(f"saved: {iemocap_dir}/iemocap_emo.pt  shape={tuple(embeddings.shape)}")

    embeddings = []
    with torch.no_grad():
        for emo in meld_emos:
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, f"{meld_dir}/meld_emo.pt")
        print(f"saved: {meld_dir}/meld_emo.pt  shape={tuple(embeddings.shape)}")