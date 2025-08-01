#!/usr/bin/env python3
"""
Local, batched RoBERTa emotion classifier
  python src/hf_emotion.py dia.json  --device cuda
Adds .emotion field to every segment and writes *_emo.json
"""
import argparse, json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from log import log, timed

tok = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
mdl = AutoModelForSequenceClassification.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base").eval()

@timed
def classify(texts, device, bs=32):
    mdl.to(device)
    labels=[]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt",
                  padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
        labels.extend([mdl.config.id2label[p] for p in logits.argmax(1).tolist()])
    return labels

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("dia_json")
    P.add_argument("--device", default="cuda")
    args = P.parse_args()

    data = json.load(open(args.dia_json))
    emos = classify([d["text"] for d in data],
                    device="cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    for d,e in zip(data, emos):
        d["emotion"] = e.lower()
    out = pathlib.Path(args.dia_json).with_name(pathlib.Path(args.dia_json).stem + "_emo.json")
    json.dump(data, open(out,"w"), indent=2)
    log.info("✓ emotions → %s", out)
