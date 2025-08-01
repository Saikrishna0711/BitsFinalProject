#!/usr/bin/env python3
"""
Convert IEMOCAP (raw) and MELD (HF arrow) into harmonised .npz files.

Usage:
python src/prepare_data.py \
       --iemocap-root datasets/IEMOCAP_full_release \
       --meld-root    datasets/meld \
       --out-dir      data/features
"""
import argparse, pathlib, json, numpy as np, librosa, tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from features import acoustic_features

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def save_npz(out, feats, emb, text, label, ds):
    np.savez(out, acoustics=feats.astype(np.float32), embed=emb.astype(np.float32),
             text=text, label=label, dataset=ds)

def process_iemocap(root, out_dir):
    import re
    utt2lab, utt2txt = {}, {}
    pat = re.compile(r"\\[(\\d+\\.\\d+) - (\\d+\\.\\d+)]\\s+(.+)\\s+([a-z]{3})")
    for ses in root.glob("Session*"):
        for f in (ses/"dialog/EmoEvaluation").glob("*.txt"):
            for l in f.read_text().splitlines():
                m = pat.match(l);   # utt-id & 3-char label
                if m: utt2lab[m.group(3)] = m.group(4)
        for f in (ses/"dialog/transcriptions").glob("*.txt"):
            for l in f.read_text().splitlines():
                if " " in l: uid, txt = l.split(" ", 1); utt2txt[uid] = txt
        for wav in (ses/"sentences/wav").rglob("*.wav"):
            uid = wav.stem
            if uid not in utt2lab: continue
            feats = acoustic_features(str(wav))
            emb   = embedder.encode(utt2txt.get(uid, ""), normalize_embeddings=True)
            save_npz(out_dir/f"iemocap_{uid}.npz", feats, emb,
                     utt2txt.get(uid,""), utt2lab[uid], "iemocap")

def process_meld(meld_root, out_dir):
    ds = load_from_disk(meld_root)
    for split in ["train","validation","test"]:
        if split not in ds: continue
        for ex in tqdm.tqdm(ds[split], desc=f"MELD {split}"):
            wav = pathlib.Path(ex["audio"]["path"])
            feats = acoustic_features(str(wav))
            emb   = embedder.encode(ex["text"], normalize_embeddings=True)
            uid   = wav.stem
            save_npz(out_dir/f"meld_{split}_{uid}.npz", feats, emb,
                     ex["text"], ex["emotion"], f"meld-{split}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iemocap-root", type=pathlib.Path)
    p.add_argument("--meld-root",   type=pathlib.Path)
    p.add_argument("--out-dir",     type=pathlib.Path, default=pathlib.Path("data/features"))
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.iemocap_root: process_iemocap(args.iemocap_root, args.out_dir)
    if args.meld_root:    process_meld(args.meld_root,   args.out_dir)
