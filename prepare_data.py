#!/usr/bin/env python3
"""
prepare_data.py  ·  v3
Turns MELD (HF Arrow) + raw IEMOCAP_full_release into harmonised .npz
feature files. Fixes MELD FLAC path resolution for audio files.

Run:
python prepare_data.py \
       --meld-path      data/meld \
       --iemocap-path   data/iemocap/IEMOCAP_full_release \
       --out-dir        data/features
"""
import sys, argparse, json, pathlib, re
import numpy as np, librosa, tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from features import acoustic_features  # ensure src/ is on PYTHONPATH

LABEL_MAP = {
    "neu": "neutral", "neutral": "neutral",
    "sad": "sad", "ang": "angry", "hap": "happy", "exc": "happy",
    "fru": "frustrated", "dis": "disgust",
}

def iter_iemocap_sessions(root: pathlib.Path):
    ses_dirs = [d for d in root.iterdir() if d.name.startswith("Session")]
    eval_re = re.compile(r"\[(\d+\.\d+) - (\d+\.\d+)]\s+(.+)\s+([a-z]{3})")
    for ses in ses_dirs:
        # parse labels
        utt2label = {}
        for txt in (ses / "dialog" / "EmoEvaluation").glob("*.txt"):
            for line in txt.read_text().splitlines():
                m = eval_re.match(line)
                if m: utt2label[m.group(3)] = LABEL_MAP.get(m.group(4), m.group(4))
        # parse transcripts
        utt2txt = {}
        for trn in (ses / "dialog" / "transcriptions").glob("*.txt"):
            for line in trn.read_text(encoding="utf8").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts)==2: utt2txt[parts[0]] = parts[1]
        # yield wavs
        for wav in (ses / "sentences" / "wav").rglob("*.wav"):
            uid = wav.stem
            lab = utt2label.get(uid)
            if not lab: continue
            yield wav, utt2txt.get(uid, ""), lab

# Process a generic (wav_path, text, label) iterator

def process_items(items, name, out_dir: pathlib.Path, embedder, sr=16_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for wav_path, text, label in tqdm.tqdm(items, desc=name):
        feats = acoustic_features(str(wav_path), sr=sr)
        emb   = embedder.encode(text, normalize_embeddings=True)
        uid = f"{name}_{wav_path.stem}"
        npz = out_dir / f"{uid}.npz"
        np.savez(npz,
                 acoustics=feats.astype(np.float32),
                 embed=emb.astype(np.float32),
                 text=text, label=label, dataset=name)
        manifest.append({"utt_id": uid,
                         "npz": str(npz),
                         "wav": str(wav_path),
                         "label": label,
                         "text": text})
    (out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meld-path", type=str, help="HF-saved MELD dir")
    p.add_argument("--iemocap-path", type=str, help="IEMOCAP_full_release dir")
    p.add_argument("--out-dir", type=str, default="data/features")
    args = p.parse_args()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if args.meld_path:
        base = pathlib.Path(args.meld_path)
        ds = load_from_disk(base)
        for split in ["train", "validation", "test"]:
            if split in ds:
                split_dir = base / split
                def gen_meld():
                    for ex in ds[split]:
                        audio = ex["audio"]
                        rel = audio["path"] if isinstance(audio, dict) else audio
                        # build full path
                        fp = pathlib.Path(rel)
                        if not fp.is_absolute():
                            # find file under split_dir
                            candidates = list(split_dir.rglob(fp.name))
                            if candidates: fp = candidates[0]
                        yield fp, ex.get("text",""), ex.get("emotion")
                process_items(gen_meld(), f"meld-{split}",
                              pathlib.Path(args.out_dir)/"meld"/split,
                              embedder)

    if args.iemocap_path:
        process_items(iter_iemocap_sessions(pathlib.Path(args.iemocap_path)),
                      "iemocap", pathlib.Path(args.out_dir)/"iemocap", embedder)
