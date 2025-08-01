#!/usr/bin/env python3
"""
Predict emotion for a single .npz feature file.

Run:
python src/predict_one.py \
       --model best_model.pth \
       --npz   data/features/iemocap/iemocap_Ses01F_impro01_F000.npz \
       --device cpu      # or cuda
"""
import argparse, pathlib, torch, numpy as np, sys
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
from model import RCNNEmotionNet

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    spec = torch.tensor(z["acoustics"], dtype=torch.float32).unsqueeze(0)  # (1,T,F)
    txt  = torch.tensor(z["embed"],      dtype=torch.float32).unsqueeze(0)  # (1,E)
    return spec, txt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--npz",   required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device)
    model  = RCNNEmotionNet(n_classes=len(ckpt["label2id"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    spec, txt = load_npz(args.npz)
    spec, txt = spec.to(device), txt.to(device)
    with torch.no_grad():
        logits = model(spec, txt)
        pred_id = logits.argmax(1).item()
        # invert the label2id map
        id2label = {v: k for k, v in ckpt["label2id"].items()}
        print("Predicted emotion:", id2label[pred_id])
