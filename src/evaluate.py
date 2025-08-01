#!/usr/bin/env python3
import argparse, glob, torch, numpy as np, pathlib, sys
from torch.utils.data import DataLoader
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0,str(project_root))
from model import RCNNEmotionNet
from train import EmotionDataset, collate_pad   # reuse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ckpt = torch.load(args.model, map_location=args.device)
    model = RCNNEmotionNet(n_classes=len(ckpt["label2id"])); model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()
    ds = EmotionDataset(args.test_dir, label2id=ckpt["label2id"])
    acc = 0; loader = DataLoader(ds,batch_size=32,collate_fn=collate_pad, num_workers=4)
    n=0
    with torch.no_grad():
        for spec, txt, lab in loader:
            spec,txt,lab=spec.to(args.device),txt.to(args.device),lab.to(args.device)
            pred = model(spec,txt).argmax(1)
            acc += (pred==lab).sum().item(); n+=lab.size(0)
    print(f"Test accuracy: {acc/n:.4f}")
