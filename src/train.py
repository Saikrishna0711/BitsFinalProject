#!/usr/bin/env python3
"""
train.py  ·  v2
Handles string emotion labels by mapping them to integer indices at runtime.

Usage
-----
python src/train.py \
    --train-dir data/features/iemocap \
    --val-dir   data/features/iemocap \
    --batch-size 16 \
    --epochs     20 \
    --lr         3e-4 \
    --device     cuda \
    --checkpoint best_model.pth
"""
import sys, os, glob, argparse, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
from model import RCNNEmotionNet

# ─── Dataset -----------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, root_dir, label2id=None):
        self.files = glob.glob(os.path.join(root_dir, "*.npz"))
        # Determine / share label mapping
        if label2id is None:
            labels = sorted({np.load(f, allow_pickle=True)["label"].item() for f in self.files})
            self.label2id = {lab: idx for idx, lab in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        spec = torch.tensor(data["acoustics"], dtype=torch.float32)
        txt  = torch.tensor(data["embed"], dtype=torch.float32)
        lab  = data["label"].item()
        lab  = self.label2id[lab]  # str -> int
        return spec, txt, lab

# ─── Collate -----------------------------------------------------------------

def collate_pad(batch):
    specs, txts, labs = zip(*batch)
    max_len = max(s.shape[0] for s in specs)
    Fdim = specs[0].shape[1]
    padded = torch.zeros(len(specs), max_len, Fdim)
    for i, s in enumerate(specs):
        padded[i, : s.shape[0]] = s
    return padded, torch.stack(txts), torch.tensor(labs)

# ─── Train / Val -------------------------------------------------------------

def train_epoch(model, loader, optim, device):
    model.train(); total=0; n=0
    for spec, txt, lab in loader:
        spec, txt, lab = spec.to(device), txt.to(device), lab.to(device)
        loss = F.cross_entropy(model(spec, txt), lab)
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item() * spec.size(0); n += spec.size(0)
    return total / n


def eval_epoch(model, loader, device):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for spec, txt, lab in loader:
            spec, txt, lab = spec.to(device), txt.to(device), lab.to(device)
            pred = model(spec, txt).argmax(1)
            correct += (pred == lab).sum().item(); total += lab.size(0)
    return correct / total

# ─── Main --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir",   required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build datasets with shared label map
    train_ds = EmotionDataset(args.train_dir)
    val_ds   = EmotionDataset(args.val_dir, label2id=train_ds.label2id)
    print("Label mapping:", train_ds.label2id)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_pad, num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=4)

    model = RCNNEmotionNet(n_classes=len(train_ds.label2id)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best=0.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_ld, optim, device)
        val_acc = eval_epoch(model, val_ld, device)
        print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best:
            best=val_acc
            torch.save({"state_dict": model.state_dict(), "label2id": train_ds.label2id}, args.checkpoint)
    print("Best val acc:", best)