#!/usr/bin/env python3
"""
Unified pipeline: audio (.wav) or transcript (.srt/.vtt/.json) ⇒ diarize ⇒ emotion ⇒ speaker-emotion report.

Outputs (in --out-dir, default current):
  <base>.dia.json          diarized segments
  <base>.dia_emo.json      segments + emotion labels
  <base>_report.txt        human-readable report:
      • full chronologic table (start-end, speaker, emotion, text)
      • per-speaker, per-emotion breakdown with sentences & timings

Example run:
    python src/pipeline_emotion.py data/Ses01F_impro01.wav \
        --hf-token hf_XXXX --device cuda --out-dir . --top 4 --samples 3
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys, torch, json, textwrap
from log import log
from transcript2json import convert as transcript2json
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Locate helper scripts in the same folder
# ---------------------------------------------------------------------------
here          = pathlib.Path(__file__).parent
asr_script    = here / "asr_diarize.py"
emo_script    = here / "hf_emotion.py"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="End-to-end emotion pipeline")
parser.add_argument("source", help="Input .wav OR transcript (.srt/.vtt/.json)")
parser.add_argument("--hf-token", help="HF token (required for .wav input)")
parser.add_argument("--device", choices=["cuda","cpu"], default="cuda",
                    help="Device to run models (default cuda if available)")
parser.add_argument("--out-dir", default=".", help="Where to write outputs")
args = parser.parse_args()

DEVICE = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
log.info("Running on %s", DEVICE)

# ---------------------------------------------------------------------------
# Prepare paths
# ---------------------------------------------------------------------------
src_path = pathlib.Path(args.source).expanduser().resolve()
base     = src_path.stem
out_dir  = pathlib.Path(args.out_dir).expanduser().resolve()
out_dir.mkdir(parents=True, exist_ok=True)

dia_json_path = out_dir / f"{base}.dia.json"
emo_json_path = out_dir / f"{base}.dia_emo.json"
report_path   = out_dir / f"{base}_report.txt"

# ---------------------------------------------------------------------------
# 1) Diarize or convert transcript
# ---------------------------------------------------------------------------
if src_path.suffix.lower() == ".wav":
    if not args.hf_token:
        sys.exit("Error: --hf-token required for audio input")
    subprocess.check_call([
        sys.executable, str(asr_script),
        str(src_path), str(dia_json_path), args.hf_token,
        "--device", DEVICE
    ])
else:
    dia_json_path = pathlib.Path(transcript2json(str(src_path)))

# ---------------------------------------------------------------------------
# 2) Emotion inference
# ---------------------------------------------------------------------------
subprocess.check_call([
    sys.executable, str(emo_script),
    str(dia_json_path), "--device", DEVICE
])

# Find the produced *_emo.json (hf_emotion appends "_emo.json")
emo_json_path = dia_json_path.with_name(dia_json_path.stem + "_emo.json")

# ---------------------------------------------------------------------------
# 3) Build report
# ---------------------------------------------------------------------------
data = json.load(open(emo_json_path, encoding="utf-8"))

lines: list[str] = []
lines.append("FULL CHRONOLOGICAL TABLE")
lines.append("Start    End      Spk  Emotion   Text")
lines.append("------   ------   ---  --------  ----")
for seg in data:
    lines.append(f"{seg['start']:6.2f}–{seg['end']:6.2f}  {seg['speaker']:<3s}  {seg['emotion']:<8s}  {textwrap.shorten(seg['text'],60)}")

# Group by speaker → emotion
spk_grp: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
for seg in data:
    spk_grp[seg['speaker']][seg['emotion']].append(seg)

lines.append("\nSPEAKER-EMOTION BREAKDOWN")
for spk in sorted(spk_grp):
    lines.append(f"\nSpeaker {spk}:")
    emo_counts = Counter({e: len(lst) for e,lst in spk_grp[spk].items()})
    for emo, cnt in emo_counts.most_common():
        lines.append(f"  {emo.upper():<8s} ({cnt})")
        for s in spk_grp[spk][emo]:
            lines.append(f"    [{s['start']:.2f}–{s['end']:.2f}] {s['text']}")

# Write & print
report_path.write_text("\n".join(lines), encoding="utf-8")
log.info("✓ Report saved to %s", report_path)
print("\n".join(lines))

log.info("✓ Pipeline complete for %s", src_path.name)