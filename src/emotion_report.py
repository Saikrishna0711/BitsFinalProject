#!/usr/bin/env python3
"""
Unified pipeline (audio→diarize→emotion) with integrated report generation.

Usage:
    python src/pipeline_emotion.py input.wav \
        --hf-token TOKEN [--device cuda] [--out-dir .] [--top N] [--samples M]
    python src/pipeline_emotion.py transcript.srt \
        [--device cpu] [--out-dir .] [--top N] [--samples M]

Outputs in --out-dir (defaults to current):
  <basename>.dia.json      (diarized segments)
  <basename>_emo.json      (with emotions)
  <basename>_report.txt    (full table + summary)
"""
import argparse, pathlib, subprocess, sys, torch
from log import log
from transcript2json import convert as transcript2json

# locate scripts in src/
here = pathlib.Path(__file__).parent
asr_script    = here / "asr_diarize.py"
emo_script    = here / "hf_emotion.py"
report_script = here / "emotion_report.py"

# CLI arguments
P = argparse.ArgumentParser()
P.add_argument("source", help=".wav OR transcript (.srt/.vtt/.json)")
P.add_argument("--hf-token", help="HuggingFace token for diarization when using audio")
P.add_argument("--device", choices=["cuda","cpu"], default="cuda",
               help="Device for models")
P.add_argument("--out-dir", default='.',
               help="Directory to save outputs")
P.add_argument("--top",   type=int, default=3,
               help="Top N emotions for summary report")
P.add_argument("--samples", type=int, default=5,
               help="Sample utterances per emotion in summary")
args = P.parse_args()

# resolve device
DEVICE = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

# prepare paths
src_path = pathlib.Path(args.source)
out_dir  = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
base = src_path.stem

dia_json    = out_dir / f"{base}.dia.json"
emo_json    = out_dir / f"{base}_emo.json"
report_file = out_dir / f"{base}_report.txt"

# 1) ASR + diarization or transcript conversion
if src_path.suffix.lower() == ".wav":
    if not args.hf_token:
        sys.exit("Error: --hf-token required for audio diarization")
    subprocess.check_call([
        sys.executable, str(asr_script),
        str(src_path), str(dia_json), args.hf_token,
        "--device", DEVICE
    ])
else:
    # convert transcript to dia.json
    dia_json = pathlib.Path(transcript2json(str(src_path)))

# 2) Emotion inference
subprocess.check_call([
    sys.executable, str(emo_script),
    str(dia_json), "--device", DEVICE
])

# 3) Generate report via emotion_report.py, capture and save output
cmd = [
    sys.executable, str(report_script),
    str(emo_json),
    "--top", str(args.top),
    "--samples", str(args.samples)
]
result = subprocess.run(cmd, capture_output=True, text=True)
report = result.stdout
# save to file
report_file.write_text(report, encoding='utf-8')
log.info("✓ report saved to %s", report_file)
# print to console
print(report)

log.info("✓ Completed pipeline for %s", src_path.name)
