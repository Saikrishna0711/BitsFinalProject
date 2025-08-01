#!/usr/bin/env python3
"""
Fast ASR + speaker diarization
  python src/asr_diarize.py wav out.json <HF_TOKEN> --device cuda --model small.en
Output = list[dict]  with  {start,end,speaker,text}
"""
import argparse, json, pathlib, shutil, sys, torch
from log import log, timed

# --- Functions (These can be safely imported by app.py) ---

@timed
def run_whisper(wav: str, model_size: str = "small.en", device: str = "cuda"):
    from faster_whisper import WhisperModel
    compute_type = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segs, _ = model.transcribe(
        wav,
        vad_filter=False,
        word_timestamps=False,
        beam_size=5,
        best_of=1,
    )
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]

@timed
def run_diarization(wav: str, hf_token: str, device: str = "cuda"):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))
    return pipeline({"audio": wav})

@timed
def merge(segments, dia):
    out = []
    for seg in segments:
        spk = "UNK"
        for track, _, who in dia.itertracks(yield_label=True):
            if track.start <= seg["start"] and seg["end"] <= track.end:
                spk = who
                break
        out.append({**seg, "speaker": spk})
    return out


# --- Main execution block (for command-line use only) ---
# This part will NOT run when app.py imports the functions above.
# It only runs if you execute "python src/asr_diarize.py ..."
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("wav")
    P.add_argument("out_json")
    P.add_argument("hf_token")
    P.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    P.add_argument("--model",  default="small.en",
                   help="faster-whisper checkpoint; tiny.en/small.en/…")
    args = P.parse_args()

    DEVICE = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    if shutil.which("ffmpeg") is None:
        sys.exit("⚠  ffmpeg not in PATH – install or add to PATH first")

    txt_segs = run_whisper(args.wav, model_size=args.model, device=str(DEVICE))
    dia      = run_diarization(args.wav, hf_token=args.hf_token, device=str(DEVICE))
    merged   = merge(txt_segs, dia)

    json.dump(merged, open(args.out_json, "w"), indent=2)
    log.info("✓ diarized transcript → %s  (segments=%d)", args.out_json, len(merged))