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
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))
    return pipeline({"audio": wav}, min_speakers=2, max_speakers=4)

@timed
def merge(segments, dia):
    """
    Merges transcription segments with speaker diarization using the maximum overlap principle.

    Args:
        segments (list of dict): A list of transcription segments, e.g.,
                                [{'start': 0.0, 'end': 7.0, 'text': '...'}, ...].
        dia (pyannote.core.Annotation): Speaker diarization output.

    Returns:
        list of dict: The list of segments with a 'speaker' key added.
    """
    out = []
    for seg in segments:
        best_speaker = "UNK"
        max_overlap = 0

        # Iterate through each speaker turn in the diarization
        for track, _, speaker_label in dia.itertracks(yield_label=True):
            # Calculate the overlapping time interval between the segment and the speaker turn.
            # The overlap starts at the maximum of the two start times.
            overlap_start = max(seg['start'], track.start)
            
            # The overlap ends at the minimum of the two end times.
            overlap_end = min(seg['end'], track.end)

            # Calculate the duration of the overlap
            overlap_duration = overlap_end - overlap_start

            # If there's a positive overlap and it's the largest we've seen for this segment,
            # update the best speaker.
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker_label
        
        # Assign the best speaker (the one with the most overlap) to the segment.
        out.append({**seg, "speaker": best_speaker})
       
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
    print(dia)
    merged   = merge(txt_segs, dia)

    json.dump(merged, open(args.out_json, "w"), indent=2)
    log.info("✓ diarized transcript → %s  (segments=%d)", args.out_json, len(merged))