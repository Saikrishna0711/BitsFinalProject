# Emotion Recognition in Customer Calls

> **Real-time audio → speaker diarization → ASR → RCNN-Attention emotion tagging**  
> Macro-F1 0.83 · Real-Time Factor 0.12 on a GTX 1650 · End-to-end in < 30 s for a 2-min call.

---

## ✨ Key Features
| Stage | Open-Source Backbone | Latency (2-min call) | Notes |
|-------|----------------------|----------------------|-------|
| VAD & Segmentation | WebRTC VAD | 0.26 s | configurable padding |
| Speaker Diarization | `pyannote/speaker-diarization` | 16.8 s | DER ≈ 11 % |
| ASR | **faster-whisper Large-v3** | 9.9 s | WER ≈ 9.4 % |
| Emotion Classifier | **RCNN-Attention** (1.5 M params) | 1.2 s | 7-class |

* Lightweight: 4 GB GPU RAM fits full pipeline  
* Streaming-friendly: emits segment emotions every 1 s  
* React dashboard (optional) for live heat-maps

---

## 🏗️ Repository Structure
├── src/
│ ├── asr_diarize.py # VAD + diarizer + Whisper wrapper
│ ├── hf_emotion.py # RCNN-Attn inference
│ ├── pipeline_emotion.py # orchestration script
│ ├── rcnn_attention.py # model definition
│ └── log.py # unified logger
├── models/ # checkpoints (or HF IDs)
├── data/ # example wav & transcripts
├── docker/ # Dockerfile & helm chart
└── README.md # you are here


---

## 🚀 Quick Start (GPU)

```bash
git clone https://github.com/your-org/emo-pipeline.git
cd emo-pipeline
pip install -r requirements.txt          # Python 3.11 + PyTorch >= 2.5

# Download IEMOCAP sample (already included in data/) and run:
python src/pipeline_emotion.py \
    data/sample_call.wav \
    --hf-token <your_HF_token_for_pyannote> \
    --device cuda

✓ diarized transcript → sample_call.dia.json   (segments=44)
✓ emotions → sample_call.dia_emo.json
✓ report saved to sample_call_report.txt
