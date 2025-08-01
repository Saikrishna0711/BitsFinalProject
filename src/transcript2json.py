#!/usr/bin/env python3
"""
SRT / WebVTT / our‑JSON transcript → dia.json (start,end,speaker,text)
"""
import json, pathlib, re, typing as T
from webvtt import WebVTT
import pysrt
from log import log, timed

def _parse_srt(path:str) -> T.List[dict]:
    subs = pysrt.open(path)
    out=[]
    for s in subs:
        start = s.start.ordinal / 1000
        end   = s.end.ordinal / 1000
        txt = s.text.strip().replace("\n"," ")
        spk = "UNK"
        if ":" in txt[:30]:
            spk, txt = txt.split(":",1); spk = spk.strip()
        out.append({"start":start,"end":end,"speaker":spk,"text":txt.strip()})
    return out

def _parse_vtt(path:str) -> T.List[dict]:
    out=[]
    for c in WebVTT().read(path).captions:
        h,m,s = re.split("[:,.]", c.start); start=int(h)*3600+int(m)*60+float(s+"."+c.start.split(".")[-1])
        h,m,s = re.split("[:,.]", c.end);   end  =int(h)*3600+int(m)*60+float(s+"."+c.end.split(".")[-1])
        txt=c.text.strip().replace("\n"," "); spk="UNK"
        if ":" in txt[:30]:
            spk, txt = txt.split(":",1); spk = spk.strip()
        out.append({"start":start,"end":end,"speaker":spk,"text":txt.strip()})
    return out

@timed
def convert(src:str) -> str:
    p = pathlib.Path(src)
    if p.suffix == ".json":
        data = json.load(open(p))
    elif p.suffix == ".srt":
        data = _parse_srt(src)
    elif p.suffix in {".vtt",".webvtt"}:
        data = _parse_vtt(src)
    else:
        raise SystemExit("Unsupported transcript format: "+src)
    out = p.with_suffix(".dia.json")
    json.dump(data, open(out,"w"), indent=2)
    log.info("✓ transcript → %s  (segments=%d)", out, len(data))
    return str(out)

# CLI helper
if __name__ == "__main__":
    import sys; convert(sys.argv[1])
