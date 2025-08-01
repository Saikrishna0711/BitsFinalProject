#!/usr/bin/env python3
"""
Central logger + @timed decorator.

Import once in every script:
    from log import log, timed
"""
import logging, functools, time, os, torch, platform

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("emo‑pipe")

# One‑liner device info on start‑up
log.info("Host: %s  |  torch %s  |  GPU: %s",
         platform.node(), torch.__version__,
         torch.cuda.get_device_name() if torch.cuda.is_available() else "None")

def timed(fn):
    @functools.wraps(fn)
    def _wrap(*a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        log.info("%s took %.2fs", fn.__name__, time.perf_counter() - t0)
        return out
    return _wrap
