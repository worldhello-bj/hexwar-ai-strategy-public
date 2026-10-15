"""One checkpoint, one subprocess per measurement; never run training/evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from env.neural_train import NeuralTrainer

p = argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('--snapshot', default='scenario/7527')
p.add_argument('--save')
a = p.parse_args()
t = time.perf_counter()
trainer = NeuralTrainer.load(a.checkpoint, a.snapshot, allow_real=True)
load_seconds = time.perf_counter() - t
load_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
result = dict(checkpoint=a.checkpoint, load_seconds=load_seconds, load_peak_rss_bytes=load_rss,
              file_bytes=Path(a.checkpoint).stat().st_size, episodes=trainer.episodes_completed)
if a.save:
    t = time.perf_counter(); trainer.save(a.save)
    result.update(save_seconds=time.perf_counter()-t,
                  load_and_save_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  saved_bytes=Path(a.save).stat().st_size)
# Streaming exact state digest, no model .tolist conversion during measurement.
def digest(value):
    import numpy as np
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,np.ndarray):
            h.update(str(v.shape).encode()); h.update(str(v.dtype).encode()); h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v): h.update(k.encode()); walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v: walk(x)
        else: h.update(json.dumps(v,allow_nan=False).encode())
    walk(value); return h.hexdigest()
result['state_sha256']=digest(trainer.checkpoint(arrays=True))
print(json.dumps(result,sort_keys=True))
