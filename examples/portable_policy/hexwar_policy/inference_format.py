"""Inference-only numeric NPZ v1. No optimizer, training RNG or resume state."""
import json
from pathlib import Path
import zipfile
import numpy as np
from .neural_policy import NeuralPolicy

FORMAT='hexwar-inference-weights'
MAX_BYTES=64*1024*1024

def save_weights(path, policy):
    metadata=json.dumps({'format':FORMAT,'version':1,'config':policy.config},
                        allow_nan=False,separators=(',',':')).encode()
    with open(path,'xb') as stream:
        np.savez_compressed(stream,metadata=np.frombuffer(metadata,dtype=np.uint8),**policy.params)

def load_weights(path):
    if Path(path).stat().st_size>MAX_BYTES: raise ValueError('Oversized model')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=8 or len(set(names))!=8 or sum(x.file_size for x in archive.infolist())>MAX_BYTES:
            raise ValueError('Invalid inference archive')
        for info in archive.infolist():
            if info.flag_bits&1 or info.file_size>MAX_BYTES: raise ValueError('Invalid member')
    with np.load(path,allow_pickle=False) as data:
        m=data['metadata']
        if m.dtype!=np.uint8 or m.ndim!=1 or m.nbytes>65536: raise ValueError('Invalid metadata')
        meta=json.loads(m.tobytes().decode())
        if set(meta)!={'format','version','config'} or meta['format']!=FORMAT or meta['version']!=1:
            raise ValueError('Unsupported inference format')
        config=meta['config']
        if any(type(config.get(k)) is not int or not 1<=config[k]<=1048576 for k in ('buckets','hidden_dim')):
            raise ValueError('Invalid dimensions')
        if config['buckets']*config['hidden_dim']*16>MAX_BYTES: raise ValueError('Oversized parameters')
        policy=NeuralPolicy(**config)
        if set(data.files)!={'metadata',*policy.params}: raise ValueError('Parameter keys differ')
        for key,expected in policy.params.items():
            value=data[key]
            if value.dtype!=np.float64 or value.shape!=expected.shape or not np.isfinite(value).all():
                raise ValueError('Invalid parameter tensor')
            policy.params[key]=value.copy()
    # Constructor supplies zero optimizer slots and a fresh seeded RNG solely
    # for API compatibility. They are not restored training state.
    return policy
