#!/usr/bin/env python3
"""Explicit offline backend migration into NEW files; never changes a source checkpoint.

Only an empty-pending, completed optimizer batch can migrate. All learned arrays,
optimizer moments, RNG states, counters, rewards and episode settings remain exact.
Changing accumulation order may change future floating-point trajectories slightly.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.neural_train import NeuralTrainer, canonical

def digest(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()

def migrate(source, destination, snapshot, *, allow_real=False):
    source,destination=Path(source),Path(destination)
    audit_path=destination.with_suffix(destination.suffix+'.migration.json')
    if source.resolve()==destination.resolve() or destination.exists() or audit_path.exists():
        raise ValueError('choose new destination and audit paths; source is immutable')
    before=source.read_bytes()
    trainer=NeuralTrainer.load(source,snapshot,allow_real=allow_real)
    if trainer.pending or trainer.episodes_completed == 0:
        raise ValueError('migration requires completed episodes and an empty pending batch')
    if trainer.policy.config.get('projection_backend','numpy')!='numpy':
        raise ValueError('source must use the numpy backend')
    original=json.loads(before)
    data=copy.deepcopy(original)
    data['policy_config']['projection_backend']='scipy_csr'
    for key in ('policy','initial_policy'):
        data[key]['config']['projection_backend']='scipy_csr'
    # Validate through the same loader without touching source files.
    trainer.policy_config['projection_backend']='scipy_csr'
    from env.neural_policy import NeuralPolicy
    trainer.policy=NeuralPolicy.from_checkpoint(data['policy'])
    trainer.initial_policy=copy.deepcopy(data['initial_policy'])
    expected=copy.deepcopy(original)
    expected['policy_config']['projection_backend']='scipy_csr'
    for key in ('policy','initial_policy'):expected[key]['config']['projection_backend']='scipy_csr'
    if data!=expected or trainer.checkpoint()!=data:
        raise ValueError('migration changed more than backend configuration')
    if source.read_bytes()!=before:
        raise ValueError('source checkpoint changed during validation')
    payload=(canonical(data)+'\n').encode()
    audit={'schema_version':1,'migration':'numpy_to_scipy_csr',
           'parent_checkpoint_sha256':hashlib.sha256(before).hexdigest(),
           'output_checkpoint_sha256':hashlib.sha256(payload).hexdigest(),
           'scoring_version':data['scoring_version'],'snapshot':data['snapshot'],
           'episodes_completed':data['episodes_completed'],'updates':data['updates'],
           'pending_empty':True,'preserved_state_sha256':{
              p:{k:digest(data[p][k]) for k in ('params','m','v','rng','steps')}
              for p in ('policy','initial_policy')},
           'changed_fields':['policy_config.projection_backend','policy.config.projection_backend',
                             'initial_policy.config.projection_backend'],
           'warning':'Future floating-point accumulation differs. Resume exactly within the selected backend; do not treat cross-backend runs as bit-identical.'}
    with destination.open('xb') as f:f.write(payload)
    with audit_path.open('x',encoding='utf-8') as f:f.write(canonical(audit)+'\n')
    return audit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--allow-real-map',action='store_true')
    a=p.parse_args(argv)
    try: result=migrate(a.source,a.destination,a.snapshot,allow_real=a.allow_real_map)
    except (ValueError,KeyError,TypeError,OSError) as exc:p.error(str(exc))
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
