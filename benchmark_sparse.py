import argparse,json,pickle,resource,time
from pathlib import Path
import numpy as np
from env.neural_train import NeuralTrainer
from env.neural_policy import NeuralPolicy
OUT=Path('perf-results');OUT.mkdir(exist_ok=True)
config=dict(seed=11,buckets=4096,hidden_dim=64,representation='public_v2_split',update_mode='ppo_clip',ppo_clip=.2,update_epochs=4,baseline_offset='prior_batch_mean')
p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','numpy','scipy_csr']);args=p.parse_args()
if args.mode=='collect':
    t=NeuralTrainer('scenario/7527',allow_real=True,seed=11,observation_schema=2,
        episode_config={'max_seconds':1800,'max_steps':1000,'move_radius':3},policy_config={k:v for k,v in config.items() if k not in ('seed','representation')})
    start=time.perf_counter();record,trajectory=t._rollout(t.policy,20000,'BLUE',training=True)
    with (OUT/'frozen-rollout.pkl').open('wb') as f:pickle.dump((trajectory,record['own_score']),f)
    (OUT/'collection.json').write_text(json.dumps({'record':record,'seconds':time.perf_counter()-start,'scoring_version':8,'source_commit':'59009d5','config':config},indent=2))
    print('collected',len(trajectory),'decisions',flush=True)
else:
    with (OUT/'frozen-rollout.pkl').open('rb') as f:trajectory,score=pickle.load(f)
    policy=NeuralPolicy(**config,projection_backend=args.mode)
    examples=[(d,score) for d in trajectory]
    times=[]
    for _ in range(3):
        start=time.perf_counter()
        forwards=[policy._forward(d) for d in trajectory]
        times.append(time.perf_counter()-start)
    predictions=np.array([[*f[3],f[4]] for f in forwards],dtype=object)
    np.save(OUT/f'{args.mode}-predictions.npy',predictions,allow_pickle=True)
    del forwards,predictions
    start=time.perf_counter();loss,gradients=policy._loss_gradients(examples,actor_normalizer=1);gradient_seconds=time.perf_counter()-start
    np.savez(OUT/f'{args.mode}-gradients.npz',**gradients);del gradients
    start=time.perf_counter();update=policy.update_batch([(trajectory,score)]);optimizer_seconds=time.perf_counter()-start
    np.savez(OUT/f'{args.mode}-updated.npz',**{g+'_'+k:v for g in ('params','m','v') for k,v in getattr(policy,g).items()})
    result={'backend':args.mode,'decisions':len(trajectory),'total_candidates':sum(len(d.candidates) for d in trajectory),'forward_seconds':times,'gradient_seconds':gradient_seconds,'optimizer_seconds':optimizer_seconds,'loss':loss,'update':update,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'rng':policy.rng.bit_generator.state}
    (OUT/f'{args.mode}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
