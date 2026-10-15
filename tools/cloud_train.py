#!/usr/bin/env python3
"""Bounded, offline NumPy training with resumable checkpoints and live JSON progress.

The episode budget counts attempts; truncated games never count as completions.
1800 seconds is simulated game duration, not a wall-clock deadline. SIGTERM and
SIGINT request a stop at the next safe episode boundary (never mid-update).
"""
import os
# Set before importing NumPy, including when a host has supplied larger defaults.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.neural_train import NeuralTrainer, canonical, summary
from env.neural_policy import NeuralPolicy
from engine import scoring
from tools.evaluate_neural_checkpoint import paired_deltas
from tools.export_policy_replay import export_replay


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_manifest(root=ROOT):
    paths = sorted([*root.glob('engine/**/*.py'), *root.glob('engine/**/*.json'), *root.glob('env/**/*.py')])
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {'sha256': hashlib.sha256(canonical(files).encode()).hexdigest(), 'files': files}


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


class Resources:
    def __init__(self):
        self.wall, self.cpu = time.monotonic(), time.process_time()

    def sample(self):
        now, cpu = time.monotonic(), time.process_time()
        result = {'rss_mb': None, 'cpu_percent': 100 * (cpu - self.cpu) / max(now - self.wall, 1e-9),
                  'available_memory_mb': None}
        self.wall, self.cpu = now, cpu
        try:
            pages = int(Path('/proc/self/statm').read_text().split()[1])
            result['rss_mb'] = pages * os.sysconf('SC_PAGE_SIZE') / 1048576
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(mem['MemAvailable'].split()[0]) * 1024
            result['available_memory_mb'] = available / 1048576
            limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
            if limit != 'max':
                available = min(available, max(0, int(limit) - int(Path('/sys/fs/cgroup/memory.current').read_text())))
            result['available_memory_mb'] = available / 1048576
        except (OSError, ValueError, KeyError):
            pass
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=ROOT / 'scenario/7527')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/cloud-training')
    p.add_argument('--run-id', default=None)
    p.add_argument('--stop-file', type=Path, default=None,
                   help='stop at a safe episode boundary when this file exists; default RUN_DIR/STOP_REQUESTED; remove before resume')
    p.add_argument('--resume', action='store_true', help='resume the explicitly named run')
    p.add_argument('--initialize-from', type=Path, help='import a completed batch into a new run; never combines with resume')
    p.add_argument('--parent-training-report', type=Path, help='complete historical training journal matching the imported checkpoint')
    p.add_argument('--episodes', type=int, default=160, help='total attempt budget, including resumed attempts')
    p.add_argument('--checkpoint-every', type=int, default=8)
    p.add_argument('--seed', type=int, default=7527)
    p.add_argument('--train-seed-start', type=int, default=0)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[1000000, 1000001, 1000002, 1000003])
    p.add_argument('--batch-episodes', type=int, default=8)
    p.add_argument('--update-mode', choices=['reinforce', 'ppo_clip'], default='ppo_clip')
    p.add_argument('--observation-schema', type=int, choices=[2, 3], default=2)
    p.add_argument('--projection-backend', choices=['numpy', 'scipy_csr'], default='numpy')
    p.add_argument('--scoring-mode', choices=['verified', 'platform'], default='verified')
    return p


def run(args, trainer_class=NeuralTrainer, replay_exporter=export_replay):
    if not 1 <= args.episodes <= 10000 or not 1 <= args.checkpoint_every <= 10000:
        raise ValueError('episode budget and checkpoint interval must be 1..10000')
    if bool(args.initialize_from) != bool(args.parent_training_report):
        raise ValueError('initialize-from requires parent-training-report and vice versa')
    if args.resume and args.initialize_from:
        raise ValueError('initialize-from cannot be combined with resume')
    if args.resume and not args.run_id:
        raise ValueError('resume requires --run-id')
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
        raise ValueError('run-id must contain only letters, numbers, hyphens and underscores')
    if any(args.train_seed_start <= seed < args.train_seed_start + args.episodes for seed in args.eval_seeds):
        raise ValueError('training budget overlaps reserved evaluation seeds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Only one writer per progress feed. Kernel releases this lock after a crash.
    import fcntl
    lock = (output / '.runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('another runner owns this output directory')
    try:
        return _run_locked(args, run_id, output, trainer_class, replay_exporter)
    finally:
        lock.close()


def requested_config(args):
    config = {'seed': args.seed, 'train_seed_start': args.train_seed_start,
              'batch_episodes': args.batch_episodes, 'heldout_seeds': args.eval_seeds,
              'observation_schema': args.observation_schema, 'allow_real': True, 'scoring_mode': args.scoring_mode,
              'episode_config': {'max_seconds': 1800, 'max_steps': 1000, 'move_radius': 3},
              'policy_config': {'buckets': 4096, 'hidden_dim': 64, 'learning_rate': .001,
                                'baseline_offset': 'prior_batch_mean', 'update_mode': args.update_mode}}
    if args.projection_backend != 'numpy':
        config['policy_config']['projection_backend'] = args.projection_backend
    return config


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_completed_checkpoint(args, config, trainer_class=NeuralTrainer):
    """Validate all inputs before creating a target run; imports never train."""
    checkpoint, report_path = args.initialize_from, args.parent_training_report
    checkpoint_sha, report_sha = file_sha256(checkpoint), file_sha256(report_path)
    trainer = trainer_class.load(checkpoint, args.snapshot, allow_real=True)
    if scoring.SCORING_VERSION != 8:
        raise ValueError('this import contract requires simulator version 8')
    if trainer.pending or trainer.episodes_completed == 0:
        raise ValueError('import requires completed episodes and empty pending batch')
    if trainer.episodes_attempted > args.episodes:
        raise ValueError('requested cumulative budget is smaller than imported attempts')
    for key in ('seed', 'train_seed_start', 'batch_episodes', 'heldout_seeds',
                'observation_schema', 'scoring_mode'):
        if getattr(trainer, key) != config[key]:
            raise ValueError('imported ' + key + ' differs from requested configuration')
    episode_config = {'max_seconds': 1800, 'max_steps': 1000, 'public_score': True,
                      **config['episode_config'], 'scoring_mode': config['scoring_mode']}
    representation = {2: 'public_v2_split', 3: 'public_v3_spatial'}[config['observation_schema']]
    policy_config = NeuralPolicy(seed=config['seed'], representation=representation,
                                 **config['policy_config']).config
    if trainer.episode_config != episode_config or trainer.policy.config != policy_config:
        raise ValueError('imported episode/policy configuration differs; explicitly migrate backend first')
    saved = json.loads(Path(report_path).read_text())
    required = {'training', 'training_summary', 'training_elapsed_seconds', 'episodes_attempted',
                'episodes_completed', 'optimizer_updates'}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError('parent training journal is missing required fields')
    for key in ('episodes_attempted', 'episodes_completed', 'optimizer_updates'):
        if isinstance(saved[key], bool) or not isinstance(saved[key], int):
            raise ValueError('parent training journal counters must be integers')
    records = saved['training']
    if (not isinstance(records, list) or len(records) != trainer.episodes_attempted or
            saved['episodes_attempted'] != trainer.episodes_attempted or
            saved['episodes_completed'] != trainer.episodes_completed or
            saved['optimizer_updates'] != trainer.policy.steps):
        raise ValueError('parent training journal counters do not match checkpoint')
    for index, row in enumerate(records):
        if not isinstance(row, dict) or not {'seed', 'learner_side', 'own_score', 'score_diff', 'result', 'status'} <= row.keys():
            raise ValueError('parent training journal row is missing required fields')
        status = row['status']
        if not isinstance(status, dict) or status.get('scoring_version') != 8 or status.get('scoring_mode') != config['scoring_mode']:
            raise ValueError('parent training journal scoring contract differs')
        if row['seed'] != trainer.train_seed_start + index or row['learner_side'] != ('BLUE', 'RED')[index % 2]:
            raise ValueError('parent training journal seed/seat order differs')
        value = row['own_score']
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('parent training journal has invalid score')
        if value is not None:
            scores = status.get('final_scores')
            if not status.get('terminated') or status.get('truncated') or not isinstance(scores, dict) or scores.get(row['learner_side']) != value:
                raise ValueError('parent training journal terminal score differs')
        elif not status.get('truncated'):
            raise ValueError('parent training journal unfinished result is not truncated')
    if sum(row['own_score'] is not None for row in records) != trainer.episodes_completed:
        raise ValueError('parent training journal completion count differs')
    if saved['training_summary'] != summary(records):
        raise ValueError('parent training journal summary differs')
    elapsed = saved['training_elapsed_seconds']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('parent training elapsed time is invalid')
    audit_path = Path(checkpoint).with_suffix(Path(checkpoint).suffix + '.migration.json')
    migration, migration_sha = None, None
    if audit_path.exists():
        migration_sha = file_sha256(audit_path)
        migration = json.loads(audit_path.read_text())
        if migration.get('output_checkpoint_sha256') != checkpoint_sha:
            raise ValueError('backend migration audit does not match imported checkpoint')
    if migration_sha is not None and migration_sha != file_sha256(audit_path):
        raise ValueError('backend migration audit changed during import')
    if checkpoint_sha != file_sha256(checkpoint) or report_sha != file_sha256(report_path):
        raise ValueError('parent checkpoint/report changed during import')
    audit = {'schema_version': 1, 'parent_checkpoint_sha256': checkpoint_sha,
             'parent_training_report_sha256': report_sha,
             'parent_checkpoint': str(Path(checkpoint).resolve()),
             'parent_training_report': str(Path(report_path).resolve()),
             'imported_episodes_attempted': trainer.episodes_attempted,
             'imported_episodes_completed': trainer.episodes_completed,
             'imported_optimizer_steps': trainer.policy.steps,
             'scoring_version': scoring.SCORING_VERSION, 'snapshot': trainer.manifest,
             'backend_migration': migration, 'backend_migration_audit_sha256': migration_sha,
             'pending_empty': True,
             'historical_records_preserved': True,
             'parameters_optimizer_rng_preserved': True}
    return trainer, records, elapsed, audit


def _run_locked(args, run_id, output, trainer_class, replay_exporter):
    directory = output / run_id
    if args.resume:
        if not directory.is_dir():
            raise ValueError('resume run does not exist')
    elif directory.exists():
        raise FileExistsError('new run directory already exists')
    config = requested_config(args)
    source = source_manifest()
    manifest_path = directory / 'manifest.json'
    checkpoint_path = directory / 'checkpoint.json'
    training_path = directory / 'training.json'
    evaluation_path = directory / 'evaluation.json'
    replay_path = directory / 'replay.json'
    records, previous_elapsed = [], 0
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['source'] != source or manifest['config'] != config:
            raise ValueError('incompatible source or training configuration; start a new run')
        trainer = trainer_class.load(checkpoint_path, args.snapshot, allow_real=True)
        saved = json.loads(training_path.read_text())
        # Checkpoint is authoritative if a crash occurred between the atomic files.
        records = saved['training'][:trainer.episodes_attempted]
        if len(records) != trainer.episodes_attempted:
            raise ValueError('training journal is missing checkpointed episodes')
        previous_elapsed = saved['training_elapsed_seconds']
        if trainer.episodes_attempted > args.episodes:
            raise ValueError('requested budget is smaller than existing attempts')
    else:
        import_audit = None
        if args.initialize_from:
            trainer, records, previous_elapsed, import_audit = import_completed_checkpoint(args, config, trainer_class)
        else:
            trainer = trainer_class(args.snapshot, **config)
        directory.mkdir(exist_ok=False)
        manifest = {'schema_version': 1, 'run_id': run_id, 'commit': git_commit(), 'source': source,
                    'snapshot': trainer.manifest, 'config': config, 'offline_only': True,
                    'backend': args.projection_backend + '_cpu', 'threads': 1}
        if import_audit is not None:
            manifest['initialization'] = import_audit
            atomic_json(directory / 'import-audit.json', import_audit)
        atomic_json(manifest_path, manifest)
    started = time.monotonic()
    resources = Resources()
    state = {'status': 'initializing', 'phase': 'training', 'run_id': run_id,
             'commit': manifest['commit'], 'episodes_target': args.episodes, 'issues': [],
             'latest_report': None, 'latest_evaluation': None, 'latest_replay': None,
             'budget_unit': 'episode_attempts', 'checkpoint': None}
    training_elapsed = previous_elapsed
    stop = [False]
    stop_file = args.stop_file.resolve() if args.stop_file else directory / 'STOP_REQUESTED'
    state['stop_file'] = str(stop_file)
    def should_stop():
        if stop_file.exists():
            state['stop_reason'] = 'stop_file'
            return True
        return stop[0]
    def request_stop(signum, frame):
        stop[0] = True
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def relative(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    def progress(status, phase=None):
        state.update(status=status, phase=phase or state['phase'], updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                     episodes_completed=trainer.episodes_completed, episodes_attempted=trainer.episodes_attempted,
                     optimizer_updates=trainer.policy.steps, elapsed_seconds=previous_elapsed + time.monotonic() - started,
                     training_elapsed_seconds=training_elapsed,
                     episodes_per_minute=trainer.episodes_completed * 60 / training_elapsed if training_elapsed else 0,
                     resources=resources.sample(), latest_training_summary=summary(records))
        atomic_json(output / 'progress.json', state)
        atomic_json(directory / 'progress.json', state)
    def save():
        # Journal first: if the process dies before checkpoint replacement, resume truncates it.
        atomic_json(training_path, {'training': records, 'training_summary': summary(records),
                                    'training_elapsed_seconds': training_elapsed,
                                    'episodes_attempted': trainer.episodes_attempted,
                                    'episodes_completed': trainer.episodes_completed,
                                    'optimizer_updates': trainer.policy.steps})
        atomic_json(checkpoint_path, trainer.checkpoint())
        state['checkpoint'], state['latest_report'] = relative(checkpoint_path), relative(training_path)
    try:
        save()
        progress('training')
        while trainer.episodes_attempted < args.episodes and not should_stop():
            tick = time.monotonic()
            records.extend(trainer.train(1))
            training_elapsed += time.monotonic() - tick
            if trainer.episodes_attempted % args.checkpoint_every == 0:
                save()
            progress('training')
        save()
        if should_stop():
            progress('interrupted')
            return 130
        progress('evaluating', 'evaluation')
        before = canonical(trainer.checkpoint())
        tick = time.monotonic()
        evaluation = trainer.evaluate()
        if canonical(trainer.checkpoint()) != before:
            raise RuntimeError('evaluation changed checkpoint state')
        atomic_json(evaluation_path, {'evaluation': evaluation,
                    'paired_comparison': paired_deltas(evaluation, trainer.heldout_seeds),
                    'elapsed_seconds': time.monotonic() - tick,
                    'models_unchanged': True, 'source': source, 'snapshot': trainer.manifest})
        state['latest_evaluation'] = relative(evaluation_path)
        if should_stop():
            progress('interrupted')
            return 130
        progress('exporting', 'replay')
        replay = replay_exporter(checkpoint_path, args.snapshot, seed=trainer.heldout_seeds[0],
                                 learner_side='BLUE', allow_real=True)
        atomic_json(replay_path, replay)
        state['latest_replay'] = relative(replay_path)
        progress('completed', 'complete')
        return 0
    except Exception as exc:
        state['issues'].append(type(exc).__name__ + ': ' + str(exc))
        # Do not serialize a potentially half-updated trainer after an unexpected failure.
        progress('failed')
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)



def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
