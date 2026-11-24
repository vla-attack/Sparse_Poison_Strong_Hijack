"""Optimize only the cup, with exact final-iteration export and no phrase gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'openvla-oft'
DEFAULT_BASE = REPO / ('runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+'
    'lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt')
DEFAULT_SOURCE = ROOT / 'paper_aligned_full_1gpu_20261120/stage1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_command(args):
    """Check input identities without loading a model or changing source data."""
    source, checkpoint, output = args.source_stage.resolve(), args.base.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError(f'Use a new output directory: {output}')
    original = source / 'original_native_collection'
    instruction_map = source / 'instruction_map.json'
    required = [original / 'summary.json', original / 'candidate/candidate.json', instruction_map,
                checkpoint / 'model.safetensors.index.json', checkpoint / 'dataset_statistics.json',
                checkpoint / 'action_head--150000_checkpoint.pt',
                checkpoint / 'proprio_projector--150000_checkpoint.pt']
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    collection = read(original / 'summary.json')
    candidate = read(original / 'candidate/candidate.json')
    policy = read(instruction_map)
    if (collection['status'] != 'completed' or not collection['physical_rollouts_regenerated']
            or collection['task5_manual_correction'] or collection['accepted_episodes'] != 5
            or candidate['relative_scale'] != 1. or any(candidate['position_delta_xy_m'])):
        raise ValueError('Require five verified original, unoptimized physical cup demonstrations')
    if candidate['proxy_checkpoint'] != str(checkpoint):
        raise ValueError('Original reference belongs to a different frozen checkpoint')
    if candidate['language_policy'] != policy:
        raise ValueError('Source candidate and instruction map differ')
    for key in ('texture_file', 'compiled_target_file'):
        path = original / 'candidate' / candidate[key]
        if not path.is_file(): raise FileNotFoundError(path)
    if sha(original / 'candidate' / candidate['texture_file']) != candidate['texture_sha256']:
        raise ValueError('Original source texture changed')
    for row in collection['accepted_detail']:
        folder = Path(row['artifact_dir'])
        for name in ('trajectory.npz', 'trajectory.json'):
            if not (folder / name).is_file(): raise FileNotFoundError(folder / name)
    for name in set(read(checkpoint / 'model.safetensors.index.json')['weight_map'].values()):
        if not (checkpoint / name).is_file(): raise FileNotFoundError(checkpoint / name)
    if (not 1 <= args.iterations <= args.schedule_iterations or any(
            not math.isfinite(weight) or weight < 0 for weight in
            (args.action_weight, args.visual_weight, args.text_weight))):
        raise ValueError('Invalid iteration budget or loss weights')
    if args.action_weight + args.visual_weight + args.text_weight == 0:
        raise ValueError('At least one loss weight must be positive')
    torchrun = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', f'--nproc-per-node={args.gpus}', '--max-restarts=0']
    command = [*torchrun, '-m', 'experiments.vla_safety.optimize_selective_task_cup',
        '--proxy-checkpoint', str(checkpoint), '--native-source', str(original),
        '--instruction-map', str(instruction_map), '--output-dir', str(output / 'optimization'),
        '--candidate-dir', str(output / 'candidate'), '--iterations', str(args.iterations),
        '--schedule-iterations', str(args.schedule_iterations),
        '--action-weight', str(args.action_weight), '--visual-weight', str(args.visual_weight),
        '--text-weight', str(args.text_weight), '--keep-weight', '0',
        '--export-final', '--optimize-position', '--verify']
    if args.smoke: command.append('--smoke')
    return source, output, torchrun, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-stage', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--gpus', type=int, choices=(1, 4), default=4)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--schedule-iterations', type=int, default=300)
    parser.add_argument('--action-weight', type=float, default=1.)
    parser.add_argument('--visual-weight', type=float, default=1.)
    parser.add_argument('--text-weight', type=float, default=.1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, output, torchrun, command = prepare_command(args)
    if args.dry_run:
        print('INPUT_PREFLIGHT_PASSED', flush=True)
        print('DRY_RUN_COMMAND', json.dumps(command), flush=True)
        print('GPU execution is not checked by --dry-run.', flush=True)
        return
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(name, None)
    import torch
    from experiments.vla_safety.training.cuda_device_mapping import visible_device_tokens
    count = torch.cuda.device_count()
    if count < args.gpus:
        raise RuntimeError(f'This command requires {args.gpus} allocated visible GPUs; found {count}. '
                           'Allocate four GPUs before submitting the four-GPU command.')
    tokens = visible_device_tokens(count)[:args.gpus]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(tokens))
    env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    files = ['experiments/vla_safety/' + name for name in (
        'run_final_cup_optimization.py', 'optimize_selective_task_cup.py', 'selective_cup_runtime.py',
        'training/cup_pcgrad.py', 'training/selective_text_attention.py', 'training/cuda_device_mapping.py')]
    for name in files: shutil.copy2(REPO / name, snapshot / name.replace('/', '__'))
    manifest = dict(export_mode='final', export_blend_factor=1., phrase_keep_weight=0.,
        protected_phrase_constraints_enforced=False, iterations=args.iterations,
        schedule_iterations=args.schedule_iterations, GPUs=args.gpus, device_tokens=tokens,
        loss_weights=dict(action=args.action_weight, visual=args.visual_weight, text=args.text_weight),
        original_source=str(source / 'original_native_collection'),
        source_candidate_sha256=sha(source / 'original_native_collection/candidate/candidate.json'),
        source_code_sha256={name: sha(REPO / name) for name in files},
        command=command, output=str(output), stage_ii_training=False)
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(name, arguments):
        print('STAGE', name, 'LOG', str(output / 'logs' / (name + '.log')), flush=True)
        (output / 'stage.txt').write_text(name + '\n')
        with (output / 'logs' / (name + '.log')).open('w') as log:
            subprocess.run(arguments, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        run('gpu_startup', [*torchrun, '-m', 'experiments.vla_safety.training.check_cuda_nccl_startup',
            '--expected-world-size', str(args.gpus), '--isolate-local-device',
            '--output', str(output / 'gpu_startup')])
        run('cup_optimization', command)
        result = read(output / 'optimization/summary.json')
        if (result['status'] != 'completed' or result['export_mode'] != 'final'
                or result['export_iteration'] != args.iterations or result['export_blend_factor'] != 1.
                or result['protected_phrase_constraints_enforced']):
            raise ValueError('Export differs from the requested full final iteration')
        (output / 'stage.txt').write_text('completed\n')
        print('CUP_FINAL_READY', str(output / 'candidate/candidate.json'), flush=True)
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(exc),
            stage=(output / 'stage.txt').read_text().strip()), indent=2) + '\n')
        raise


if __name__ == '__main__': main()
