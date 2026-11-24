"""Complete stage 1: original native demos -> constrained PCGrad cup -> new native RLDS."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from experiments.vla_safety.collect_native_cup_demonstrations import read, sha, write
from experiments.vla_safety.run_grouped_fusion_firstcup_single import environment
from experiments.vla_safety.training.selective_text_attention import VERSION
from experiments.vla_safety.task_matched_language import load_policy, policy_digest


def run(args):
    root = Path(os.environ.get('OFT_ROOT', '/project_c20250301/241207013/project/huangsimin'))
    repo = root / 'openvla-oft'
    base = Path(os.environ.get('OFT_CLEAN150K_CHECKPOINT', str(repo / 'runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or repo / 'experiments/vla_safety/artifacts' / ('selective_clean150k_cup_stage1_' + stamp)).resolve()
    if output.exists() and not args.reuse_output:
        raise FileExistsError('Use a fresh output or --reuse-output to validate and continue an existing stage')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'failure.json').exists():
        history = output / 'previous_failures'; history.mkdir(exist_ok=True)
        (output / 'failure.json').rename(history / (stamp + '.json'))
    (root / 'latest_selective_cup_stage1_pipeline.txt').write_text(str(output) + '\n')
    data = (args.dataset_root or root / ('modified_libero_rlds_clean150k_selective_text_native5_' + output.name.removeprefix('selective_clean150k_cup_stage1_'))).resolve()
    (output / 'dataset_root.txt').write_text(str(data) + '\n')
    env = environment(repo)
    env['PYTHONPATH'] = str(root / 'envs/torch/local-src/nvdiffrast-main') + ':' + env['PYTHONPATH']
    import torch
    from experiments.vla_safety.training.cuda_device_mapping import visible_device_tokens
    count = torch.cuda.device_count()
    if count < args.nproc: raise ValueError(f'Need {args.nproc} allocated GPUs; found {count}')
    tokens = visible_device_tokens(count)[:args.nproc]
    env['CUDA_VISIBLE_DEVICES'] = ','.join(tokens)
    python = sys.executable
    stage = 'preflight'
    def mark(value):
        nonlocal stage
        stage = value; (output / 'stage.txt').write_text(value + '\n')
        print('STAGE', datetime.now(timezone.utc).isoformat(), value, flush=True)
    def execute(command, name, *, cpu=False, single=False):
        child = dict(env)
        if cpu: child['CUDA_VISIBLE_DEVICES'] = ''
        elif single: child['CUDA_VISIBLE_DEVICES'] = tokens[0]
        print('EXEC', json.dumps(list(map(str, command))), flush=True)
        with (output / name).open('a') as log:
            process = subprocess.Popen(list(map(str, command)), cwd=repo, env=child,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = process.wait()
                if code: raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None: process.terminate(); process.wait()
                raise
    try:
        mark('clean_language_audit')
        execute([python, '-u', '-m', 'experiments.vla_safety.task_matched_language',
            '--clean-root', root / 'modified_libero_rlds', '--output', output / 'instruction_map.json'],
            'instruction_map.log', cpu=True)
        policy = load_policy(output / 'instruction_map.json')
        config = dict(version=VERSION, checkpoint=str(base.resolve()), instruction_mode='matched_clean_task',
            language_policy_sha256=policy_digest(policy), iterations=args.iterations, nproc=args.nproc,
            cuda_device_tokens=tokens, keep_weight=args.keep_weight, keep_tolerance=args.keep_tolerance,
            optimize_position=args.optimize_position,
            dataset_root=str(data), tasks=[0, 2, 4, 5, 7], state_id=0, seed=7,
            training_stage='Cup texture/size and optional XY search; frozen policy, no dual-LoRA training',
            original_native_collection=str(output / 'original_native_collection'),
            optimized_native_collection=str(output / 'optimized_native_collection'))
        if (output / 'pipeline_config.json').exists() and read(output / 'pipeline_config.json') != config:
            raise ValueError('Existing stage has a different optimization/data protocol')
        write(output / 'pipeline_config.json', config)
        snapshot = output / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
        names = ['experiments/vla_safety/' + name for name in (
            'run_selective_cup_stage1.py', 'prepare_original_cup_reference.py', 'optimize_selective_task_cup.py',
            'selective_cup_runtime.py', 'training/selective_text_attention.py', 'training/cup_pcgrad.py',
            'optimize_task_matched_cup.py', 'optimize_p1_fixed_xy_multimodal.py',
            'collect_native_cup_demonstrations.py', 'build_native_cup_rlds.py', 'verify_task_matched_dataset.py',
            'task_matched_language.py')]
        for name in names: shutil.copy2(repo / name, snapshot / name.replace('/', '__'))
        write(snapshot / 'sha256.json', {name: sha(repo / name) for name in names})
        original = output / 'original_candidate/candidate.json'
        mark('original_unoptimized_cup_reference')
        if not original.exists():
            execute([python, '-u', '-m', 'experiments.vla_safety.prepare_original_cup_reference',
                '--output', original.parent, '--instruction-map', output / 'instruction_map.json',
                '--checkpoint', base], 'prepare_original_reference.log', single=True)
        reference = read(original)
        if reference['relative_scale'] != 1. or reference['language_policy_sha256'] != policy_digest(policy):
            raise ValueError('Original reference cup or language changed')
        mark('original_native_collection')
        source = output / 'original_native_collection'
        if not (source / 'summary.json').exists():
            command = [python, '-u', '-m', 'experiments.vla_safety.collect_native_cup_demonstrations',
                '--source-candidate', original, '--instruction-map', output / 'instruction_map.json', '--output', source]
            if source.exists(): command.append('--verify-existing')
            execute(command, 'original_native_collection.log', single=True)
        from experiments.vla_safety.build_native_cup_rlds import native_collection
        native_collection(source)
        mark('selective_pcgrad_cup_optimization')
        optimization = output / 'optimization'
        if not (optimization / 'summary.json').exists():
            torchrun = ([python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                         f'--nproc-per-node={args.nproc}'] if args.nproc > 1 else [python, '-u'])
            command = [*torchrun, '-m', 'experiments.vla_safety.optimize_selective_task_cup',
                '--proxy-checkpoint', base, '--native-source', source,
                '--instruction-map', output / 'instruction_map.json', '--output-dir', optimization,
                '--candidate-dir', output / 'candidate', '--iterations', args.iterations,
                '--schedule-iterations', args.iterations, '--keep-weight', args.keep_weight,
                '--keep-tolerance', args.keep_tolerance]
            if args.optimize_position: command.append('--optimize-position')
            verified = output / 'verification/verification.json'
            has_check = verified.exists() and read(verified).get('status') == 'passed'
            if has_check:
                has_check = all(sha(repo / name) == digest for name, digest in read(verified).get('source_sha256', {}).items())
                has_check = has_check and bool(read(verified).get('source_sha256'))
            if not has_check: command.append('--verify')
            if (optimization / 'optimization_state.pt').exists(): command.append('--resume')
            execute(command, 'optimization.log')
        result = read(optimization / 'summary.json')
        candidate = output / 'candidate/candidate.json'
        if (result['version'] != VERSION or not result['model_unchanged']
                or result.get('best_iteration', 0) <= 0
                or result.get('attack_score_improvement', 0.) <= 1e-6
                or not result.get('attack_components_improved', False)
                or not result.get('texture_changed_from_original', False)
                or not result['protected_phrases_all_source_and_native_passed']
                or result['candidate_sha256'] != sha(candidate)):
            raise ValueError('Stage I produced no non-initial candidate improving all attack terms under the phrase/native gates')
        mark('optimized_native_physical_collection')
        collected = output / 'optimized_native_collection'
        if not (collected / 'summary.json').exists():
            command = [python, '-u', '-m', 'experiments.vla_safety.collect_native_cup_demonstrations',
                '--source-candidate', candidate, '--instruction-map', output / 'instruction_map.json', '--output', collected]
            if collected.exists(): command.append('--verify-existing')
            execute(command, 'optimized_native_collection.log', single=True)
        native_collection(collected)
        mark('native_rlds_export')
        if not (data / 'DATASET_MANIFEST.json').exists():
            execute([python, '-u', '-m', 'experiments.vla_safety.build_native_cup_rlds',
                '--source', collected, '--output', data, '--normalization-statistics', base / 'dataset_statistics.json',
                '--audit', output / 'rlds_audit'], 'rlds_build.log', cpu=True)
        mark('independent_native_dataset_validation')
        execute([python, '-u', '-m', 'experiments.vla_safety.verify_task_matched_dataset',
            '--root', data, '--instruction-map', output / 'instruction_map.json',
            '--output', output / 'dataset_validation.json'], 'dataset_validation.log', cpu=True)
        dataset = read(data / 'DATASET_MANIFEST.json')
        summary = dict(status='completed', version=VERSION, optimization=result,
            physical_rollouts_regenerated=True, dataset_root=str(data), episodes=dataset['episodes'],
            frames=dataset['frames'], task_frame_counts=dataset['task_frame_counts'],
            dataset_validation=str(output / 'dataset_validation.json'))
        write(output / 'summary.json', summary)
        (root / 'latest_selective_cup_stage1_dataset.txt').write_text(str(data) + '\n')
        mark('completed'); print('SELECTIVE_STAGE1_COMPLETED', json.dumps(summary), flush=True)
    except BaseException as error:
        write(output / 'failure.json', dict(stage=stage, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--reuse-output', action='store_true')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--nproc', type=int, choices=[1, 4], default=1)
    parser.add_argument('--keep-weight', type=float, default=20.)
    parser.add_argument('--keep-tolerance', type=float, default=.05)
    parser.add_argument('--optimize-position', action='store_true')
    run(parser.parse_args())
