"""One four-GPU job: cup optimization, native RLDS, dual20k, merge, 500+500."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.vla_safety.collect_native_cup_demonstrations import read, sha, write
from experiments.vla_safety.run_grouped_fusion_firstcup_single import environment
from experiments.vla_safety.training.cuda_device_mapping import visible_device_tokens
from experiments.vla_safety.training.selective_text_attention import VERSION
from experiments.vla_safety.training.selective_trigger_objective import TRAINING_OBJECTIVE


def configure_dataset_objective(data, stage_config):
    manifest = read(data / 'DATASET_MANIFEST.json')
    if not manifest.get('physical_rollouts_regenerated') or manifest['status'] != 'passed':
        raise ValueError('Training requires the newly collected native MuJoCo dataset')
    manifest['objective'] = dict(**TRAINING_OBJECTIVE, version=VERSION,
        weights=stage_config['weights'], keep_weight=stage_config['keep_weight'],
        reference_floor=stage_config['text_reference_floor'])
    write(data / 'DATASET_MANIFEST.json', manifest)


def training_command(python, base, clean, data, instruction_map, sidecar, run_id,
                     live, steps=20000, nproc=4, objective=None, resume=None):
    objective = objective or dict(weights=dict(action=1., visual=1., text=1.), keep_weight=20., text_reference_floor=1e-6)
    command = [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={nproc}',
        'vla-scripts/finetune_dual_lora.py', '--vla_path', resume[0] if resume else base,
        '--auxiliary_checkpoint_dir', base, '--continue_clean_lora_checkpoint', base,
        '--data_root_dir', clean, '--clean_data_root_dir', clean, '--trigger_data_root_dir', data,
        '--normalization_statistics_path', data / 'normalization_statistics.json',
        '--instruction_map_path', instruction_map, '--center_crop', 'True', '--clean_uses_trigger', 'True',
        '--strict_source_batches', 'True', '--trigger_fraction', .5, '--batch_size', 2, '--grad_accumulation_steps', 1,
        '--max_steps', steps, '--resume', 'True' if resume else 'False',
        '--learning_rate', 5e-5, '--lr_schedule', 'cosine', '--lr_warmup_steps', 200 if steps > 200 else 0,
        '--minimum_learning_rate', 5e-6, '--use_l1_regression', 'True', '--use_diffusion', 'False',
        '--use_film', 'False', '--use_val_set', 'False', '--num_images_in_input', 2, '--use_proprio', 'True',
        '--lora_rank', 32, '--lora_alpha', 16, '--lora_dropout', 0, '--image_aug', 'False',
        '--shuffle_buffer_size', 100000 if steps > 200 else 32,
        '--use_multimodal_trigger_loss', 'True', '--trigger_objective_mode', 'selective_native',
        '--trigger_sidecar_path', sidecar, '--trigger_action_weight', objective['weights']['action'],
        '--trigger_visual_weight', objective['weights']['visual'], '--trigger_text_weight', objective['weights']['text'],
        '--trigger_keep_weight', objective['keep_weight'], '--trigger_reference_floor', objective['text_reference_floor'],
        '--live_loss_dir', live, '--save_freq', 1000 if steps > 200 else steps,
        '--gradient_audit_dir', live.parent / 'gradient_audit',
        '--save_final_checkpoint_only', 'False', '--save_latest_checkpoint_only', 'False',
        '--merge_lora_during_training', 'False', '--run_root_dir', base.parents[1] / 'runs',
        '--run_id_override', run_id, '--wandb_entity', 'openvla-oft-local',
        '--wandb_project', 'clean150k-selective-native-dual20k', '--wandb_log_freq', 10 if steps > 200 else 1]
    if resume: command += ['--resume_step', resume[1]]
    return list(map(str, command))


def clone_cup_progress(source, target):
    """Copy mutable optimizer state; reference the validated immutable source demos."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ('original_candidate', 'original_native_collection'):
        path = target / name
        if not path.exists(): path.symlink_to((source / name).resolve(), target_is_directory=True)
    if not (target / 'instruction_map.json').exists(): shutil.copy2(source / 'instruction_map.json', target / 'instruction_map.json')
    if not (target / 'optimization').exists():
        shutil.copytree(source / 'optimization', target / 'optimization',
            ignore=shutil.ignore_patterns('native_replay_audits', 'workers', 'native_audit_*', '*.tmp'))
    write(target / 'copied_cup_progress.json', dict(source=str(source.resolve()),
        optimizer_checkpoint_sha256=sha(source / 'optimization/optimization_state.pt'),
        note='Only cup optimization resumes; dual LoRA starts from original Clean150k'))


def preflight(root, need_four=False):
    import torch
    from experiments.vla_safety.training.clean_lora_continuation import read_original_adapter
    repo = root / 'openvla-oft'
    base = Path(os.environ.get('OFT_CLEAN150K_CHECKPOINT', str(repo / 'runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt')))
    values, identity = read_original_adapter(base); del values
    stats = read(base / 'dataset_statistics.json')
    if (set(stats) != {'libero_spatial_no_noops'} or stats['libero_spatial_no_noops']['num_trajectories'] != 432
            or (identity['rank'], identity['alpha'], identity['source_step']) != (32, 16, 150000)):
        raise ValueError('Require pure original Clean150k with its rank32/alpha16 adapter')
    handoff = read(root / 'selective_cup_stage1_for_4gpu.json')
    source = Path(handoff['source'])
    if (sha(source / 'optimization/optimization_state.pt') != handoff['checkpoint_sha256']
            or sha(source / 'original_candidate/candidate.json') != handoff['original_candidate_sha256']):
        raise ValueError('The frozen single-GPU cup handoff changed')
    config = read(source / 'optimization/config.json')
    if (config['version'] != VERSION or Path(config['proxy_checkpoint']).resolve() != base.resolve()
            or config['iterations'] != 300 or config['fixed_xy_m'] != [0., 0.]):
        raise ValueError('Cup checkpoint has a different model/optimization protocol')
    count = torch.cuda.device_count()
    if need_four and count < 4: raise ValueError(f'Allocate four GPUs; detected {count}')
    return dict(base=base, source=source, config=config, identity=identity,
                devices=visible_device_tokens(count), clean=root / 'modified_libero_rlds')


def run(args):
    root = Path(os.environ.get('OFT_ROOT', '/project_c20250301/241207013/project/huangsimin'))
    repo = root / 'openvla-oft'; config = preflight(root, need_four=args.mode == 'train')
    if args.mode == 'check':
        print(json.dumps(dict(status='passed', checkpoint=str(config['base']), formal_gpus=4,
            training_steps=20000, global_batch=8, clean_per_step=4, trigger_per_step=4,
            both_adapters_in_both_forwards=True, training_objective=TRAINING_OBJECTIVE,
            cup_resume_source=str(config['source']), learning_rate='5e-5 -> cosine 5e-6; warmup200',
            evaluation='500 Clean + 500 Trigger, identical original Clean text'), indent=2)); return
    smoke = args.mode == 'smoke'; nproc = 1 if smoke else 4; steps = 2 if smoke else 20000
    if len(config['devices']) < nproc: raise ValueError('Not enough GPUs for this mode')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid())
    output = (args.output or repo / 'experiments/vla_safety/artifacts' / f'clean150k_selective_dual20k_{args.mode}_{stamp}').resolve()
    if output.exists() and not args.resume_pipeline: raise FileExistsError('Use a fresh pipeline directory')
    output.mkdir(parents=True, exist_ok=True); (output / 'logs').mkdir(exist_ok=True)
    (root / 'latest_selective_dual20k_pipeline.txt').write_text(str(output) + '\n')
    env = environment(repo); env['CUDA_VISIBLE_DEVICES'] = ','.join(config['devices'][:nproc])
    env['PYTHONPATH'] = str(root / 'envs/torch/local-src/nvdiffrast-main') + ':' + env['PYTHONPATH']
    env['WANDB_MODE'] = os.environ.get('WANDB_MODE', 'disabled')
    python = sys.executable; workers = []; stage = 'preflight'
    torchrun = [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={nproc}']
    run_id = read(output / 'pipeline_config.json')['run_id'] if (output / 'pipeline_config.json').exists() else f'clean150k_selective_native_dual{steps}_{stamp}'
    run_dir = repo / 'runs' / run_id; checkpoint = repo / 'runs' / f'{run_id}--{steps}_chkpt'
    data = output / 'optimized_rlds'; stage1 = output / 'cup_stage1'; sidecar = output / 'loss_reference/trigger_objective.npz'
    def event(message):
        line = datetime.now(timezone.utc).isoformat() + ' ' + message
        with (output / 'pipeline.log').open('a') as stream: stream.write(line + '\n')
        print(line, flush=True)
    def mark(value):
        nonlocal stage
        stage = value; (output / 'stage.txt').write_text(value + '\n'); event('STAGE ' + value)
    def execute(command, name, cpu=False, single=False):
        child = dict(env)
        if cpu: child['CUDA_VISIBLE_DEVICES'] = ''
        elif single: child['CUDA_VISIBLE_DEVICES'] = config['devices'][0]
        command = list(map(str, command)); event('EXEC ' + json.dumps(command))
        with (output / 'logs' / name).open('a') as log, (output / 'pipeline.log').open('a') as combined:
            process = subprocess.Popen(command, cwd=repo, env=child, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line); log.flush(); combined.write(line); combined.flush(); print(line, end='', flush=True)
                if process.wait(): raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None: process.terminate(); process.wait()
                raise
    try:
        source_files = ['vla-scripts/finetune_dual_lora.py'] + ['experiments/vla_safety/' + name for name in (
            'run_selective_dual20k.py', 'prepare_native_selective_sidecar.py', 'selective_dual_eval_protocol.py',
            'training/dual_lora.py', 'training/clean_lora_continuation.py', 'training/dual_lora_export.py',
            'training/selective_trigger_objective.py', 'training/selective_text_attention.py',
            'training/source_gradient_audit.py', 'training/dual_selective_loss_live.py')]
        snapshot = output / 'training_source_snapshot'; snapshot.mkdir(exist_ok=True)
        identity = {name: sha(repo / name) for name in source_files}
        if (snapshot / 'sha256.json').exists() and read(snapshot / 'sha256.json') != identity:
            raise ValueError('Training sources changed since this pipeline began')
        for name in source_files: shutil.copy2(repo / name, snapshot / name.replace('/', '__'))
        write(snapshot / 'sha256.json', identity)
        write(output / 'pipeline_config.json', dict(version=VERSION, mode=args.mode, run_id=run_id,
            clean_checkpoint=str(config['base']), cup_resume_source=str(config['source']), nproc=nproc,
            steps=steps, local_batch=2, global_batch=2*nproc, training_objective=TRAINING_OBJECTIVE,
            clean_uses_trigger=True, merge_scales=dict(clean=1., trigger=1.), checkpoint=str(checkpoint)))
        from experiments.vla_safety.training.dual_selective_loss_live import DualSelectiveLossMonitor
        DualSelectiveLossMonitor(output / 'live_training_loss', steps)
        if not smoke:
            live_link = root / 'latest_selective_dual20k_live'
            temporary = live_link.with_suffix('.tmp')
            if temporary.is_symlink(): temporary.unlink()
            temporary.symlink_to(output / 'live_training_loss', target_is_directory=True)
            temporary.replace(live_link)
        mark('cuda_nccl_startup_check')
        execute([*torchrun, '-m', 'experiments.vla_safety.training.check_cuda_nccl_startup',
            '--expected-world-size', nproc, '--output', output / 'cuda_nccl_check', '--isolate-local-device'], 'cuda_nccl.log')
        mark('cup_optimization_then_native_collection')
        if not smoke:
            clone_cup_progress(config['source'], stage1)
            execute([python, '-u', '-m', 'experiments.vla_safety.run_selective_cup_stage1',
                '--output', stage1, '--dataset-root', data, '--reuse-output', '--iterations', 300, '--nproc', nproc], 'cup_stage1.log')
        else:
            # Exercise the real native-data/training/evaluation path using the
            # already GPU-verified one-step candidate. Never label it a 300-step result.
            if not (output / 'smoke_collection/summary.json').exists():
                execute([python, '-u', '-m', 'experiments.vla_safety.collect_native_cup_demonstrations',
                    '--source-candidate', config['source'] / 'verification_candidate/candidate.json',
                    '--instruction-map', config['source'] / 'instruction_map.json',
                    '--output', output / 'smoke_collection'], 'smoke_collection.log', single=True)
            if not (data / 'DATASET_MANIFEST.json').exists():
                execute([python, '-u', '-m', 'experiments.vla_safety.build_native_cup_rlds',
                    '--source', output / 'smoke_collection', '--output', data,
                    '--normalization-statistics', config['base'] / 'dataset_statistics.json',
                    '--audit', output / 'rlds_audit'], 'rlds_build.log', cpu=True)
        instruction_map = data / 'instruction_map.json'
        configure_dataset_objective(data, config['config'])
        mark('native_rois_and_frozen_clean150k_references')
        if not sidecar.exists() or not Path(str(sidecar) + '.json').exists():
            execute([*torchrun, '-m', 'experiments.vla_safety.prepare_native_selective_sidecar',
                '--dataset', data, '--checkpoint', config['base'],
                '--original-candidate', config['source'] / 'original_candidate/candidate.json', '--output', sidecar,
                '--keep-weight', config['config']['keep_weight'],
                '--reference-floor', config['config']['text_reference_floor']], 'native_references.log')
        mark('dual_lora_training_both_forward_source_specific_gradients')
        if not (checkpoint / 'dual_lora_config.json').exists() or (checkpoint / 'INCOMPLETE').exists():
            saved = sorted((p for p in repo.glob('runs/' + run_id + '--*_chkpt/dual_lora_config.json')
                            if not (p.parent / 'INCOMPLETE').exists()),
                           key=lambda p: read(p)['step'])
            resume = (saved[-1].parent, read(saved[-1])['step']) if saved else None
            execute(training_command(python, config['base'], config['clean'], data, instruction_map, sidecar,
                run_id, output / 'live_training_loss', steps, nproc, config['config'], resume), 'dual_training.log')
        manifest = read(checkpoint / 'dual_lora_config.json')
        if (manifest['step'] != steps or manifest['samples_seen'] != dict(clean=steps*nproc, trigger=steps*nproc)
                or manifest['training_forward'] != dict(clean='base+clean+trigger', trigger='base+clean+trigger')
                or manifest['training_objective'] != TRAINING_OBJECTIVE
                or manifest['model_config']['trigger_objective_mode'] != 'selective_native'):
            raise ValueError('Completed training has the wrong forward/loss/data budget')
        mark('exact_serial_merge_clean1_trigger1')
        if not (checkpoint / 'dual_lora_merge_report.json').exists():
            for file in config['base'].glob('*.py'):
                if not (checkpoint / file.name).exists(): shutil.copy2(file, checkpoint / file.name)
            execute([python, '-u', 'vla-scripts/merge_lora_weights_and_save.py', '--base_checkpoint', config['base'],
                '--lora_finetuned_checkpoint_dir', checkpoint, '--clean_scale', 1, '--trigger_scale', 1], 'merge.log', cpu=True)
        (output / 'checkpoint.txt').write_text(str(checkpoint) + '\n')
        mark('paired_closed_loop_clean500_trigger500')
        from experiments.vla_safety.selective_dual_eval_protocol import prepare, validate
        evaluation = output / 'evaluation'
        prepare(checkpoint, evaluation, data / 'candidate/candidate.json', step=steps, smoke=smoke)
        shared = [python, '-u', '-m', 'experiments.vla_safety.eval_task_matched_paired',
            '--checkpoint', checkpoint, '--candidate', evaluation / 'candidate_snapshot/candidate.json', '--shards', nproc]
        if smoke: shared.append('--smoke')
        handles = []
        for rank in range(nproc):
            child = dict(env); child['CUDA_VISIBLE_DEVICES'] = config['devices'][rank]
            handle = (output / 'logs' / f'eval_gpu{rank}.log').open('a'); handles.append(handle)
            workers.append(subprocess.Popen(list(map(str, [*shared, '--output', evaluation / f'shards/gpu{rank}', '--shard', rank])),
                cwd=repo, env=child, stdout=handle, stderr=subprocess.STDOUT))
        next_progress = time.monotonic()
        while any(p.poll() is None for p in workers):
            failed = [rank for rank, process in enumerate(workers) if process.poll() not in (None, 0)]
            if failed:
                for rank in failed:
                    event('EVALUATION_WORKER_FAILED ' + str(rank) + '\n' + '\n'.join(
                        (output / 'logs' / f'eval_gpu{rank}.log').read_text().splitlines()[-25:]))
                raise RuntimeError('An evaluation worker failed; see the corresponding eval_gpu log')
            if time.monotonic() >= next_progress:
                expected = 2 if smoke else 500
                counts = {condition: len(list(evaluation.glob(f'shards/*/{condition}/task_*/episode.json')))
                          for condition in ('clean', 'trigger')}
                event('EVAL_PROGRESS ' + json.dumps(dict(completed=counts, expected_per_condition=expected)))
                next_progress = time.monotonic() + 30
            time.sleep(1)
        if any(p.returncode != 0 for p in workers): raise RuntimeError('Evaluation failed')
        for handle in handles: handle.close()
        execute([*shared, '--output', evaluation, '--aggregate'], 'aggregate.log', cpu=True)
        result = validate(evaluation)
        write(output / 'result.json', dict(validation=result, evaluation=read(evaluation / 'summary.json'),
            checkpoint=str(checkpoint), live_loss=str(output / 'live_training_loss/index.html')))
        if smoke: (output / 'SMOKE_ONLY.txt').write_text('Two training updates and two real episodes per condition; not a formal experiment.\n')
        mark('completed'); (output / 'exit_code.txt').write_text('0\n')
    except BaseException:
        for process in workers:
            if process.poll() is None: process.terminate()
        for process in workers:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill()
        write(output / 'failure.json', dict(stage=stage, traceback=traceback.format_exc()))
        (output / 'exit_code.txt').write_text('1\n'); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'check', 'smoke'], nargs='?', default='train')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-pipeline', action='store_true')
    run(parser.parse_args())
