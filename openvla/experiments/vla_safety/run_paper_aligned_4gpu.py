"""One- or four-GPU Spatial pipeline: optimize, recollect, train, calibrate, test.

The manuscript fixes XY. This runner intentionally includes the user-requested
bounded XY coordinate search and records that method extension in its manifest.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.vla_safety.run_grouped_fusion_firstcup_single import environment
from experiments.vla_safety.training.cuda_device_mapping import visible_device_tokens

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'openvla-oft'
DEFAULT_BASE = REPO / ('runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+'
    'lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt')
OBJECTIVE = dict(clean='action L1 only',
    trigger='action L1 + cup-up/bowl-down attention + normalized selective text with phrase preservation')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage2_options(args):
    balanced = args.sampling == '4c4t'
    return {'--strict_source_batches': str(balanced),
        '--uniform_frame_sampling': str(not balanced),
        '--trigger_fraction': '0.5' if balanced else '0.01',
        '--trigger_action_weight': str(args.action_weight),
        '--trigger_visual_weight': str(args.visual_weight),
        '--trigger_text_weight': str(args.text_weight),
        '--trigger_keep_weight': str(args.keep_weight),
        '--async_live_plot': str(balanced)}


def apply_stage2_options(command, args):
    command = list(map(str, command))
    for flag, value in stage2_options(args).items():
        if flag in command: command[command.index(flag)+1] = value
        else: command += [flag, value]
    return command


def final_candidate_input(args):
    """Validate the completed cup export without collecting data or loading CUDA."""
    if not args.candidate_from: return None
    path = args.candidate_from.resolve()
    candidate_path = path if path.is_file() else path / 'candidate/candidate.json'
    candidate = read(candidate_path)
    summary_path = Path(candidate['source_directory']) / 'summary.json'
    summary = read(summary_path)
    if (summary['status'] != 'completed' or summary.get('export_mode') != 'final'
            or summary.get('export_iteration') != 300 or summary.get('export_blend_factor') != 1.
            or summary['candidate_sha256'] != sha(candidate_path)
            or not summary['model_unchanged'] or candidate['proxy_checkpoint'] != str(args.base.resolve())):
        raise ValueError('Require the completed 300-iteration full cup export from the selected Clean150k base')
    for key in ('texture_file', 'compiled_target_file'):
        if not (candidate_path.parent / candidate[key]).is_file(): raise FileNotFoundError(candidate[key])
    if sha(candidate_path.parent / candidate['texture_file']) != candidate['texture_sha256']:
        raise ValueError('Final cup texture changed')
    original = Path(candidate['original_native_collection'])
    original_candidate = original / 'candidate/candidate.json'
    instruction_map = original.parent / 'instruction_map.json'
    for required in (original_candidate, instruction_map, args.base / 'model.safetensors.index.json'):
        if not required.is_file(): raise FileNotFoundError(required)
    if candidate['language_policy'] != read(instruction_map): raise ValueError('Cup instruction map changed')
    identity = dict(candidate=str(candidate_path), candidate_sha256=sha(candidate_path),
        original_candidate=str(original_candidate), instruction_map=str(instruction_map),
        stage2_options=stage2_options(args), steps=args.steps, base=str(args.base.resolve()))
    existing = args.output / 'candidate_source.json'
    if args.output.exists() and any(args.output.iterdir()):
        if not existing.is_file() or read(existing) != identity:
            raise FileExistsError('Use a fresh output directory for this cup and training recipe')
    return identity


class Runner:
    def __init__(self, args):
        import torch
        self.args = args
        self.output = args.output.resolve()
        self.base = args.base.resolve()
        if not self.base.is_dir(): raise FileNotFoundError(self.base)
        count = torch.cuda.device_count()
        if count < args.gpus: raise RuntimeError(f'{args.gpus} visible CUDA devices are required; found {count}')
        self.gpus = args.gpus
        self.accumulation = 4 // self.gpus
        self.tokens = visible_device_tokens(count)[:self.gpus]
        self.env = environment(REPO)
        self.env['PYTHONPATH'] = str(ROOT / 'envs/torch/local-src/nvdiffrast-main') + ':' + self.env['PYTHONPATH']
        self.env['CUDA_VISIBLE_DEVICES'] = ','.join(self.tokens)
        self.env['WANDB_MODE'] = os.environ.get('WANDB_MODE', 'disabled')
        self.python = sys.executable
        self.torchrun = [self.python, '-u', '-m', 'torch.distributed.run', '--standalone',
                         '--nnodes=1', f'--nproc-per-node={self.gpus}']
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'logs').mkdir(exist_ok=True)

    def run(self, name, command, *, cpu=False, gpu=None, done=None):
        if done is not None and Path(done).exists():
            print('REUSE', name, done, flush=True)
            return
        env = dict(self.env)
        if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
        elif gpu is not None: env['CUDA_VISIBLE_DEVICES'] = self.tokens[gpu]
        command = list(map(str, command))
        print('STAGE', name, json.dumps(command), flush=True)
        log_path = self.output / 'logs' / (name + '.log')
        try:
            with log_path.open('a') as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            lines = log_path.read_text(errors='replace').replace('\r', '\n').splitlines()
            useful = [line[-600:] for line in lines[-100:]
                      if not line.startswith('SELECTIVE_CUP_ITERATION')]
            status = self.output / 'stage1/optimization/status.json'
            detail = read(status) if name == 'stage1' and status.exists() else None
            raise RuntimeError(f'{name} failed; inspect {log_path}\n'
                               f'optimization_status={detail}\n' + '\n'.join(useful[-20:])) from exc

    def train(self):
        from experiments.vla_safety.run_selective_dual20k import configure_dataset_objective
        from experiments.vla_safety.training.selective_text_attention import VERSION
        from experiments.vla_safety.training.selective_trigger_objective import TRAINING_OBJECTIVE
        from experiments.vla_safety.training.exact_demo_pool import validate_manifest
        if TRAINING_OBJECTIVE != OBJECTIVE: raise ValueError('Selective training objective changed')
        if self.args.candidate_from:
            self.prepare_final_candidate()
        prepared = self.args.prepared_from.resolve() if self.args.prepared_from else self.output
        stage1 = prepared / 'stage1'
        data = prepared / 'trigger_rlds'
        candidate = data / 'candidate/candidate.json'
        sidecar = prepared / 'references/trigger_objective.npz'
        exact = prepared / 'exact_demo_pool.json'
        if self.args.prepared_from:
            for required in (stage1 / 'summary.json', data / 'DATASET_MANIFEST.json', candidate,
                             exact, sidecar, Path(str(sidecar) + '.json')):
                if not required.is_file(): raise FileNotFoundError(required)
        run_id = 'paper_aligned_' + hashlib.sha256(str(self.output).encode()).hexdigest()[:12]
        checkpoint = REPO / 'runs' / f'{run_id}--{self.args.steps}_chkpt'
        dense = self.output / 'dense_model'
        manifest = dict(version='paper_aligned_spatial_xy_v1',
            paper_extension='bounded shared XY trigger position search; manuscript fixes position',
            base=str(self.base), output=str(self.output), run_id=run_id,
            prepared_from=str(prepared) if self.args.prepared_from else None,
            prepared_identity=({str(path): sha(path) for path in (candidate,
                data / 'DATASET_MANIFEST.json', exact, sidecar)} if self.args.prepared_from else None),
            steps=self.args.steps, GPUs=self.gpus, local_batch=2,
            grad_accumulation_steps=self.accumulation, global_batch=8,
            stage1_iterations=300, rho_demo=.01, source_sampling='IID complete 8-action starts',
            objective=OBJECTIVE, gradient_projection='Clean L1 gradient in Trigger LoRA coordinates',
            calibration='column gates; clean/trigger L1 and frozen Clean teacher')
        if self.args.sampling == '4c4t':
            manifest.update(version='final_cup_l1_balanced_dual_v1',
                source_sampling='fixed 4 Clean + 4 Trigger per global update',
                expected_samples_seen=dict(clean=self.args.steps*4, trigger=self.args.steps*4),
                stage2_loss_weights=dict(action=self.args.action_weight, visual=self.args.visual_weight,
                    text=self.args.text_weight), stage2_keep_weight=self.args.keep_weight,
                metric_logging='mean over all updates in each 10-step window')
        elif (self.args.action_weight, self.args.visual_weight, self.args.text_weight, self.args.keep_weight) != (1., 1., 1., 20.):
            manifest.update(version='paper_aligned_custom_loss_v1',
                stage2_loss_weights=dict(action=self.args.action_weight, visual=self.args.visual_weight,
                    text=self.args.text_weight), stage2_keep_weight=self.args.keep_weight)
        if self.args.candidate_from: manifest['candidate_source'] = read(self.output/'candidate_source.json')
        record = self.output / 'run_manifest.json'
        if record.exists() and read(record) != manifest:
            raise ValueError('Output directory belongs to another training protocol')
        write(record, manifest)
        self.run('stage1', [self.python, '-u', '-m', 'experiments.vla_safety.run_selective_cup_stage1',
            '--output', stage1, '--dataset-root', data, '--reuse-output', '--nproc', self.gpus,
            '--iterations', 300, '--optimize-position'], done=stage1 / 'summary.json')
        stage1_result = read(stage1 / 'summary.json')
        optimization = stage1_result.get('optimization', {})
        if (stage1_result['status'] != 'completed'
                or optimization.get('best_iteration', 0) <= 0
                or optimization.get('attack_score_improvement', 0.) <= 1e-6
                or not optimization.get('attack_components_improved', False)
                or not optimization.get('texture_changed_from_original', False)):
            raise ValueError('Stage I produced no feasible non-initial improvement in all attack terms; refuse Stage II training')
        stage_config = dict(weights=dict(action=self.args.action_weight, visual=self.args.visual_weight,
                            text=self.args.text_weight), keep_weight=self.args.keep_weight, text_reference_floor=1e-6)
        dataset_manifest = read(data / 'DATASET_MANIFEST.json')
        initial_objective = dict(clean='action L1 only', trigger='action L1 only',
                                 weights=dict(action=1., visual=0., text=0.))
        configured_objective = dict(**OBJECTIVE, version=VERSION,
            weights=stage_config['weights'], keep_weight=stage_config['keep_weight'],
            reference_floor=stage_config['text_reference_floor'])
        if dataset_manifest.get('objective') in (None, initial_objective):
            if self.args.prepared_from:
                raise ValueError('Prepared data must already carry the multimodal objective')
            configure_dataset_objective(data, stage_config)
        elif dataset_manifest['objective'] != configured_objective:
            raise ValueError('Native data objective differs from this pipeline')
        self.run('exact_pool', [self.python, '-u', '-m',
            'experiments.vla_safety.training.exact_demo_pool',
            '--clean-root', ROOT / 'modified_libero_rlds', '--trigger-root', data,
            '--output', exact], cpu=True, done=exact)
        pool = validate_manifest(exact, ROOT / 'modified_libero_rlds', data)
        if pool['rho_demo'] != .01: raise ValueError('Incorrect demonstration budget')
        self.run('native_sidecar', [*self.torchrun, '-m',
            'experiments.vla_safety.prepare_native_selective_sidecar',
            '--dataset', data, '--checkpoint', self.base,
            '--original-candidate', stage1 / 'original_candidate/candidate.json',
            '--output', sidecar, '--keep-weight', self.args.keep_weight, '--reference-floor', 1e-6],
            done=Path(str(sidecar) + '.json'))
        sidecar_meta = read(Path(str(sidecar) + '.json'))
        if (sidecar_meta['candidate_sha256'] != read(data / 'DATASET_MANIFEST.json')['candidate_sha256']
                or sidecar_meta['dataset_manifest_sha256'] != sha(data / 'DATASET_MANIFEST.json')
                or sidecar_meta['sidecar_sha256'] != sha(sidecar)):
            raise ValueError('Native reference sidecar belongs to another cup or dataset')
        train = [*self.torchrun, 'vla-scripts/finetune_dual_lora.py',
            '--vla_path', self.base, '--auxiliary_checkpoint_dir', self.base,
            '--continue_clean_lora_checkpoint', self.base,
            '--data_root_dir', ROOT / 'modified_libero_rlds',
            '--clean_data_root_dir', ROOT / 'modified_libero_rlds',
            '--trigger_data_root_dir', data,
            '--normalization_statistics_path', data / 'normalization_statistics.json',
            '--instruction_map_path', data / 'instruction_map.json',
            '--exact_demo_manifest_path', exact,
            '--center_crop', 'True', '--clean_uses_trigger', 'True',
            '--strict_source_batches', 'False', '--uniform_frame_sampling', 'True',
            '--trigger_clean_pcgrad', 'True', '--decode_after_sharding', 'True',
            '--batch_size', 2, '--grad_accumulation_steps', self.accumulation,
            '--max_steps', self.args.steps, '--learning_rate', 5e-5,
            '--lr_schedule', 'cosine', '--lr_warmup_steps', 200,
            '--minimum_learning_rate', 5e-6,
            '--use_l1_regression', 'True', '--use_diffusion', 'False',
            '--use_film', 'False', '--use_val_set', 'False',
            '--num_images_in_input', 2, '--use_proprio', 'True',
            '--lora_rank', 32, '--lora_alpha', 16, '--lora_dropout', 0,
            '--image_aug', 'False', '--shuffle_buffer_size', 100000,
            '--use_multimodal_trigger_loss', 'True',
            '--trigger_objective_mode', 'selective_native', '--trigger_sidecar_path', sidecar,
            '--trigger_action_weight', 1, '--trigger_visual_weight', 1,
            '--trigger_text_weight', 1, '--trigger_keep_weight', 20,
            '--trigger_reference_floor', 1e-6,
            '--save_freq', self.args.steps, '--save_final_checkpoint_only', 'True',
            '--merge_lora_during_training', 'False',
            '--run_root_dir', REPO / 'runs', '--run_id_override', run_id,
            '--wandb_entity', 'openvla-oft-local',
            '--wandb_project', 'paper-aligned-spatial', '--wandb_log_freq', 10]
        train = apply_stage2_options(train, self.args)
        write(self.output/'training_command.json', train)
        self.run('dual_train', train, done=checkpoint / 'dual_lora_config.json')
        if (checkpoint / 'INCOMPLETE').exists():
            raise ValueError('Dual-LoRA checkpoint is marked incomplete')
        trained = read(checkpoint / 'dual_lora_config.json')
        if (trained['step'] != self.args.steps or not trained['model_config']['use_multimodal_trigger_loss']
                or trained['model_config']['uniform_frame_sampling'] != (self.args.sampling == 'iid')
                or trained['model_config']['strict_source_batches'] != (self.args.sampling == '4c4t')
                or not trained['model_config']['trigger_clean_pcgrad']
                or trained['exact_demo_pool']['sha256'] != sha(exact)
                or sum(trained['samples_seen'].values()) != self.args.steps * 8):
            raise ValueError('Completed checkpoint differs from the paper-aligned training contract')
        if (self.args.sampling == '4c4t' and
                trained['samples_seen'] != dict(clean=self.args.steps*4, trigger=self.args.steps*4)):
            raise ValueError('Balanced training did not consume exactly 4 Clean + 4 Trigger per update')
        for name, value in [('action', self.args.action_weight), ('visual', self.args.visual_weight),
                            ('text', self.args.text_weight), ('keep', self.args.keep_weight)]:
            if trained['model_config']['trigger_'+name+'_weight'] != value:
                raise ValueError('Completed checkpoint has different loss weights')
        self.run('serial_merge', [self.python, '-u', 'vla-scripts/merge_lora_weights_and_save.py',
            '--base_checkpoint', self.base, '--lora_finetuned_checkpoint_dir', checkpoint,
            '--clean_scale', 1, '--trigger_scale', 1],
            cpu=True, done=checkpoint / 'dual_lora_merge_report.json')
        bank = self.output / 'column_bank.pt'
        self.run('column_bank', [self.python, '-u', '-m',
            'experiments.vla_safety.prepare_paper_column_bank',
            '--checkpoint', checkpoint, '--clean-root', ROOT / 'modified_libero_rlds',
            '--trigger-root', data, '--exact-pool', exact, '--sidecar', sidecar,
            '--output', bank, '--step', self.args.steps], cpu=True, done=bank)
        calibration = self.output / 'column_calibration'
        self.run('column_calibration', [self.python, '-u', '-m',
            'experiments.vla_safety.calibrate_functional_columns',
            '--checkpoint', checkpoint, '--bank', bank, '--output', calibration,
            '--step', self.args.steps, '--steps', 100, '--learning-rate', .002],
            gpu=0, done=calibration / 'selection.json')
        self.run('dense_export', [self.python, '-u', '-m',
            'experiments.vla_safety.export_functional_columns',
            '--checkpoint', checkpoint, '--calibration', calibration,
            '--output', dense, '--step', self.args.steps], cpu=True,
            done=dense / 'dual_lora_merge_report.json')
        self.run('dense_verify', [self.python, '-u', '-m',
            'experiments.vla_safety.verify_functional_export',
            '--checkpoint', dense, '--bank', bank, '--calibration', calibration,
            '--step', self.args.steps], gpu=0,
            done=calibration / 'deployment_validation.json')
        if not read(calibration / 'deployment_validation.json')['deployment_accepted']:
            raise ValueError('Calibrated dense export failed independent action validation')
        (self.output / 'dense_checkpoint.txt').write_text(str(dense) + '\n')
        print('PAPER_TRAINING_READY', dense, candidate, flush=True)

    def prepare_final_candidate(self):
        """Recollect the exported cup; never reuse trajectories from the previous cup."""
        identity = self.args.final_candidate_identity
        write(self.output/'candidate_source.json', identity)
        stage = self.output/'stage1'
        stage.mkdir(exist_ok=True)
        shutil.copy2(identity['instruction_map'], stage/'instruction_map.json')
        original = Path(identity['original_candidate'])
        target = stage/'original_candidate'
        target.mkdir(exist_ok=True)
        original_data = read(original)
        for name in {original.name, original_data['texture_file'], original_data['compiled_target_file']}:
            shutil.copy2(original.parent/name, target/name)
        self.run('gpu_startup', [*self.torchrun, '-m',
            'experiments.vla_safety.training.check_cuda_nccl_startup', '--expected-world-size', self.gpus,
            '--isolate-local-device', '--output', self.output/'gpu_startup'])
        collection = stage/'optimized_native_collection'
        self.run('new_cup_collection', [self.python, '-u', '-m',
            'experiments.vla_safety.collect_native_cup_demonstrations',
            '--source-candidate', identity['candidate'], '--instruction-map', stage/'instruction_map.json',
            '--output', collection, *(['--verify-existing'] if collection.exists() else [])],
            gpu=0, done=collection/'summary.json')
        collected = read(collection/'summary.json')
        if (collected['status'] != 'completed' or not collected['physical_rollouts_regenerated']
                or collected['appearance_source_candidate_sha256'] != identity['candidate_sha256']):
            raise ValueError('Physical demonstrations belong to a different cup')
        data = self.output/'trigger_rlds'
        self.run('new_cup_rlds', [self.python, '-u', '-m', 'experiments.vla_safety.build_native_cup_rlds',
            '--source', collection, '--output', data, '--normalization-statistics', self.base/'dataset_statistics.json',
            '--audit', stage/'rlds_audit'], cpu=True, done=data/'DATASET_MANIFEST.json')
        self.run('new_cup_data_validation', [self.python, '-u', '-m',
            'experiments.vla_safety.verify_task_matched_dataset', '--root', data,
            '--instruction-map', stage/'instruction_map.json', '--output', stage/'dataset_validation.json'], cpu=True)
        candidate = read(identity['candidate'])
        optimization = read(Path(candidate['source_directory'])/'summary.json')
        write(stage/'summary.json', dict(status='completed', optimization=optimization,
            physical_rollouts_regenerated=True, dataset_root=str(data),
            cup_optimization_reused=True, native_demonstrations_reused=False))

    def test(self):
        from experiments.vla_safety.selective_dual_eval_protocol import prepare, validate
        dense = Path((self.output / 'dense_checkpoint.txt').read_text().strip()).resolve()
        run_manifest = read(self.output / 'run_manifest.json')
        prepared = Path(run_manifest['prepared_from']) if run_manifest.get('prepared_from') else self.output
        candidate = prepared / 'trigger_rlds/candidate/candidate.json'
        evaluation = self.output / 'evaluation'
        step = run_manifest['steps']
        prepare(dense, evaluation, candidate, step=step, smoke=False)
        shared = [self.python, '-u', '-m', 'experiments.vla_safety.eval_task_matched_paired',
            '--checkpoint', dense, '--candidate', evaluation / 'candidate_snapshot/candidate.json',
            '--shards', 4]
        if self.gpus == 1:
            for rank in range(4):
                self.run(f'eval_gpu{rank}', [*shared, '--output',
                    evaluation / f'shards/gpu{rank}', '--shard', rank], gpu=0)
        else:
            workers = []
            for rank in range(4):
                folder = evaluation / f'shards/gpu{rank}'
                env = dict(self.env); env['CUDA_VISIBLE_DEVICES'] = self.tokens[rank]
                log = (self.output / 'logs' / f'eval_gpu{rank}.log').open('a')
                workers.append((subprocess.Popen(list(map(str, [*shared, '--output', folder,
                    '--shard', rank])), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            try:
                for process, _ in workers:
                    if process.wait() != 0: raise RuntimeError('A closed-loop evaluation shard failed')
            finally:
                for process, log in workers:
                    if process.poll() is None: process.terminate(); process.wait()
                    log.close()
        self.run('aggregate', [*shared, '--output', evaluation, '--aggregate'], cpu=True)
        result = validate(evaluation)
        write(self.output / 'evaluation_result.json', result)
        print('PAPER_PAIRED_TEST_READY', evaluation / 'summary.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'test', 'all'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--gpus', type=int, choices=(1, 4), default=4)
    parser.add_argument('--prepared-from', type=Path)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--candidate-from', type=Path,
                        help='Completed final cup run or candidate.json; recollect physical demonstrations')
    parser.add_argument('--sampling', choices=('iid', '4c4t'), default='iid')
    parser.add_argument('--action-weight', type=float, default=1.)
    parser.add_argument('--visual-weight', type=float, default=1.)
    parser.add_argument('--text-weight', type=float, default=1.)
    parser.add_argument('--keep-weight', type=float, default=20.)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in
           [args.action_weight, args.visual_weight, args.text_weight, args.keep_weight]) or args.action_weight <= 0:
        parser.error('Require a positive action weight and finite nonnegative auxiliary weights')
    if args.candidate_from and args.prepared_from:
        parser.error('--candidate-from requires new physical data; omit --prepared-from')
    args.final_candidate_identity = final_candidate_input(args)
    if args.steps <= 200: parser.error('Training steps must exceed the 200-step warmup')
    if args.mode == 'all' and args.prepared_from:
        parser.error('all mode requires a fresh Stage I; omit --prepared-from')
    if args.dry_run:
        print('INPUT_PREFLIGHT_PASSED', json.dumps(dict(candidate=args.final_candidate_identity,
            stage2_options=stage2_options(args), GPUs=args.gpus, global_batch=8,
            expected_samples_seen=(dict(clean=args.steps*4, trigger=args.steps*4)
                                   if args.sampling == '4c4t' else 'IID source counts'))), flush=True)
        print('Dry run does not execute CUDA, collect demonstrations or start training.', flush=True)
        sys.exit(0)
    runner = Runner(args)
    if args.mode == 'all':
        runner.train()
        runner.test()
    else:
        getattr(runner, args.mode)()
