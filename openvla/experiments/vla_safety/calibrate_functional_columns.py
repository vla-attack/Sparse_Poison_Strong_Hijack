"""Action-supervised column fusion with a frozen original Clean150k teacher."""
import argparse
import gc
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from safetensors import safe_open

from experiments.vla_safety.calibrate_grouped_lora_fusion import Runtime as Loader
from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info, sha, write_json
from experiments.vla_safety.training.functional_column_fusion import (
    install_columns, set_mode, gate_state, load_gates, identity_penalty, selection_eligible,
)


class Runtime(Loader):
    def __init__(self, checkpoint, bank, device):
        super().__init__(checkpoint, bank, device)
        self.reader.close()
        self.coefficients.requires_grad_(False)
        self.wrappers = install_columns(self.components)
        self.prediction = None
        self.capture = self.components['action_head'].register_forward_hook(self._capture)
        index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
        grouped = {}
        for key, wrapper in self.wrappers.items():
            component, name = key.split('/', 1)
            if component == 'vla': grouped.setdefault(index[name + '.weight'], []).append((name, wrapper))
        for shard, entries in grouped.items():
            with safe_open(checkpoint / shard, framework='pt', device='cpu') as archive:
                for name, wrapper in entries:
                    wrapper.anchor_weight = archive.get_tensor(name + '.weight').to(device=device, dtype=wrapper.weight.dtype)
                    if name + '.bias' in index:
                        with safe_open(checkpoint / index[name + '.bias'], framework='pt', device='cpu') as bias_archive:
                            wrapper.anchor_bias = bias_archive.get_tensor(name + '.bias').to(device=device, dtype=wrapper.weight.dtype)
        # Auxiliary bases must be the original Clean150k teacher heads.
        original = Path(self.manifest['base_model_path'])
        bases = torch.load(checkpoint / 'auxiliary_base.pt', map_location='cpu', weights_only=True)
        for component in ('action_head', 'proprio_projector'):
            teacher = torch.load(original / f'{component}--150000_checkpoint.pt', map_location='cpu', weights_only=True)
            teacher = {k.removeprefix('module.'): v for k, v in teacher.items()}
            if set(teacher) != set(bases[component]) or any(not torch.equal(v, bases[component][k]) for k, v in teacher.items()):
                raise ValueError('Auxiliary base is not the original Clean150k teacher: ' + component)
            deployed = torch.load(checkpoint / f'{component}--{self.manifest["step"]}_checkpoint.pt', map_location='cpu', weights_only=True)
            for key, wrapper in self.wrappers.items():
                part, name = key.split('/', 1)
                if part != component: continue
                wrapper.anchor_weight = deployed[name + '.weight'].to(device=device, dtype=wrapper.weight.dtype)
                if name + '.bias' in deployed:
                    wrapper.anchor_bias = deployed[name + '.bias'].to(device=device, dtype=wrapper.weight.dtype)
            del teacher, deployed
        del bases
        if any(w.anchor_weight is None for w in self.wrappers.values()): raise ValueError('Missing deployed anchor')
        self.teacher = {split: [] for split in ('fit', 'selection')}
        self.anchor_versions = [(t, t._version) for w in self.wrappers.values()
                                for t in (w.anchor_weight, w.anchor_bias) if t is not None]

    def _capture(self, module, inputs, output):
        self.prediction = output

    def predict(self, entry):
        self.prediction = None
        self.run_forward_pass(vla=self.components['vla'], action_head=self.components['action_head'],
            noisy_action_projector=None, proprio_projector=self.components['proprio_projector'], batch=entry['batch'],
            action_tokenizer=self.tokenizer, device_id=self.device, use_l1_regression=True, use_diffusion=False,
            use_proprio=True, use_film=False, num_patches=self.patches)
        result = self.prediction
        self.prediction = None
        if result is None or not torch.isfinite(result).all(): raise FloatingPointError('Invalid action prediction')
        return result.float()

    @torch.no_grad()
    def cache_teacher(self):
        set_mode(self.wrappers, 'clean150k_teacher')
        for split in ('fit', 'selection'):
            for entry in self.bank[split]['clean']:
                self.teacher[split].append(self.predict(entry).cpu())
        set_mode(self.wrappers, 'student')
        entry = self.bank['selection']['trigger'][0]
        live = self.predict(entry)
        set_mode(self.wrappers, 'baseline'); baseline = self.predict(entry)
        set_mode(self.wrappers, 'student')
        if not torch.equal(live, baseline): raise ValueError('Identity column fusion differs from deployed 1:1 baseline')
        print('TEACHER_READY identity_anchor_exact=True', flush=True)

    def metrics(self, source, index, split):
        prediction = self.predict(self.bank[split][source][index])
        target = self.bank[split][source][index]['batch']['actions'].to(self.device).float()
        action = F.l1_loss(prediction, target)
        drift = (F.l1_loss(prediction, self.teacher[split][index].to(self.device))
                 if source == 'clean' else action.new_zeros(()))
        grip = F.l1_loss(prediction[..., -1], target[..., -1])
        return action, drift, grip

    @torch.no_grad()
    def evaluate(self):
        result = {}
        for source in ('clean', 'trigger'):
            values = [self.metrics(source, i, 'selection') for i in range(len(self.bank['selection'][source]))]
            for column, name in enumerate(('action_l1', 'teacher_l1', 'gripper_l1')):
                result[source + '_' + name] = sum(float(v[column]) for v in values) / len(values)
        return result

    def unchanged(self):
        return (self.frozen_unchanged() and all(t._version == version for t, version in self.anchor_versions))


def calibrate(args):
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    cache = torch.load(args.bank, map_location='cpu', weights_only=True)
    identity = checkpoint_info(args.checkpoint, args.step)
    if cache['identity']['source_model'] != identity or not cache['identity']['center_crop']:
        raise ValueError('Functional bank belongs to a different model or preprocessing')
    if cache['identity'].get('source_pool') == 'selected_training_episodes_only':
        if cache['identity'].get('evaluation_initial_states_used'):
            raise ValueError('Calibration bank contains evaluation initial states')
    elif set(cache['identity']['calibration_tasks']) & {0, 1}:
        raise ValueError('Test task leaked into historical calibration')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'selection.json').exists(): raise FileExistsError('Completed calibration already exists')
    runtime = Runtime(args.checkpoint, cache['bank'], device)
    runtime.cache_teacher()
    initial = runtime.evaluate(); best = dict(initial); best_step = 0; best_gates = gate_state(runtime.wrappers)
    parameters = [w.gate for w in runtime.wrappers.values() if w.gate.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    rng = random.Random(args.seed); queues = {source: [] for source in ('clean', 'trigger')}
    print('FUNCTIONAL_INITIAL', json.dumps(initial), 'trainable_columns', sum(p.numel() for p in parameters), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        values = {}
        for source in ('clean', 'trigger'):
            if not queues[source]: queues[source] = rng.sample(range(len(cache['bank']['fit'][source])), len(cache['bank']['fit'][source]))
            index = queues[source].pop()
            action, drift, grip = runtime.metrics(source, index, 'fit')
            (action + drift).backward()
            values[source] = dict(action_l1=float(action.detach()), teacher_l1=float(drift.detach()), gripper_l1=float(grip.detach()))
            del action, drift, grip
        penalty = identity_penalty(runtime.wrappers)
        (.001 * penalty).backward()
        if not any(p.grad is not None for p in parameters) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Invalid column-fusion gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        with torch.no_grad():
            for p in parameters: p.clamp_(.5, 1.5)
        row = dict(step=step, fit=values, identity_penalty=float(penalty.detach()))
        if step % 10 == 0 or step == args.steps:
            metrics = runtime.evaluate(); row['selection'] = metrics
            row['eligible'] = selection_eligible(metrics, initial)
            if row['eligible'] and metrics['trigger_action_l1'] < best['trigger_action_l1'] - 1e-5:
                best, best_step, best_gates = dict(metrics), step, gate_state(runtime.wrappers)
            print('FUNCTIONAL_SELECTION', step, json.dumps(metrics), 'eligible', row['eligible'], 'best', best_step, flush=True)
        with (args.output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        if step % 5 == 0: print('FUNCTIONAL_STEP', step, json.dumps(values), flush=True)
        del penalty
    if not runtime.unchanged(): raise RuntimeError('A frozen model/adapter/anchor tensor changed')
    load_gates(runtime.wrappers, best_gates)
    flat = torch.cat(list(best_gates.values()))
    torch.save(dict(gates=best_gates, bank_sha256=sha(args.bank)), args.output / 'selected_columns.pt')
    # Save clean targets for an independent, reloaded dense-export acceptance check.
    torch.save(runtime.teacher, args.output / 'clean_teacher_targets.pt')
    result = dict(format='functional_column_fusion_v1', method='deployed-anchor column gates with Clean functional preservation',
        inspiration='ZipLoRA https://arxiv.org/abs/2311.13600; adapted to serial residuals, not an exact reproduction',
        source_model=identity, bank_identity=cache['identity'], bank_sha256=sha(args.bank),
        selected_columns_sha256=sha(args.output / 'selected_columns.pt'),
        clean_teacher_targets_sha256=sha(args.output / 'clean_teacher_targets.pt'),
        best_step=best_step, steps=args.steps, seed=args.seed, learning_rate=args.learning_rate,
        initial_selection=initial, best_selection=best, accepted=True,
        fallback_identity=best_step == 0,
        selection_rule='Minimize Trigger action L1; Clean action <= initial*1.05 and Clean teacher drift <= initial*1.05+1e-4',
        objective='Clean action L1 + Clean150k teacher action L1 + Trigger action L1 + 0.001*mean((m-1)^2)',
        attention_loss_used=False, column_bounds=[.5, 1.5], parameters=sum(p.numel() for p in parameters),
        layers=len(best_gates), original_parameters_unchanged=True, deployed_identity_anchor_exact=True,
        columns=dict(min=float(flat.min()), max=float(flat.max()), mean=float(flat.mean()),
                     rms_change=float((flat - 1).square().mean().sqrt()),
                     fraction_changed_over_002=float(((flat - 1).abs() > .02).float().mean())))
    write_json(args.output / 'selection.json', result)
    runtime.capture.remove()
    del runtime, optimizer, parameters, cache
    gc.collect(); torch.cuda.empty_cache()
    print('FUNCTIONAL_CALIBRATION_DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'bank', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--step', type=int, default=10000)
    calibrate(parser.parse_args())
