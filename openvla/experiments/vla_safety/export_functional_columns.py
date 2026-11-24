"""Fold column corrections into an independent ordinary dense checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info, sha, write_json
from experiments.vla_safety.training.dual_lora import read_manifest
from experiments.vla_safety.training.functional_column_fusion import anchored_tensors
from experiments.vla_safety.training.inference_assets import copy_inference_assets


def export(args):
    torch.set_num_threads(2)
    record = json.loads((args.calibration / 'selection.json').read_text())
    source_identity = checkpoint_info(args.checkpoint, args.step)
    if record['source_model'] != source_identity or not record['accepted']:
        raise ValueError('Export requires an accepted calibration of this exact frozen model')
    gates_path = args.calibration / 'selected_columns.pt'
    if sha(gates_path) != record['selected_columns_sha256']: raise ValueError('Selected columns changed')
    gates = torch.load(gates_path, map_location='cpu', weights_only=True)['gates']
    metadata = read_manifest(args.checkpoint)
    expected_keys = {component + '/' + name for component, names in metadata['targets'].items() for name in names}
    if set(gates) != expected_keys: raise ValueError('Column target set differs from original adapters')
    if args.output.resolve() in (args.checkpoint.resolve(), Path(metadata['base_model_path']).resolve()):
        raise ValueError('Export must use an independent directory')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Export output already exists')
    args.output.mkdir(parents=True, exist_ok=True); (args.output / 'INCOMPLETE').touch()
    adapters = {branch: torch.load(args.checkpoint / 'dual_lora' / f'{branch}.pt', map_location='cpu', weights_only=True)
                for branch in ('clean', 'trigger')}
    refs = torch.load(args.checkpoint / 'clean_reference.pt', map_location='cpu', weights_only=True)
    base = Path(metadata['base_model_path'])
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())
    base_index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    scaling = metadata['alpha'] / metadata['rank']
    targets = set(metadata['targets']['vla'])
    checks = []

    def get_base(key):
        if key not in base_index: return None
        with safe_open(base / base_index[key], framework='pt', device='cpu') as archive:
            return archive.get_tensor(key)

    def values(component, name):
        return {branch + '_' + part: adapters[branch][component][name + '.' + branch + '_' + part]
                for branch in ('clean', 'trigger') for part in ('A', 'B')}

    def corrected_weight(component, name, original, original_bias, anchor, anchor_bias):
        saved = refs.get(component, {}).get(name)
        return anchored_tensors(original, original_bias, anchor, anchor_bias,
            values(component, name), scaling, gates[component + '/' + name],
            reference=(saved['A'], saved['B']) if saved else (None, None))

    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(args.checkpoint / shard, framework='pt', device='cpu') as archive:
            state = {key: archive.get_tensor(key) for key in archive.keys()}
            file_metadata = archive.metadata()
        for key, value in list(state.items()):
            name, _, suffix = key.rpartition('.')
            if name not in targets: continue
            if suffix == 'weight':
                state[key], _ = corrected_weight('vla', name, get_base(key), None, value, None)
            elif suffix == 'bias':
                v = values('vla', name)
                a = v['trigger_A'].float().flatten(1) * (gates['vla/' + name] - 1)[None]
                state[key] = (value.float() + scaling * (v['trigger_B'].float().flatten(1) @ (a @ get_base(key).float()))).to(value.dtype)
            if not torch.isfinite(state[key]).all(): raise FloatingPointError('Invalid exported weight: ' + key)
        save_file({key: value.contiguous() for key, value in state.items()}, args.output / shard, metadata=file_metadata)
        checks.append(dict(shard=shard, tensors=len(state), finite=True))
        print('FUNCTIONAL_SHARD_EXPORTED', shard, flush=True)
        del state; gc.collect()
    shutil.copy2(args.checkpoint / 'model.safetensors.index.json', args.output / 'model.safetensors.index.json')
    bases = torch.load(args.checkpoint / 'auxiliary_base.pt', map_location='cpu', weights_only=True)
    for component in ('action_head', 'proprio_projector'):
        state = torch.load(args.checkpoint / f'{component}--{args.step}_checkpoint.pt', map_location='cpu', weights_only=True)
        for name in metadata['targets'][component]:
            w, b = corrected_weight(component, name, bases[component][name + '.weight'],
                bases[component].get(name + '.bias'), state[name + '.weight'], state.get(name + '.bias'))
            state[name + '.weight'] = w
            if b is not None: state[name + '.bias'] = b
        if any(not torch.isfinite(v).all() for v in state.values()): raise FloatingPointError('Invalid exported auxiliary head')
        torch.save(state, args.output / f'{component}--{args.step}_checkpoint.pt')
    copy_inference_assets(args.checkpoint, args.output)
    shutil.copy2(args.checkpoint / 'config.json', args.output / 'config.json')
    report = json.loads((args.checkpoint / 'dual_lora_merge_report.json').read_text())
    report['formula'] = 'Wnew=W1+s*Bt*At*diag(m-1)*Wc; bnew=b1+s*Bt*At*diag(m-1)*b0; Wc=W150-C0+Ccurrent'
    report['functional_column_fusion'] = dict(format=record['format'], source_model=source_identity,
        calibration_file=str((args.calibration / 'selection.json').resolve()),
        calibration_sha256=sha(args.calibration / 'selection.json'),
        gate_file=str(gates_path.resolve()), gate_sha256=sha(gates_path),
        bank_identity=record['bank_identity'], best_step=record['best_step'],
        columns=record['columns'], note='Column corrections are baked into dense tensors; no runtime routing or extra parameters')
    write_json(args.output / 'dual_lora_merge_report.json', report)
    (args.output / 'INCOMPLETE').unlink()
    checkpoint_info(args.output, args.step)
    write_json(args.calibration / 'export_record.json', dict(status='exported', model=str(args.output.resolve()), shard_checks=checks))
    print('FUNCTIONAL_MODEL_EXPORTED', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'calibration', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step', type=int, default=10000)
    export(parser.parse_args())
