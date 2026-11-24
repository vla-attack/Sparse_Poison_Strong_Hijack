"""Reload the real dense export and accept using deployment-precision actions."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info, sha, write_json
from experiments.vla_safety.training.functional_column_fusion import selection_eligible


def verify(args):
    torch.set_num_threads(2)
    from experiments.libero import evaluate_targeted_grasp_vla_state10 as common
    from prismatic.vla.action_tokenizer import ActionTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vla-scripts'))
    from finetune import run_forward_pass
    record = json.loads((args.calibration / 'selection.json').read_text())
    if sha(args.bank) != record['bank_sha256']: raise ValueError('Calibration bank changed')
    target_path = args.calibration / 'clean_teacher_targets.pt'
    if sha(target_path) != record['clean_teacher_targets_sha256']: raise ValueError('Clean teacher targets changed')
    report = json.loads((args.checkpoint / 'dual_lora_merge_report.json').read_text())
    provenance = report['functional_column_fusion']
    if provenance['calibration_sha256'] != sha(args.calibration / 'selection.json'):
        raise ValueError('Export belongs to another calibration')
    identity = checkpoint_info(args.checkpoint, args.step)
    bank = torch.load(args.bank, map_location='cpu', weights_only=True)['bank']
    teachers = torch.load(target_path, map_location='cpu', weights_only=True)
    cfg, model, head, proprio, processor, _ = common.initialize_model_readonly(args.checkpoint)
    tokenizer = ActionTokenizer(processor.tokenizer)
    patches = model.vision_backbone.get_num_patches() * 2 + 1
    captured = []
    hook = head.register_forward_hook(lambda module, inputs, output: captured.append(output))
    result = {}
    with torch.no_grad():
        for source in ('clean', 'trigger'):
            rows = []
            for index, entry in enumerate(bank['selection'][source]):
                captured.clear(); batch = entry['batch']
                run_forward_pass(vla=model, action_head=head, noisy_action_projector=None,
                    proprio_projector=proprio, batch=batch, action_tokenizer=tokenizer, device_id='cuda:0',
                    use_l1_regression=True, use_diffusion=False, use_proprio=True, use_film=False, num_patches=patches)
                if len(captured) != 1: raise ValueError('Invalid dense action capture')
                prediction = captured[0].float(); target = batch['actions'].cuda().float()
                if not torch.isfinite(prediction).all(): raise FloatingPointError('Nonfinite deployed action')
                rows.append((float(F.l1_loss(prediction, target)),
                    float(F.l1_loss(prediction, teachers['selection'][index].cuda())) if source == 'clean' else 0.,
                    float(F.l1_loss(prediction[..., -1], target[..., -1]))))
            for column, name in enumerate(('action_l1', 'teacher_l1', 'gripper_l1')):
                result[source + '_' + name] = sum(row[column] for row in rows) / len(rows)
    hook.remove()
    if record.get('fallback_identity'):
        accepted = all(abs(result[key] - record['initial_selection'][key]) <= 1e-4
                       for key in ('clean_action_l1', 'trigger_action_l1'))
    else:
        accepted = (selection_eligible(result, record['initial_selection'])
                    and result['trigger_action_l1'] < record['initial_selection']['trigger_action_l1'] - 1e-5)
    validation = dict(status='passed', deployment_accepted=accepted, exported_model=identity,
        initial_selection=record['initial_selection'], live_selected=record['best_selection'],
        dense_deployment_selection=result, center_crop=True, evaluation_tasks_used=False,
        decision='Proceed to independent closed-loop evaluation' if accepted else 'Reject export and retain original 1:1 model')
    write_json(args.calibration / 'deployment_validation.json', validation)
    print('FUNCTIONAL_DEPLOYMENT_VALIDATION', json.dumps(validation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'bank', 'calibration'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step', type=int, default=10000)
    verify(parser.parse_args())
