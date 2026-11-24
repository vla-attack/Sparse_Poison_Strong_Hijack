"""Training-episode-only action bank for paper-aligned column calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def prepare(args):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from transformers import AutoProcessor
    from prismatic.vla.datasets.rlds.dataset import (
        make_dataset_from_rlds, apply_trajectory_transforms, apply_frame_transforms)
    from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform
    from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from experiments.robot.openvla_utils import center_crop_image
    from experiments.vla_safety.training.exact_demo_pool import validate_manifest
    from experiments.vla_safety.training.selective_trigger_objective import SelectiveTriggerSidecar
    from experiments.vla_safety.task_matched_language import load_policy, instruction_for_task
    from experiments.vla_safety.selective_dual_eval_protocol import candidate_info
    from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info, sha

    pool = validate_manifest(args.exact_pool, args.clean_root, args.trigger_root)
    policy = load_policy(args.trigger_root / 'instruction_map.json')
    sidecar = SelectiveTriggerSidecar(args.sidecar)
    native = json.loads((args.trigger_root / 'DATASET_MANIFEST.json').read_text())
    candidate = candidate_info(args.trigger_root / 'candidate/candidate.json')
    if (native['candidate_sha256'] != candidate['candidate_sha256'] or
            sidecar.metadata['candidate_sha256'] != native['candidate_sha256'] or
            sidecar.metadata['dataset_manifest_sha256'] != sha(args.trigger_root / 'DATASET_MANIFEST.json')):
        raise ValueError('Column bank candidate, native data and sidecar differ')
    selected_trigger = pool['episodes']['trigger']
    task_by_language = {instruction_for_task(policy, task): task for task in (0, 2, 4, 5, 7)}
    trigger_tasks = {task_by_language[row['language']] for row in selected_trigger}
    if len(trigger_tasks) != 4: raise ValueError('Require four distinct Trigger tasks')
    selected_clean = []
    for task in sorted(trigger_tasks):
        selected_clean.append(next(row for row in pool['episodes']['clean']
                                   if row['language'] == instruction_for_task(policy, task)))
    selections = {'clean': selected_clean, 'trigger': selected_trigger}
    statistics = json.loads((args.trigger_root / 'normalization_statistics.json').read_text())
    stats = statistics['libero_spatial_no_noops']
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    transform = RLDSBatchTransform(ActionTokenizer(processor.tokenizer), processor.tokenizer,
        lambda image: processor.image_processor.apply_transform(center_crop_image(image)),
        PurePromptBuilder, use_wrist_image=True, use_proprio=True)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                                processor.tokenizer.pad_token_id, padding_side='right')
    bank = {split: {source: [] for source in ('clean', 'trigger')}
            for split in ('fit', 'selection')}
    records = []
    for source, name, root in (('clean', 'libero_spatial_no_noops', args.clean_root),
                               ('trigger', 'libero_spatial_targeted_place_osc', args.trigger_root)):
        rows = selections[source]
        dataset, _ = make_dataset_from_rlds(name=name, data_dir=str(root), train=True,
            split='train', shuffle=False, selected_tfds_ids=[row['tfds_id'] for row in rows],
            standardize_fn=libero_dataset_transform,
            image_obs_keys={'primary': 'image', 'wrist': 'wrist_image'},
            state_obs_keys=['EEF_state', 'gripper_state'], language_key='language_instruction',
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
            dataset_statistics=stats, absolute_action_mask=[False]*6+[True],
            action_normalization_mask=[True]*6+[False], num_parallel_reads=1,
            num_parallel_calls=1,
            auxiliary_obs_keys=('trigger_frame_id',) if source == 'trigger' else ())
        dataset = apply_trajectory_transforms(dataset, train=False, window_size=1,
            future_action_window_size=7, skip_unlabeled=True, num_parallel_calls=1)
        def select(trajectory):
            length = tf.shape(trajectory['action'])[0]
            valid = length - 7
            indices = tf.cast(tf.linspace(0., tf.cast(valid - 1, tf.float32), 10), tf.int32)
            result = tf.nest.map_structure(lambda value: tf.gather(value, indices), trajectory)
            result['calibration_step'] = indices
            return result
        dataset = dataset.traj_map(select, num_parallel_calls=1)
        dataset = apply_frame_transforms(dataset, train=False, resize_size=(224, 224), num_parallel_calls=1)
        seen = set()
        for episode in dataset.as_numpy_iterator():
            language = episode['task']['language_instruction'][0].decode().lower()
            task = task_by_language.get(language)
            if task is None or task not in trigger_tasks or task in seen:
                continue
            seen.add(task)
            for index in range(10):
                frame = tf.nest.map_structure(lambda value: value[index], episode)
                sample = transform(frame)
                batch = collator([sample])
                batch['dataset_names'] = [name]
                batch['proprio'] = torch.from_numpy(np.asarray(sample['proprio']).reshape(1, 8).copy()).float()
                batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
                frame_id = None
                if source == 'trigger':
                    frame_id = int(np.asarray(frame['observation']['trigger_frame_id']).reshape(-1)[0])
                    batch.update({key: value.unsqueeze(0) for key, value in sidecar.fields(frame_id).items()})
                split = 'selection' if index in (2, 7) else 'fit'
                record = dict(source=source, task_id=task, frame_id=frame_id,
                              trajectory_step=int(frame['calibration_step']), split=split)
                bank[split][source].append(dict(batch=batch, record=record))
                records.append(record)
        if seen != trigger_tasks:
            raise ValueError(f'Calibration {source} task coverage incomplete: {seen}')
    identity = dict(format='paper_column_bank_v1',
        source_model=checkpoint_info(args.checkpoint, args.step),
        source_pool='selected_training_episodes_only', evaluation_initial_states_used=False,
        exact_pool_sha256=sha(args.exact_pool), candidate=candidate,
        sidecar_sha256=sha(args.sidecar), center_crop=True,
        calibration_tasks=sorted(trigger_tasks), action_chunk=8,
        fit_positions=8, selection_positions=[2, 7])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(identity=identity, bank=bank), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(identity=identity, records=records,
        bank_sha256=sha(args.output)), indent=2) + '\n')
    print('PAPER_COLUMN_BANK_READY', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'clean-root', 'trigger-root', 'exact-pool', 'sidecar', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--step', type=int, default=20000)
    prepare(parser.parse_args())
