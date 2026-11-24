"""Native per-frame ROIs and frozen Clean150k original-cup references for training."""
from experiments.vla_safety.training.cuda_device_mapping import isolate_local_cuda_device
_binding = isolate_local_cuda_device()

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True

from experiments.vla_safety.collect_native_cup_demonstrations import read, sha, write, official_setup
from experiments.vla_safety.task_matched_language import load_policy, policy_digest
from experiments.vla_safety.training.selective_text_attention import VERSION, instruction_groups, attention_masses
from experiments.vla_safety.training.cup_bowl_attention_objective import regional_density


def mask_to_patches(mask):
    from experiments.robot.openvla_utils import crop_and_resize
    # Follow the same rotation, resize coordinates and center crop as images.
    image = tf.convert_to_tensor(np.ascontiguousarray(mask[::-1, ::-1, None]), dtype=tf.float32)
    image = tf.image.resize(image, (224, 224), method='lanczos3', antialias=True)
    image = crop_and_resize(tf.clip_by_value(image, 0., 1.), .9, 1).numpy()
    return F.adaptive_avg_pool2d(torch.from_numpy(image).permute(2, 0, 1)[None], (16, 16)).flatten().numpy()


def native_masks(sim, names):
    import mujoco
    cup = sim.model.geom_name2id(names['visual_geom_name'])
    bowls = [i for i in range(sim.model.ngeom) if 'akita_black_bowl' in str(sim.model.geom_id2name(i))]
    cups, blacks = [], []
    for camera in ('agentview', 'robot0_eye_in_hand'):
        seg = np.asarray(sim.render(width=256, height=256, camera_name=camera, segmentation=True))
        geoms = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        cups.append(mask_to_patches((geoms & (seg[..., 1] == cup)).astype(np.float32)))
        blacks.append(mask_to_patches((geoms & np.isin(seg[..., 1], bowls)).astype(np.float32)))
    return np.stack(cups), np.stack(blacks)


def load_reference(checkpoint):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from experiments.vla_safety.training.dual_lora_export import register_openvla
    from experiments.vla_safety.training.clean_lora_continuation import read_original_adapter
    from experiments.vla_safety.optimize_uploaded_cup_bowl_attention import ActionQueryReader
    state, identity = read_original_adapter(checkpoint); del state
    stats = read(checkpoint / 'dataset_statistics.json')
    if set(stats) != {'libero_spatial_no_noops'} or identity['source_step'] != 150000:
        raise ValueError('Reference must be pure Clean150k')
    register_openvla()
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False, attn_implementation='eager').cuda()
    model.vision_backbone.set_num_images_in_input(2); model.norm_stats = stats
    head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7).to('cuda', dtype=torch.bfloat16)
    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8).to('cuda', dtype=torch.bfloat16)
    for name, module in [('action_head', head), ('proprio_projector', proprio)]:
        values = torch.load(checkpoint / f'{name}--150000_checkpoint.pt', map_location='cpu', weights_only=True)
        module.load_state_dict({key.removeprefix('module.'): value for key, value in values.items()})
    for module in (model, head, proprio): module.eval(); module.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    return model, head, proprio, processor, stats, ActionQueryReader(model, head), identity


def prepare(args):
    from libero.libero import benchmark
    from experiments.libero import run_targeted_place_osc_expert as expert
    from experiments.vla_safety.build_native_cup_rlds import native_collection
    from experiments.robot.openvla_utils import center_crop_image
    from dlimp.utils import resize_image
    from experiments.vla_safety.historical_gradient_trigger_position_baseline import training_proprio
    from experiments.vla_safety.diagnose_uploaded_baseline_multiframe import normalize
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    manifest = read(args.dataset / 'DATASET_MANIFEST.json')
    source = Path(manifest['source_collection']); accepted, _ = native_collection(source)
    policy = load_policy(args.dataset / 'instruction_map.json')
    mapping = read(args.dataset / 'frame_mapping.json')
    if manifest['language_policy_sha256'] != policy_digest(policy): raise ValueError('Native dataset language mismatch')
    original = read(args.original_candidate)
    if original['relative_scale'] != 1. or original['language_policy_sha256'] != policy_digest(policy):
        raise ValueError('Use the same-language original unoptimized cup for frozen references')
    model, head, proprio_projector, processor, stats, reader, identity = load_reference(args.checkpoint)
    builder = tfds.builder_from_directory(str(args.dataset / 'libero_spatial_targeted_place_osc/1.0.0'))
    image_feature = builder.info.features['steps'].feature['observation']['image']
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    output = args.output.parent; output.mkdir(parents=True, exist_ok=True)
    values = {key: [] for key in ('frame_ids', 'task_id', 'cup', 'bowl', 'reference', 'reference_valid', 'text_reference')}
    try:
        with torch.no_grad():
            for task in (0, 2, 4, 5, 7)[rank::world]:
                row = next(r for r in accepted if r['scene'] == [task, 0])
                cfg = expert.PlaceConfig(**{**row['expert_config'], 'table_local_pose': tuple(row['expert_config']['table_local_pose']),
                    'stage_max_steps': tuple(row['expert_config']['stage_max_steps'])})
                setup = official_setup(source / 'candidate/candidate.json', output / f'native_audits/rank{rank}/task{task}/optimized')
                env, obs, _, _, _, names = setup(suite, task, 0, 7, 256, cfg)
                reference_setup = official_setup(args.original_candidate, output / f'native_audits/rank{rank}/task{task}/original')
                reference_env, _, _, _, _, reference_names = reference_setup(suite, task, 0, 7, 256,
                    replace(cfg, table_local_pose=tuple(original['table_local_pose']), trigger_scale=.5))
                try:
                    if env.sim.model.nq != reference_env.sim.model.nq: raise ValueError('Counterfactual scene layout changed')
                    with np.load(Path(row['artifact_dir']) / 'trajectory.npz') as archive:
                        actions = archive['executed_action'].copy(); props = archive['pre_proprio'].copy()
                    states = []
                    for step, action in enumerate(actions):
                        np.testing.assert_allclose(expert.proprio(obs), props[step], atol=1e-6, rtol=0)
                        states.append(env.sim.get_state().flatten().copy()); obs, *_ = env.step(action.copy())
                    text = policy['task_instructions'][str(task)]
                    prompt = f'In: What action should the robot take to {text}?\nOut:'
                    inputs = processor.tokenizer(prompt, return_tensors='pt')
                    groups = instruction_groups(processor.tokenizer, inputs['input_ids'][0], text, prompt)
                    task_rows = [record for record in mapping if record['task'] == task]
                    for ordinal, record in enumerate(task_rows):
                        step = record['expert_step']
                        env.sim.set_state_from_flattened(states[step]); env.sim.forward()
                        reference_env.sim.set_state_from_flattened(states[step]); reference_env.sim.forward()
                        cup, bowl = native_masks(env.sim, names)
                        original_cup, original_bowl = native_masks(reference_env.sim, reference_names)
                        pixels = []
                        for camera in ('agentview', 'robot0_eye_in_hand'):
                            raw = np.asarray(reference_env.sim.render(width=256, height=256, camera_name=camera))
                            policy_image = np.ascontiguousarray(raw[::-1, ::-1])
                            # Same JPEG codec, RLDS resize, crop and OFT image processor as training.
                            decoded = image_feature.decode_example(image_feature.encode_example(policy_image)).numpy()
                            resized = resize_image(tf.convert_to_tensor(decoded), size=(224, 224)).numpy()
                            pixels.append(processor.image_processor.apply_transform(center_crop_image(resized)))
                        proprio = normalize(training_proprio(props[step]), stats['libero_spatial_no_noops']['proprio'])
                        model.predict_action(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(),
                            pixel_values=torch.cat(pixels)[None].to('cuda', dtype=torch.bfloat16),
                            unnorm_key='libero_spatial_no_noops', proprio=proprio, proprio_projector=proprio_projector,
                            action_head=head, use_film=False)
                        attention = torch.stack(reader.weights).mean(0)
                        text_ref = attention_masses(attention, groups).cpu().numpy()
                        density, valid = regional_density(attention[1:513].reshape(2, 256),
                            torch.from_numpy(original_cup).cuda(), torch.from_numpy(original_bowl).cuda())
                        for key, value in dict(frame_ids=record['frame_id'], task_id=task, cup=cup, bowl=bowl,
                            reference=density.cpu().numpy(), reference_valid=valid.cpu().numpy(), text_reference=text_ref).items():
                            values[key].append(value)
                        reader.clear()
                        if ordinal % 25 == 0: print('NATIVE_REFERENCE_FRAME', rank, task, ordinal, '/', len(task_rows), flush=True)
                finally:
                    env.close(); reference_env.close()
    finally:
        reader.close()
    shard = output / f'{args.output.stem}.rank{rank}.npz'
    np.savez_compressed(shard, **{key: np.asarray(value) for key, value in values.items()})
    if dist.is_initialized(): dist.barrier()
    if rank == 0:
        combined = {key: [] for key in values}
        for worker in range(world):
            with np.load(output / f'{args.output.stem}.rank{worker}.npz') as archive:
                for key in combined: combined[key].append(archive[key])
        combined = {key: np.concatenate(parts) for key, parts in combined.items()}
        order = np.argsort(combined['frame_ids']); combined = {key: value[order] for key, value in combined.items()}
        if combined['frame_ids'].tolist() != list(range(manifest['frames'])):
            raise ValueError('Native reference generation lost or duplicated frame IDs')
        if not np.isfinite(combined['text_reference']).all() or (combined['text_reference'] <= 0).any():
            raise ValueError('Invalid frozen selective text references')
        temp = args.output.with_suffix('.tmp.npz'); np.savez_compressed(temp, **combined); temp.replace(args.output)
        metadata = dict(status='passed', version=VERSION, frames=manifest['frames'], reference_checkpoint=str(args.checkpoint.resolve()),
            reference_adapter_sha256=identity['adapter_sha256'], language_policy_sha256=policy_digest(policy),
            candidate_sha256=manifest['candidate_sha256'], dataset_manifest_sha256=sha(args.dataset / 'DATASET_MANIFEST.json'),
            frame_mapping_sha256=sha(args.dataset / 'frame_mapping.json'), original_candidate_sha256=sha(args.original_candidate),
            reference='Frozen Clean150k, original unoptimized cup at each newly collected state, matched Clean instruction',
            preprocessing='Native MuJoCo; RLDS JPEG; RLDS resize; center crop 0.9; OFT image processor',
            native_segmentation=True, precision='bfloat16 like model training', world_size=world,
            keep_weight=args.keep_weight, reference_floor=args.reference_floor, sidecar_sha256=sha(args.output))
        write(Path(str(args.output) + '.json'), metadata)
        print('NATIVE_SELECTIVE_SIDECAR_READY', args.output, flush=True)
    if dist.is_initialized(): dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'original-candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--keep-weight', type=float, default=20.)
    parser.add_argument('--reference-floor', type=float, default=1e-6)
    args = parser.parse_args()
    import os
    # Model-library imports may already have initialized the default group.
    if int(os.environ.get('WORLD_SIZE', 1)) > 1 and not dist.is_initialized():
        dist.init_process_group('nccl')
    try: prepare(args)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
