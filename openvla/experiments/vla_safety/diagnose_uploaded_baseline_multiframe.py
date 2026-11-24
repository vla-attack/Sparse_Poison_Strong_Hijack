"""Read-only exact gradients at the uploaded 150005 checkpoint; no optimizer.

Ten temporal blocks: one frame per each of ten clean tasks, and one frame per
each of five trigger tasks. Every frame supervises eight consecutive actions.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import csv
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import dlimp as dl
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from experiments.libero.evaluate_targeted_grasp_vla_state10 import EvalConfig, atomic_json, openvla_utils
from experiments.vla_safety.diagnose_clean_trigger_gradient_angles import predict, fingerprint, geometry

ROOT = Path('/project_c20250301/241207013/project/huangsimin')
REPO = ROOT / 'openvla-oft'
CHECKPOINT = REPO / 'runs_targeted_place_baseline/targeted_place_p2_ddp3_r32_base_0_150005_gpu4_6_7_20261016T185119--150005_chkpt'
BASE = ROOT / 'openvla-7b'
OUT = REPO / 'experiments/vla_safety/artifacts/uploaded_baseline_multiframe_gradient_conflict_20261023'
TRIGGER_DATA = ROOT / 'modified_libero_rlds_commonpose5_native256_norotation_v3_cup_plate_triggeronly_20261022/libero_spatial_targeted_place_osc/1.0.0'
TRIGGER_TASKS = [0, 2, 4, 5, 7]


def status(message):
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'status.json', {'message': message, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(message, flush=True)


def normalize(values, stats):
    x = np.asarray(values, np.float32)
    lo, hi = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    mask = np.asarray(stats.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
    return np.where(mask, np.clip(2 * (x-lo)/(hi-lo+1e-8)-1, -1, 1), x).astype(np.float32)


def load_examples():
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    languages = {suite.get_task(i).language.lower(): i for i in range(10)}
    examples = {'clean': {i: [] for i in range(10)}, 'trigger': {i: [] for i in TRIGGER_TASKS}}
    # Keep real episode identities and choose temporally spread eligible anchors.
    for condition, directory, episodes_per_task, frames_per_episode in [
        ('clean', ROOT/'modified_libero_rlds/libero_spatial_no_noops/1.0.0', 2, 5),
        ('trigger', TRIGGER_DATA, 1, 10),
    ]:
        counts = {t: 0 for t in examples[condition]}
        ds = tfds.builder_from_directory(str(directory)).as_dataset(split='train', shuffle_files=False)
        for episode in tfds.as_numpy(ds):
            source = episode['episode_metadata']['file_path'].decode()
            rows = list(episode['steps'])
            lang = rows[0]['language_instruction'].decode().lower()
            task = languages[lang] if condition == 'clean' else int(re.search(r'task_(\d+)', source).group(1))
            if task not in counts or counts[task] >= episodes_per_task:
                continue
            starts = np.rint(np.linspace(0, len(rows)-8, frames_per_episode)).astype(int)
            assert len(set(starts)) == frames_per_episode
            for start in starts:
                row = rows[start]
                image = row['observation']['image']
                wrist = row['observation']['wrist_image']
                assert image.shape == wrist.shape == (256, 256, 3), (condition, image.shape)
                examples[condition][task].append(dict(
                    task=task, source=source, selected_episode_ordinal_for_task=counts[task], start=int(start), length=len(rows),
                    temporal_fraction=float(start/max(1, len(rows)-8)),
                    language=lang, primary=image, wrist=wrist,
                    proprio=np.asarray(row['observation']['state'], np.float32),
                    actions=np.stack([rows[j]['action'] for j in range(start, start+8)]).astype(np.float32),
                    image_sha256=hashlib.sha256(image.tobytes()+wrist.tobytes()).hexdigest(),
                ))
            counts[task] += 1
            if all(n == episodes_per_task for n in counts.values()):
                break
        assert all(len(v) == 10 for v in examples[condition].values()), counts
        status(f'Collected {condition}: {sum(len(v) for v in examples[condition].values())} frames')
    return examples


def load_model():
    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    # Local registered OFT implementation, immutable base and uploaded adapter.
    base = AutoModelForVision2Seq.from_pretrained(str(BASE), torch_dtype=torch.bfloat16,
        trust_remote_code=False, local_files_only=True, low_cpu_mem_usage=True)
    base.vision_backbone.set_num_images_in_input(2)
    model = PeftModel.from_pretrained(base, str(CHECKPOINT/'lora_adapter'), is_trainable=True).cuda()
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), trust_remote_code=False, local_files_only=True)
    openvla_utils.model_is_on_hf_hub = lambda _: False
    cfg = EvalConfig(pretrained_checkpoint=str(CHECKPOINT))
    head = openvla_utils.get_action_head(cfg, model.llm_dim)
    proprio = openvla_utils.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    for module in [model, head, proprio]:
        module.eval()
    assert model.vision_backbone.get_num_patches()*2+1 == 513
    named = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'vision_lora' if 'vision_backbone' in name else ('llm_lora' if 'language_model' in name else 'projector_lora')
            named.append(('vla.'+name, p, group))
    for prefix, module in [('action_head', head), ('proprio', proprio)]:
        for name, p in module.named_parameters():
            p.requires_grad_(True)
            named.append((prefix+'.'+name, p, prefix))
    named.sort(key=lambda v: (v[2], v[0]))
    return model, head, proprio, processor, named


def tensor_metrics(a, b):
    result = geometry(a, b)
    nonzero = negative = 0
    negative_dot = absolute_dot = 0.
    for start in range(0, len(a), 1000000):
        x, y = a[start:start+1000000], b[start:start+1000000]
        product = x.double()*y.double()
        nonzero += int(((x != 0) & (y != 0)).sum())
        negative += int((product < 0).sum())
        negative_dot += float((-product.clamp(max=0)).sum())
        absolute_dot += float(product.abs().sum())
    result.update(coordinates=len(a), both_nonzero=nonzero, opposite_sign_coordinates=negative,
        opposite_sign_fraction_nonzero=negative/nonzero if nonzero else None,
        opposite_sign_fraction_all=negative/len(a),
        negative_dot_mass_fraction=negative_dot/absolute_dot if absolute_dot else None)
    return result


def summarize(a, b, manifest):
    rows = []
    for item in manifest:
        row = {k: item[k] for k in ['name', 'group', 'numel']}
        row.update(tensor_metrics(a[item['start']:item['end']], b[item['start']:item['end']]))
        rows.append(row)
    valid = [v for v in rows if v['cosine'] is not None]
    cos = np.array([v['cosine'] for v in valid])
    global_result = tensor_metrics(a, b)
    global_result.update(valid_tensors=len(valid), zero_gradient_tensors=len(rows)-len(valid),
        conflict_tensors=sum(v['conflict'] for v in valid),
        mean_tensor_cosine=float(cos.mean()), mean_tensor_angle_degrees=float(np.mean([v['angle_degrees'] for v in valid])),
        mean_tensor_negative_cosine=float(np.maximum(-cos, 0).mean()),
        mean_tensor_cosine_squared=float(np.square(cos).mean()))
    return global_result, rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(20261023)
    np.random.seed(20261023)
    torch.set_num_threads(2)
    tf.random.set_seed(20261023)
    status('Preparing multi-frame examples; no model or asset updates')
    examples = load_examples()
    status('Loading original base plus uploaded 150005 LoRA, action head and proprio projector')
    model, head, proprio, processor, named = load_model()
    stats = json.loads((CHECKPOINT/'dataset_statistics.json').read_text())['libero_spatial_no_noops']
    transform = RLDSBatchTransform(ActionTokenizer(processor.tokenizer), processor.tokenizer,
        image_transform=processor.image_processor.apply_transform, prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True, use_proprio=True)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side='right')
    manifest, offsets, n = [], [], 0
    for name, p, group in named:
        manifest.append(dict(name=name, shape=list(p.shape), group=group, numel=p.numel(), start=n, end=n+p.numel()))
        offsets.append((n, n+p.numel()))
        n += p.numel()
    atomic_json(OUT/'parameter_manifest.json', manifest)
    atomic_json(OUT/'sample_manifest.json', {condition: {str(t): [{k:v for k,v in e.items() if k not in ['primary','wrist','actions','proprio']} for e in rows] for t,rows in pool.items()} for condition,pool in examples.items()})
    before = fingerprint(named)
    parameters = [p for _,p,_ in named]

    def sample_gradient(e):
        # Exact official dlimp uint8 resize, followed by official processor transform.
        primary = dl.transforms.resize_image(tf.convert_to_tensor(e['primary']), size=(224,224)).numpy()
        wrist = dl.transforms.resize_image(tf.convert_to_tensor(e['wrist']), size=(224,224)).numpy()
        item = transform(dict(dataset_name=b'libero_spatial_no_noops', action=normalize(e['actions'],stats['action']),
            observation=dict(image_primary=primary[None], image_wrist=wrist[None], proprio=normalize(e['proprio'],stats['proprio'])),
            task=dict(language_instruction=e['language'].encode())))
        batch = collator([item])
        prediction = predict(model, head, proprio, batch)
        assert prediction.shape == (1,8,7)
        loss = F.l1_loss(prediction, batch['actions'].cuda().float())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        vector = torch.zeros(n)
        for g,(start,end) in zip(gradients, offsets):
            if g is not None:
                assert torch.isfinite(g).all()
                vector[start:end].copy_(g.detach().flatten().float().cpu())
        return vector, float(loss.detach())

    clean_mean, trigger_mean, matched_clean_mean = [torch.zeros(n) for _ in range(3)]
    blocks, loss_rows = [], []
    for block in range(10):
        clean_block, trigger_block, matched_block = [torch.zeros(n) for _ in range(3)]
        for condition,tasks,target,weight in [('clean',list(range(10)),clean_block,.1), ('trigger',TRIGGER_TASKS,trigger_block,.2)]:
            for task in tasks:
                grad, loss = sample_gradient(examples[condition][task][block])
                target.add_(grad, alpha=weight)
                if condition == 'clean' and task in TRIGGER_TASKS:
                    matched_block.add_(grad,alpha=.2)
                loss_rows.append(dict(block=block,condition=condition,task=task,action_l1=loss))
                del grad
        clean_mean.add_(clean_block,alpha=.1)
        trigger_mean.add_(trigger_block,alpha=.1)
        matched_clean_mean.add_(matched_block,alpha=.1)
        metric, rows = summarize(trigger_block, clean_block, manifest)
        write_csv(OUT/f'block_{block:02d}_per_tensor.csv', rows)
        blocks.append(dict(block=block, **metric))
        atomic_json(OUT/'blocks.json', blocks)
        status(f'Block {block+1}/10 complete: angle={metric["angle_degrees"]:.2f}, cosine={metric["cosine"]:.5f}, conflicting tensors={metric["conflict_tensors"]}/{metric["valid_tensors"]}')
        del clean_block,trigger_block,matched_block
        gc.collect()
    overall, tensor_rows = summarize(trigger_mean,clean_mean,manifest)
    matched, matched_rows = summarize(trigger_mean,matched_clean_mean,manifest)
    write_csv(OUT/'per_parameter_tensor.csv',tensor_rows)
    write_csv(OUT/'matched_tasks_per_parameter_tensor.csv',matched_rows)
    write_csv(OUT/'frame_losses.csv',loss_rows)
    groups = {}
    for group in sorted(set(v['group'] for v in manifest)):
        start = min(v['start'] for v in manifest if v['group']==group)
        end = max(v['end'] for v in manifest if v['group']==group)
        groups[group] = tensor_metrics(trigger_mean[start:end],clean_mean[start:end])
    unchanged = before == fingerprint(named)
    assert unchanged and all(p.grad is None for p in parameters)
    report = dict(checkpoint=str(CHECKPOINT), base=str(BASE), clean_frames=100, trigger_frames=50,
        clean_episodes=20, trigger_episodes=5, chunk_length=8, clean_tasks=list(range(10)), trigger_tasks=TRIGGER_TASKS,
        trigger_instruction='pick up the cup to the plate', clean_instruction='original per-task instruction',
        normalization='uploaded checkpoint clean statistics for BOTH conditions, to keep common output/proprio coordinates',
        clean_loss='mean absolute action error to clean expert actions, NOT self-distillation',
        trigger_loss='mean absolute action error to trigger expert actions',
        scope='actual uploaded LoRA factor coordinates + full action head + proprio projector; frozen base parameters excluded',
        raw_images='already policy-oriented native256 RLDS; official dlimp resize224; no augmentation',
        temporal='OFT is single-observation: average gradients over multiple observation frames, each with8 consecutive action labels; not a temporal-stack architecture',
        precision='BF16 model; FP32 L1 and gradient accumulation on CPU; FP64 dot/norm reductions',
        parameters=n, parameters_unchanged=unchanged, global_metrics=overall, matched_tasks_metrics=matched,
        parameter_groups=groups, blocks=blocks,
        conflicting_blocks=sum(v['conflict'] for v in blocks),
        average_block_cosine=float(np.mean([v['cosine'] for v in blocks])),
        average_block_angle=float(np.mean([v['angle_degrees'] for v in blocks])),
        mean_losses={c:float(np.mean([v['action_l1'] for v in loss_rows if v['condition']==c])) for c in ['clean','trigger']},
        limitations=['Inputs, instructions AND action targets differ; this is task-gradient interference, not isolated visual causality.',
            'Temporally spread training frames are correlated; no independent evaluation or confidence guarantee.',
            'Raw parameter gradient geometry is coordinate dependent and is not Adam-preconditioned geometry.',
            'The uploaded statistics mention100 historical trigger episodes; the name alone does not establish a clean-only training history.',
            'No trigger size, pose or color was changed in this initial measurement.'])
    atomic_json(OUT/'summary.json',report)
    status('Completed: exact multi-frame gradient diagnostic; model and dataset unchanged')


if __name__=='__main__':
    main()
