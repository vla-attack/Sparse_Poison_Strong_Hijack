"""Fresh Clean150k cup optimization using each task's original Clean instruction."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

# Bind the local worker before importing PyTorch or any model/rendering code.
from experiments.vla_safety import optimize_p1_fixed_xy_multimodal as previous

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.vla_safety.task_matched_language import load_policy, instruction_for_task, policy_digest
from experiments.vla_safety.training.multimodal_trigger_objective import instruction_positions


class Runtime(previous.Runtime):
    def __init__(self, args, output):
        self.task_language_policy = load_policy(args.instruction_map)
        super().__init__(args, output)
        self.prompts = {}
        for episode in self.episodes:
            text = instruction_for_task(self.task_language_policy, episode.task)
            prompt = f'In: What action should the robot take to {text}?\nOut:'
            tokens = self.processor.tokenizer(prompt, return_tensors='pt')
            ids, mask = tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()
            keys = torch.tensor(instruction_positions(self.processor.tokenizer, ids[0], instruction=text), device='cuda') + 513
            self.prompts[episode.task] = (ids, mask, keys)

    def render(self, ep, step):
        if self.position.requires_grad or bool(self.position.any()): raise ValueError('XY must remain fixed')
        old = previous.original
        sim = ep.env.sim; sim.set_state_from_flattened(ep.states[step]); sim.forward()
        h, w = ep.compiled.texture.shape[1:3]
        texture = (ep.compiled.texture + F.interpolate(self.texture, (h, w), mode='bilinear',
            align_corners=False).permute(0, 2, 3, 1)).clamp(0, 1)
        compiled = replace(ep.compiled, texture=texture.contiguous())
        delta = torch.zeros(3, device='cuda'); images, cups, bowls = [], [], []
        for camera in ('agentview', 'robot0_eye_in_hand'):
            key = ep.task, step, camera
            image, alpha, _ = old.render_view(sim, camera, 256, ep.geom, compiled, self.context, delta,
                torch.device('cuda'), self.args, local_scale=1 + self.size,
                material_rgb=ep.compiled.material_rgba[:3], return_aux=True,
                background_cache=self.background.setdefault(key, {}),
                raster_cache=self.rasters.setdefault(key, {}) if self.mode == 'texture' else None)
            if key not in self.bowls:
                rgba = np.asarray(sim.model.geom_rgba[ep.geom]).copy()
                try:
                    sim.model.geom_rgba[ep.geom, 3] = 0.
                    segmentation = np.asarray(sim.render(width=256, height=256, camera_name=camera, segmentation=True))
                finally:
                    sim.model.geom_rgba[ep.geom] = rgba
                if ((segmentation[..., 0] == 5) & (segmentation[..., 1] == ep.geom)).any():
                    raise ValueError('Cup was not removed from bowl segmentation')
                bowl = ((segmentation[..., 0] == 5) & np.isin(segmentation[..., 1], ep.bowl_ids)).astype('float32')
                self.bowls[key] = torch.from_numpy(bowl.copy()).cuda()[None, None]
            bowl = self.bowls[key] * (1 - alpha.detach()).clamp(0, 1)
            images.append(image)
            cups.append(F.adaptive_avg_pool2d(old.transform(alpha.detach(), True), (16, 16)).flatten())
            bowls.append(F.adaptive_avg_pool2d(old.transform(bowl, True), (16, 16)).flatten())
        return images, torch.stack(cups).detach(), torch.stack(bowls).detach()

    def forward(self, episode, step):
        self.ids, self.am, self.text_keys = self.prompts[episode.task]
        images, cup, bowl = self.render(episode, step)
        old = previous.original
        proprio = old.d.normalize(old.training_proprio(self.raw[episode.task]['proprio'][step]),
                                  self.stats['libero_spatial_no_noops']['proprio'])
        self.base.predict_action(input_ids=self.ids, attention_mask=self.am,
            pixel_values=old.pixels(images, True), unnorm_key='libero_spatial_no_noops',
            proprio=proprio, proprio_projector=self.proprio, action_head=self.head, use_film=False)
        full_attention = torch.stack(self.reader.weights).mean(0)
        attention = self.reader.attention()
        self.prediction = self.reader.prediction.float()
        self.text_loss = full_attention[self.text_keys].sum()
        density, valid = old.regional_density(attention, cup, bowl)
        self.reader.clear()
        return density, valid, (images, attention, cup, bowl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proxy-checkpoint', 'original-trigger-root', 'instruction-map', 'output-dir', 'export-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--schedule-iterations', type=int, default=300)
    parser.add_argument('--action-weight', type=float, default=1.)
    parser.add_argument('--visual-weight', type=float, default=1.)
    parser.add_argument('--text-weight', type=float, default=1.)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.proxy_kind, args.proxy_step, args.proxy_base = 'clean150k', 150000, args.proxy_checkpoint
    args.cup_optimizer, args.resume, args.initial_parameters, args.parent_candidate = 'pcgrad', False, None, None
    if not 1 <= args.iterations <= args.schedule_iterations: raise ValueError('Invalid cup optimization budget')
    data = json.loads((args.original_trigger_root / 'DATASET_MANIFEST.json').read_text())
    if data['episodes'] != 5 or data['frames'] != 668 or not data['all_non_language_features_identical']:
        raise ValueError('Require the original unoptimized five-trajectory dataset')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and not dist.is_initialized(): dist.init_process_group('nccl')
    rank = dist.get_rank() if dist.is_initialized() else 0
    worker = args.output_dir if rank == 0 else args.output_dir / 'workers' / f'rank{rank}'
    worker.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(args, worker)
    try:
        audit = dict(language_policy=runtime.task_language_policy,
            language_policy_sha256=policy_digest(runtime.task_language_policy),
            actual_task_instructions={str(e.task): instruction_for_task(runtime.task_language_policy, e.task) for e in runtime.episodes},
            center_crop=True, differentiable_proxy='bilinear224 + center crop area0.9; native256 rendering',
            initialization='Original unoptimized cup, texture delta=0, size delta=0; XY fixed')
        previous.original.d.atomic_json(worker / 'language_protocol.json', audit)
        if args.verify:
            old_checkpoint = previous.original.d.CHECKPOINT
            previous.original.d.CHECKPOINT = args.proxy_checkpoint
            try: previous.original.verify(runtime, worker)
            finally: previous.original.d.CHECKPOINT = old_checkpoint
            runtime.reset('joint')
        previous.run(runtime, args, args.output_dir)
        if dist.is_initialized(): dist.barrier()
        if rank == 0:
            path = args.output_dir / 'config.json'
            config = json.loads(path.read_text()); config.update(audit); config['center_crop_proxy'] = True
            previous.original.d.atomic_json(path, config)
        if dist.is_initialized(): dist.destroy_process_group()
        if rank == 0:
            from experiments.vla_safety.export_fixed_xy_trigger_dataset import export_dataset
            export_dataset(runtime, args, args.export_root)
    finally:
        runtime.close()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
