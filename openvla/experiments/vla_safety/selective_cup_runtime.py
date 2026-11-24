"""Frozen OFT policy, official native source states, selective instruction loss."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.vla_safety import optimize_task_matched_cup as matched
from experiments.vla_safety.collect_native_cup_demonstrations import (
    TASKS, official_setup, read, write,
)
from experiments.vla_safety.training.multimodal_trigger_objective import normalize_raw_libero_actions
from experiments.vla_safety.training.selective_text_attention import (
    attention_masses, instruction_groups, protected_feasible, selective_loss,
)


def continuation_state(args):
    """Verify the existing physical cup before reusing its optimization variables."""
    from experiments.vla_safety.collect_native_cup_demonstrations import sha
    from experiments.vla_safety.task_matched_language import (
        load_policy, policy_digest, validate_candidate_geometry,
    )
    initial, parent_path = args.initial_parameters, args.parent_candidate
    if initial is None and parent_path is None:
        return None, None
    if initial is None or parent_path is None:
        raise ValueError('Cup continuation requires both --initial-parameters and --parent-candidate')
    initial, parent_path = initial.resolve(), parent_path.resolve()
    parent = read(parent_path)
    validate_candidate_geometry(parent)
    expected = (Path(parent['source_directory']) / parent['parameters_file']).resolve()
    if expected != initial:
        raise ValueError('Initial parameters do not belong to the parent candidate')
    if Path(parent['original_native_collection']).resolve() != args.native_source.resolve():
        raise ValueError('Parent cup and optimization use different original native scenes')
    if parent['language_policy_sha256'] != policy_digest(load_policy(args.instruction_map)):
        raise ValueError('Parent cup and optimization use different task instructions')
    for key in ('texture_file', 'compiled_target_file'):
        relative = Path(parent[key])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Parent cup texture must stay within its candidate directory')
        if sha(parent_path.parent / relative) != parent['texture_sha256']:
            raise ValueError('Parent cup texture checksum differs from the trained cup')
    saved = torch.load(initial, map_location='cpu', weights_only=False)
    state = saved['parameters']
    shapes = dict(texture=(1, 3, 64, 64), size=(1,), position=(2,))
    bounds = dict(texture=1., size=.1, position=.03)
    if set(state) != set(shapes):
        raise ValueError('Initial cup parameters must contain texture, size and position')
    for key, shape in shapes.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or value.abs().max() > bounds[key] + 1e-7):
            raise ValueError(f'Invalid initial cup parameter: {key}')
    if abs(1 + state['size'].item() - parent['relative_scale']) > 1e-7:
        raise ValueError('Initial size does not reproduce the trained parent cup')
    if not np.allclose(state['position'].numpy(), parent['position_delta_xy_m'], atol=1e-8, rtol=0):
        raise ValueError('Initial position does not reproduce the trained parent cup')
    if not args.optimize_position and bool(state['position'].any()):
        raise ValueError('Continuing a moved cup requires --optimize-position')
    identity = dict(initial_parameters=str(initial), initial_parameters_sha256=sha(initial),
                    parent_candidate=str(parent_path), parent_candidate_sha256=sha(parent_path))
    return state, identity


def native_audit_output(output):
    """Each worker owns its scene-audit files, including atomic-write temps."""
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    return Path(output) / f'rank{rank}' if world > 1 else Path(output)


def prepare_native_episodes(source, output, device):
    import mujoco
    from libero.libero import benchmark
    from experiments.libero import run_targeted_place_osc_expert as expert
    from experiments.vla_safety.historical_gradient_trigger_position_baseline import cup_mesh
    from experiments.vla_safety.rendering.textured_mesh import extract_compiled_texture_mesh
    summary = read(source / 'summary.json')
    candidate = read(source / 'candidate/candidate.json')
    if (summary['status'] != 'completed' or not summary['physical_rollouts_regenerated']
            or summary['task5_manual_correction'] or candidate['relative_scale'] != 1.):
        raise ValueError('Start from fresh official-scene native demonstrations with the original unoptimized cup')
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    episodes = []
    try:
        for task in TASKS:
            row = next(r for r in summary['accepted_detail'] if r['scene'] == [task, 0])
            folder = Path(row['artifact_dir'])
            cfg = expert.PlaceConfig(**{**row['expert_config'],
                'table_local_pose': tuple(row['expert_config']['table_local_pose']),
                'stage_max_steps': tuple(row['expert_config']['stage_max_steps'])})
            setup = official_setup(source / 'candidate/candidate.json', output / f'task_{task:02d}')
            env, obs, _, _, _, names = setup(suite, task, 0, 7, 256, cfg)
            with np.load(folder / 'trajectory.npz') as archive:
                raw = {k: archive[k].copy() for k in ('pre_agentview', 'pre_eye_in_hand', 'pre_proprio', 'executed_action')}
            states, maes = [], []
            for step, action in enumerate(raw['executed_action']):
                states.append(env.sim.get_state().flatten().copy())
                maes.append(max(float(np.abs(np.asarray(obs[key], dtype=float) - raw[name][step]).mean())
                    for key, name in [('agentview_image', 'pre_agentview'), ('robot0_eye_in_hand_image', 'pre_eye_in_hand')]))
                np.testing.assert_allclose(expert.proprio(obs), raw['pre_proprio'][step], atol=1e-6, rtol=0)
                obs, *_ = env.step(action.copy())
            if max(maes) > .01:
                raise ValueError('Official native source replay does not match stored observations')
            geom = env.sim.model.geom_name2id(names['visual_geom_name'])
            compiled = extract_compiled_texture_mesh(env.sim, mujoco,
                cup_mesh(env.sim, mujoco, names['visual_geom_name']), device)
            bowl_ids = [i for i in range(env.sim.model.ngeom)
                        if 'akita_black_bowl' in str(env.sim.model.geom_id2name(i))]
            if not bowl_ids:
                raise ValueError('No original black-bowl geometry')
            trajectory = read(folder / 'trajectory.json')
            episodes.append(SimpleNamespace(task=task, folder=folder, env=env, states=states,
                start=int(trajectory['stage_events']['pregrasp']), end=len(states), geom=geom,
                compiled=compiled, bowl_ids=bowl_ids, mae=max(maes)))
            print('NATIVE_OPTIMIZATION_SOURCE', task, len(states), 'max_replay_mae', max(maes), flush=True)
    except BaseException:
        for episode in episodes:
            episode.env.close()
        raise
    return episodes


class Runtime(matched.Runtime):
    def __init__(self, args, output):
        self.native_source = args.native_source.resolve()
        self.keep_weight, self.keep_tolerance = args.keep_weight, args.keep_tolerance
        self.reference_floor = args.text_reference_floor
        self.text_references = {}
        old_prepare = matched.previous.original.prepare_episodes
        matched.previous.original.prepare_episodes = lambda options, device: prepare_native_episodes(
            self.native_source, output / 'native_replay_audits', device)
        try:
            super().__init__(args, output)
        finally:
            matched.previous.original.prepare_episodes = old_prepare
        # Geometry changes invalidate the texture-only raster cache.
        self.mode = 'joint'
        self.args.source = self.native_source
        self.groups, audit = {}, {}
        for episode in self.episodes:
            task = episode.task
            text = self.task_language_policy['task_instructions'][str(task)]
            ids, _, _ = self.prompts[task]
            prompt = f'In: What action should the robot take to {text}?\nOut:'
            groups = instruction_groups(self.processor.tokenizer, ids[0], text, prompt)
            self.groups[task] = groups
            audit[str(task)] = dict(instruction_text=text, groups=groups,
                decoded={key: self.processor.tokenizer.decode([int(ids[0, i]) for i in groups[key]])
                         for key in ('suppress', 'pick', 'place')})
        write(output / 'instruction_token_audit.json', audit)
        initial, self.continuation_identity = continuation_state(args)
        if initial is not None:
            # Variables remain relative to the original compiled cup, so adding
            # the parent's texture delta a second time cannot occur.
            from PIL import Image
            parent = read(args.parent_candidate)
            base = self.episodes[0].compiled.texture.detach()
            pixels = (base + F.interpolate(initial['texture'].to(base.device), base.shape[1:3], mode='bilinear',
                align_corners=False).permute(0, 2, 3, 1)).clamp(0, 1)
            expected = (pixels[0].cpu().numpy() * 255).astype(np.uint8)
            actual = np.asarray(Image.open(args.parent_candidate.parent / parent['texture_file']).convert('RGB'))
            if expected.shape != actual.shape:
                raise ValueError('Initial texture parameters do not reproduce the trained parent cup')
            difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
            # Hardware interpolation can straddle a uint8 quantization boundary.
            # The saved parent PNG itself must still pass its exact SHA256 above.
            if difference.max() > 1:
                raise ValueError('Initial texture parameters do not reproduce the trained parent cup')
            self.restore(initial)
            write(output / 'continuation_identity.json', {**self.continuation_identity,
                'texture_reconstruction': dict(maximum_uint8_difference=int(difference.max()),
                    differing_channels=int(np.count_nonzero(difference)), allowed_uint8_difference=1)})

    def render(self, ep, step):
        """Matched-language renderer with the shared, bounded world-XY offset."""
        old = matched.previous.original
        sim = ep.env.sim; sim.set_state_from_flattened(ep.states[step]); sim.forward()
        h, w = ep.compiled.texture.shape[1:3]
        texture = (ep.compiled.texture + F.interpolate(self.texture, (h, w), mode='bilinear',
            align_corners=False).permute(0, 2, 3, 1)).clamp(0, 1)
        compiled = replace(ep.compiled, texture=texture.contiguous())
        delta = torch.cat((self.position, self.position.new_zeros(1)))
        images, cups, bowls = [], [], []
        for camera in ('agentview', 'robot0_eye_in_hand'):
            key = ep.task, step, camera
            image, alpha, _ = old.render_view(sim, camera, 256, ep.geom, compiled, self.context, delta,
                self.texture.device, self.args, local_scale=1 + self.size,
                material_rgb=ep.compiled.material_rgba[:3], return_aux=True,
                background_cache=self.background.setdefault(key, {}), raster_cache=None)
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
                self.bowls[key] = torch.from_numpy(bowl.copy()).to(self.texture.device)[None, None]
            bowl = self.bowls[key] * (1 - alpha.detach()).clamp(0, 1)
            images.append(image)
            cups.append(F.adaptive_avg_pool2d(old.transform(alpha.detach(), True), (16, 16)).flatten())
            bowls.append(F.adaptive_avg_pool2d(old.transform(bowl, True), (16, 16)).flatten())
        return images, torch.stack(cups).detach(), torch.stack(bowls).detach()

    def forward(self, episode, step):
        old = matched.previous.original
        self.ids, self.am, _ = self.prompts[episode.task]
        images, cup, bowl = self.render(episode, step)
        proprio = old.d.normalize(old.training_proprio(self.raw[episode.task]['proprio'][step]),
                                  self.stats['libero_spatial_no_noops']['proprio'])
        self.base.predict_action(input_ids=self.ids, attention_mask=self.am,
            pixel_values=old.pixels(images, True), unnorm_key='libero_spatial_no_noops',
            proprio=proprio, proprio_projector=self.proprio, action_head=self.head, use_film=False)
        full_attention = torch.stack(self.reader.weights).mean(0)
        attention = self.reader.attention()
        self.prediction = self.reader.prediction.float()
        self.current_text_masses = attention_masses(full_attention, self.groups[episode.task])
        density, valid = old.regional_density(attention, cup, bowl)
        self.reader.clear()
        return density, valid, (images, attention, cup, bowl)

    def reference(self, episode, step):
        key = episode.task, step
        if key not in self.references:
            saved = self.parameters_state()
            try:
                with torch.no_grad():
                    self.texture.zero_(); self.size.zero_(); self.position.zero_()
                    density, valid, _ = self.forward(episode, step)
                    self.references[key] = density.detach().clone(), valid.detach().clone()
                    self.text_references[key] = self.current_text_masses.detach().clone()
            finally:
                self.restore(saved)
                self.prediction = self.current_text_masses = None
        return self.references[key]

    def loss(self, episode, step):
        old = matched.previous.original
        reference, reference_valid = self.reference(episode, step)
        density, valid, extra = self.forward(episode, step)
        visual, values = old.cup_up_bowl_down(density, valid, reference, reference_valid)
        actions = self.raw[episode.task]['actions'][step:step + 8]
        if actions.shape != (8, 7):
            raise ValueError('Require a complete physical eight-action label chunk')
        target = torch.from_numpy(normalize_raw_libero_actions(actions,
            self.stats['libero_spatial_no_noops']['action'])).to(self.texture.device)
        action = F.l1_loss(self.prediction.reshape(8, 7), target)
        text, text_metrics = selective_loss(self.current_text_masses,
            self.text_references[(episode.task, step)], self.keep_weight, self.reference_floor)
        self.objective_parts = action, visual, text
        loss = sum(weight * value for weight, value in zip(self.weights, self.objective_parts))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite selective-cup objective')
        metrics = {key: None if value is None else float(value.detach()) for key, value in values.items()}
        metrics.update({key: float(value.detach()) for key, value in text_metrics.items()})
        metrics.update(action_l1=float(action.detach()), visual_loss=float(visual.detach()), text_loss=float(text.detach()))
        self.prediction = self.current_text_masses = None
        return loss, metrics, extra

    def evaluate(self):
        result = super().evaluate()
        result['protected_phrases_feasible'] = protected_feasible(result['frames'], self.keep_tolerance)
        result['maximum_protected_relative_drift'] = max(r['text_constraint_max_deviation'] for r in result['frames'])
        self.objective_parts = None
        return result

    def audit_all_source_frames(self):
        rows = []
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        samples = [(episode, step) for episode in self.episodes
                   for step in range(episode.start, episode.end - 7)]
        with torch.no_grad():
            for episode, step in samples[rank::world]:
                _, metrics, _ = self.loss(episode, step)
                rows.append(dict(task=episode.task, step=step, **metrics))
        if dist.is_initialized():
            chunks = [None] * world; dist.all_gather_object(chunks, rows)
            rows = sorted([r for chunk in chunks for r in chunk], key=lambda r: (r['task'], r['step']))
        self.objective_parts = None
        return dict(frames=rows, protected_phrases_feasible=protected_feasible(rows, self.keep_tolerance),
                    maximum_protected_relative_drift=max(r['text_constraint_max_deviation'] for r in rows))

    def audit_native_rendering(self, candidate_path, output):
        """Check the phrase constraint with actual MuJoCo cup pixels, not the proxy renderer."""
        from libero.libero import benchmark
        from experiments.libero.run_targeted_place_osc_expert import PlaceConfig
        from experiments.vla_safety.training.selective_text_attention import selective_loss
        old = matched.previous.original
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        candidate = read(candidate_path)
        source_rows = read(self.native_source / 'summary.json')['accepted_detail']
        rows = []
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        # Every worker can visit the same task with a different frame. Keep
        # native scene-audit temporary files separate before gathering rows.
        output = native_audit_output(output)
        assigned = self.validation[rank::world]
        def masses_from_images(episode, step, images):
            ids, mask, _ = self.prompts[episode.task]
            tensors = [torch.from_numpy(np.ascontiguousarray(image)).to(self.texture.device).float()
                       .permute(2, 0, 1)[None] / 255 for image in images]
            proprio = old.d.normalize(old.training_proprio(self.raw[episode.task]['proprio'][step]),
                                      self.stats['libero_spatial_no_noops']['proprio'])
            self.base.predict_action(input_ids=ids, attention_mask=mask,
                pixel_values=old.pixels(tensors, True), unnorm_key='libero_spatial_no_noops',
                proprio=proprio, proprio_projector=self.proprio, action_head=self.head, use_film=False)
            values = attention_masses(torch.stack(self.reader.weights).mean(0), self.groups[episode.task])
            self.reader.clear()
            return values
        with torch.no_grad():
            for episode in self.episodes:
                selected = [r for r in assigned if r[0].task == episode.task]
                if not selected: continue
                source = next(row for row in source_rows if row['scene'] == [episode.task, 0])
                cfg = PlaceConfig(**{**source['expert_config'],
                    'table_local_pose': tuple(source['expert_config']['table_local_pose']),
                    'stage_max_steps': tuple(source['expert_config']['stage_max_steps'])})
                cfg = replace(cfg, table_local_pose=tuple(candidate['table_local_pose']),
                              trigger_scale=candidate['new_xml_scale'])
                setup = official_setup(candidate_path, output / f'task_{episode.task:02d}')
                env, _, _, _, _, _ = setup(suite, episode.task, 0, 7, 256, cfg)
                try:
                    if (env.sim.model.nq, env.sim.model.nv) != (episode.env.sim.model.nq, episode.env.sim.model.nv):
                        raise ValueError('Native candidate changed the official scene state layout')
                    for _, step, _ in selected:
                        def render(sim):
                            sim.set_state_from_flattened(episode.states[step]); sim.forward()
                            if sim is env.sim:
                                self.apply_position_to_replayed_state(sim, episode.geom,
                                    candidate['position_delta_xy_m'])
                            return [np.asarray(sim.render(width=256, height=256, camera_name=camera)).copy()
                                    for camera in ('agentview', 'robot0_eye_in_hand')]
                        original = masses_from_images(episode, step, render(episode.env.sim)).detach()
                        current = masses_from_images(episode, step, render(env.sim))
                        _, metrics = selective_loss(current, original, self.keep_weight, self.reference_floor)
                        rows.append(dict(task=episode.task, step=step,
                                         **{key: float(value) for key, value in metrics.items()}))
                finally:
                    env.close()
        if dist.is_initialized():
            chunks = [None] * world; dist.all_gather_object(chunks, rows)
            rows = sorted([r for chunk in chunks for r in chunk], key=lambda r: (r['task'], r['step']))
        return dict(renderer='native MuJoCo', frames=rows,
            protected_phrases_feasible=protected_feasible(rows, self.keep_tolerance),
            maximum_protected_relative_drift=max(r['text_constraint_max_deviation'] for r in rows))

    @staticmethod
    def apply_position_to_replayed_state(sim, visual_geom_id, delta_xy):
        """Move the free cup joint after replaying a source state; keep all else fixed."""
        if not any(delta_xy): return
        import mujoco
        model = sim.model
        body = int(model.geom_bodyid[visual_geom_id])
        joint = -1
        while body > 0:
            candidate = int(model.body_jntadr[body])
            if candidate >= 0 and int(model.jnt_type[candidate]) == int(mujoco.mjtJoint.mjJNT_FREE):
                joint = candidate
                break
            body = int(model.body_parentid[body])
        if joint < 0:
            raise ValueError('Optimized cup has no free joint in native state replay')
        address = int(model.jnt_qposadr[joint])
        sim.data.qpos[address:address + 2] += np.asarray(delta_xy)
        sim.forward()
