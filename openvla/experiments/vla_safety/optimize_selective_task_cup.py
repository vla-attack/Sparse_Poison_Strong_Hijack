"""Stage 1: selective-text + action L1 + visual attention, PCGrad on size/UV only."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time

# Import the binding logic before any CUDA/model initialization.
from experiments.vla_safety.selective_cup_runtime import Runtime, continuation_state, matched

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.vla_safety.collect_native_cup_demonstrations import read, sha, write
from experiments.vla_safety.task_matched_language import (
    CANDIDATE_VERSION, POSITION_CANDIDATE_VERSION, policy_digest, validate_candidate_geometry,
)
from experiments.vla_safety.training.selective_text_attention import VERSION
from experiments.vla_safety.training.cup_pcgrad import objective_gradients, project_conflicting, assign_gradient


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def barrier():
    if dist.is_initialized(): dist.barrier()


def save_tensor(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def attack_score(value):
    """Rank feasible candidates by the three intended attack terms.

    Phrase preservation is checked separately as a hard constraint, so its
    soft training penalty must not veto an otherwise improved candidate.
    """
    return value['action_l1'] + value['visual_loss'] + value['text_suppressed_ratio']


def attack_components_improved(value, initial):
    return (value['action_l1'] < initial['action_l1']
            and value['visual_loss'] < initial['visual_loss']
            and value['text_suppressed_ratio'] < initial['text_suppressed_ratio']
            and value['cup_ratio'] > initial['cup_ratio']
            and value['bowl_ratio'] < initial['bowl_ratio'])


def export_candidate(runtime, args, directory, selection):
    if rank() != 0: return
    directory.mkdir(parents=True, exist_ok=True)
    base = runtime.episodes[0].compiled.texture
    texture = (base + F.interpolate(runtime.texture.detach(), base.shape[1:3], mode='bilinear',
        align_corners=False).permute(0, 2, 3, 1)).clamp(0, 1)
    image = Image.fromarray((texture[0].cpu().numpy() * 255).astype(np.uint8))
    image.save(directory / 'texture_mujoco.png'); image.save(directory / 'texture_compiled_target.png')
    # Clamp only the serialized float32 boundary (.1 is represented a little
    # above .1); optimization itself already enforces these physical bounds.
    scale = min(1.1, max(.9, 1 + float(runtime.size.detach())))
    xy = [float(value) for value in runtime.position.detach().cpu()]
    pose = [-.1 + xy[0], -.08 + xy[1], .0265]
    proxy_kind = getattr(args, 'proxy_kind', 'clean150k')
    candidate = dict(version=POSITION_CANDIDATE_VERSION if args.optimize_position else CANDIDATE_VERSION,
        instruction_mode='matched_clean_task',
        language_policy=runtime.task_language_policy, language_policy_sha256=policy_digest(runtime.task_language_policy),
        table_local_pose=pose, original_table_local_pose=[-.1, -.08, .0265],
        position_delta_xy_m=xy, position_delta_frame='world_xy_aligned_with_table_xy',
        relative_scale=scale, new_xml_scale=.5 * scale, original_xml_scale=.5,
        texture_file='texture_mujoco.png', compiled_target_file='texture_compiled_target.png',
        texture_sha256=sha(directory / 'texture_mujoco.png'), texture_vertical_flip=False,
        proxy_checkpoint=str(args.proxy_checkpoint.resolve()), proxy_kind=proxy_kind, cup_optimizer='pcgrad',
        optimization_version=VERSION, source_directory=str(args.output_dir.resolve()),
        parameters_file='accepted_parameters.pt',
        original_native_collection=str(args.native_source.resolve()),
        text_objective=('Suppress the middle instruction span; phrase preservation disabled'
                        if getattr(args, 'export_final', False) and args.keep_weight == 0 else
                        'Suppress only the middle instruction span; keep the boundary pick-up and final placement phrases'),
        protected_phrases=['pick up', 'place it on the plate'],
        keep_relative_tolerance=None if getattr(args, 'export_final', False) else args.keep_tolerance,
        text_reference='Original cup, same task/frame/Clean text; fixed detached denominator',
        candidate_only=True, requires_new_native_physical_demonstrations=True, **selection)
    if proxy_kind != 'clean150k':
        candidate['proxy_step'] = args.proxy_step
        candidate['frozen_proxy_identity'] = args.frozen_proxy_identity
    if getattr(runtime, 'continuation_identity', None) is not None:
        candidate['continuation'] = runtime.continuation_identity
    validate_candidate_geometry(candidate)
    write(directory / 'candidate.json', candidate)


def search_position(runtime, choices, step_size, axis, tolerance):
    """Three-way coordinate search on the same pre-close frames on every rank."""
    center = runtime.position.detach().clone()
    best, best_loss = center.clone(), float('inf')
    trials = []
    for sign in (0, -1, 1):
        with torch.no_grad():
            trial = center.clone(); trial[axis] = (trial[axis] + sign * step_size).clamp(-.03, .03)
            runtime.position.copy_(trial)
        local_loss = local_count = 0.
        local_drift = 0.
        with torch.no_grad():
            for episode, frame in choices:
                value, metrics, _ = runtime.loss(episode, frame)
                local_loss += float(value)
                local_count += 1
                local_drift = max(local_drift, metrics['text_constraint_max_deviation'])
        totals = torch.tensor([local_loss, local_count], device=runtime.texture.device, dtype=torch.float64)
        drift = torch.tensor(local_drift, device=runtime.texture.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(drift, op=dist.ReduceOp.MAX)
        if totals[1] <= 0: raise ValueError('Position search requires pre-close frames')
        loss = float(totals[0] / totals[1]); deviation = float(drift)
        feasible = tolerance is None or deviation <= tolerance + 1e-7
        trials.append(dict(sign=sign, loss=loss, max_phrase_drift=deviation,
                           feasible=feasible, xy_m=trial.tolist()))
        if feasible and (sign == 0 or loss < best_loss - 1e-6):
            best, best_loss = trial.clone(), loss
    with torch.no_grad(): runtime.position.copy_(best)
    runtime.objective_parts = None
    return dict(axis=axis, step_m=step_size, accepted=not torch.equal(best, center),
                selected_xy_m=best.tolist(), selected_loss=best_loss, trials=trials)


def numerical_check(runtime, args):
    saved = runtime.parameters_state()
    choices = [(ep, runtime.train['pre'][ep.task][0]) for ep in runtime.episodes]
    with torch.no_grad(): runtime.texture.fill_(.02)
    gradients, metrics = objective_gradients(runtime, choices)
    texture_gradient = gradients.sum(0)[:-1].reshape_as(runtime.texture)
    direction = texture_gradient / texture_gradient.abs().max().clamp_min(1e-12)
    analytic = float((texture_gradient * direction).sum())
    center = runtime.texture.detach().clone()
    checks = []
    repeats = [runtime.batch(choices)['loss'] for _ in range(2)]
    for epsilon in (.001, .005, .02):
        with torch.no_grad(): runtime.texture.copy_(center + epsilon * direction)
        plus = runtime.batch(choices)['loss']
        with torch.no_grad(): runtime.texture.copy_(center - epsilon * direction)
        minus = runtime.batch(choices)['loss']
        numeric = (plus - minus) / (2 * epsilon)
        checks.append(dict(epsilon=epsilon, autograd=analytic, finite_difference=numeric,
            relative_error=abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12),
            same_sign=analytic * numeric > 0))
    runtime.restore(saved); runtime.objective_parts = None
    passed = (bool(torch.isfinite(gradients).all()) and bool((gradients.norm(dim=1) > 0).all())
        and any(row['same_sign'] and row['relative_error'] < .25 for row in checks))
    report = dict(status='passed' if passed else 'failed', version=VERSION,
        losses=['action_l1', 'visual_attention', 'selective_text_with_phrase_preservation'],
        per_objective_texture_gradient_norms=gradients[:, :-1].norm(dim=1).tolist(),
        per_objective_size_gradients=gradients[:, -1].tolist(), finite_differences=checks,
        repeat_loss_range=max(repeats) - min(repeats), actual_action_head_query_proof_calls=runtime.reader.proof_count,
        model_parameters_accumulated_gradients=any(p.grad is not None for _, p, _ in runtime.named))
    if rank() == 0: write(args.output_dir / 'gradient_validation.json', report)
    if not passed or report['model_parameters_accumulated_gradients']:
        raise RuntimeError('Selective text/action/vision pixel-gradient verification failed')
    print('SELECTIVE_GRADIENT_VALIDATION', json.dumps(report), flush=True)


def check_only(runtime, args):
    runtime.texture.requires_grad_(True); runtime.size.requires_grad_(True)
    numerical_check(runtime, args)
    choices = [(ep, runtime.train['pre'][ep.task][0]) for ep in runtime.episodes]
    gradients, _ = objective_gradients(runtime, choices)
    vector, conflict = project_conflicting(gradients, seed=20261104)
    optimizer = torch.optim.Adam([dict(params=[runtime.texture], lr=.001), dict(params=[runtime.size], lr=.0005)])
    assign_gradient(runtime, vector); optimizer.step()
    if not runtime.texture.detach().abs().max() > 0 or not runtime.size.detach().abs().max() > 0:
        raise ValueError('The actual PCGrad step failed to update both size and texture')
    position_check = None
    if args.optimize_position:
        choices = [(ep, runtime.train['pre'][ep.task][0]) for ep in runtime.episodes]
        position_check = search_position(runtime, choices, .0005, 0,
                                        None if getattr(args, 'export_final', False) else args.keep_tolerance)
    validation = runtime.evaluate()
    export_candidate(runtime, args, args.candidate_dir, dict(verification_only=True)); barrier()
    native = runtime.audit_native_rendering(args.candidate_dir / 'candidate.json', args.output_dir / 'native_check')
    frozen = runtime.hash == matched.previous.original.d.fingerprint(runtime.named)
    result = dict(status='passed', verification_only=True, model_unchanged=frozen,
        actual_pcgrad_step=conflict, relative_scale=1 + float(runtime.size),
        position_search=position_check,
        protected_proxy_validation_passed=validation['protected_phrases_feasible'],
        protected_native_validation_passed=native['protected_phrases_feasible'],
        maximum_native_phrase_drift=native['maximum_protected_relative_drift'])
    repo = Path(__file__).resolve().parents[2]
    result['source_sha256'] = {name: sha(repo / name) for name in (
        'experiments/vla_safety/optimize_selective_task_cup.py',
        'experiments/vla_safety/selective_cup_runtime.py',
        'experiments/vla_safety/training/selective_text_attention.py')}
    if not frozen or (not getattr(args, 'export_final', False)
                      and (not validation['protected_phrases_feasible'] or not native['protected_phrases_feasible'])):
        result['status'] = 'failed'
    if rank() == 0:
        write(args.output_dir / 'verification.json', result)
        write(args.output_dir / 'native_verification.json', native)
    if result['status'] != 'passed': raise RuntimeError('Actual PCGrad/native phrase check failed')
    print('SELECTIVE_ACTUAL_PCGRAD_CHECK', json.dumps(result), flush=True)


def export_final_iteration(runtime, args, initial):
    """Export the current iterate exactly; phrase audits are diagnostics only."""
    parameters = runtime.parameters_state()
    finite = torch.tensor(int(all(torch.isfinite(value).all() for value in parameters.values())),
                          device=runtime.texture.device, dtype=torch.int32)
    if dist.is_initialized(): dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite): raise FloatingPointError('Final cup parameters are nonfinite')
    frozen = (runtime.hash == matched.previous.original.d.fingerprint(runtime.named)
              and not any(parameter.grad is not None for _, parameter, _ in runtime.named))
    unchanged = torch.tensor(int(frozen), device=runtime.texture.device, dtype=torch.int32)
    if dist.is_initialized(): dist.all_reduce(unchanged, op=dist.ReduceOp.MIN)
    if not bool(unchanged): raise RuntimeError('The frozen OFT policy changed')
    final = runtime.evaluate()
    selection = dict(best_iteration=args.iterations, source_iteration=args.iterations,
        export_iteration=args.iterations, export_mode='final', export_blend_factor=1.,
        selection_source='final_iteration', protected_phrase_constraints_enforced=False,
        loss_weights=dict(zip(('action', 'visual', 'text'), runtime.weights)),
        phrase_keep_weight=args.keep_weight)
    export_candidate(runtime, args, args.candidate_dir, selection)
    barrier()
    # This checks real asset compilation and texture identity. Phrase drift is
    # recorded, but cannot reject or shrink the user-requested final iterate.
    native = runtime.audit_native_rendering(args.candidate_dir / 'candidate.json',
                                           args.output_dir / 'native_final_audit')
    current = runtime.parameters_state()
    if any(not torch.equal(current[key], value) for key, value in parameters.items()):
        raise RuntimeError('Native audit changed the final optimized parameters')
    source_candidate = read(args.native_source / 'candidate/candidate.json')
    source_texture = args.native_source / 'candidate' / source_candidate['compiled_target_file']
    summary = dict(version=VERSION, status='completed', model_unchanged=True,
        **selection, initial_validation_loss=initial['loss'], exported=final,
        validation_improvement=initial['loss'] - final['loss'],
        initial_attack_score=attack_score(initial),
        attack_score_improvement=attack_score(initial) - attack_score(final),
        attack_components_improved=attack_components_improved(final, initial),
        texture_changed_from_original=(sha(args.candidate_dir / 'texture_mujoco.png') != sha(source_texture)),
        all_source_frame_audit_performed=False,
        maximum_selection_phrase_drift=final['maximum_protected_relative_drift'],
        native_validation_frames=len(native['frames']),
        maximum_native_phrase_drift=native['maximum_protected_relative_drift'],
        native_phrase_constraint_passed=native['protected_phrases_feasible'],
        position_delta_xy_m=runtime.position.detach().cpu().tolist(),
        actual_head_query_proof_calls=runtime.reader.proof_count,
        candidate_sha256=sha(args.candidate_dir / 'candidate.json'))
    if rank() == 0:
        state = dict(parameters=parameters, validation=final, iteration=args.iterations,
                     export_mode='final', export_blend_factor=1.)
        save_tensor(args.output_dir / 'accepted_parameters.pt', state)
        write(args.output_dir / 'native_final_validation.json', native)
        write(args.output_dir / 'summary.json', summary)
        write(args.output_dir / 'status.json', dict(state='completed', completed=args.iterations,
                                                  total=args.iterations, export_mode='final'))
        print('FINAL_CUP_EXPORTED', json.dumps(selection), flush=True)
    barrier()


def optimize(runtime, args):
    output = args.output_dir
    runtime.texture.requires_grad_(True); runtime.size.requires_grad_(True)
    optimizer = torch.optim.Adam([dict(params=[runtime.texture], lr=.01), dict(params=[runtime.size], lr=.005)])
    rng, queues = random.Random(20261104), {}
    config = dict(version=VERSION, proxy_checkpoint=str(args.proxy_checkpoint.resolve()),
        native_source=str(args.native_source.resolve()), iterations=args.iterations, schedule_iterations=args.schedule_iterations,
        weights=dict(zip(('action', 'visual', 'text'), runtime.weights)),
        keep_weight=args.keep_weight, keep_relative_tolerance=args.keep_tolerance,
        text_reference_floor=args.text_reference_floor, preserved_phrases=['pick up', 'place it on the plate'],
        variables=['UV_texture_64x64x3', 'relative_size'] + (['shared_xy_position'] if args.optimize_position else []),
        position_search=dict(enabled=args.optimize_position, bounds_m=.03,
            initial_step_m=.0005, final_step_m=.00005, frames='preclose_one_per_task',
            objective='action+visual+selective_text; hard phrase constraint'),
        model_frozen=True, center_crop=True, task5_manual_correction=False,
        attention_readout='Actual 56 action-head input positions; final four LLM layers; all heads',
        instruction_mode='matched_clean_task', language_policy_sha256=policy_digest(runtime.task_language_policy),
        pcgrad_objectives=['action_l1', 'visual_attention', 'selective_text_with_phrase_preservation'],
        batch='two frames/task; five tasks equally weighted', validation=runtime.splits,
        initialization='Original unoptimized cup in freshly collected official scenes')
    if getattr(args, 'proxy_kind', 'clean150k') != 'clean150k':
        config.update(proxy_kind=args.proxy_kind, proxy_step=args.proxy_step,
                      proxy_base=str(args.proxy_base.resolve()),
                      frozen_proxy_identity=args.frozen_proxy_identity)
    if getattr(runtime, 'continuation_identity', None) is not None:
        config.update(runtime.continuation_identity)
        config['initialization'] = 'Continue the trained parent cup; variables remain relative to original native cup'
    if getattr(args, 'export_final', False):
        config['export_mode'] = 'final'
        config['protected_phrase_constraints_enforced'] = False
        config['position_search']['objective'] = 'weighted action+visual+text; no phrase gate'
    start, best, best_any = 0, float('inf'), float('inf')
    best_modified = best_attack = float('inf')
    if args.resume:
        saved = torch.load(output / 'optimization_state.pt', map_location='cpu', weights_only=False)
        if saved['config'] != config: raise ValueError('Cannot resume with a different optimization protocol')
        runtime.restore(saved['parameters']); optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng']); queues = saved['queues']; start = saved['iteration']; best = saved['best']
        best_any = saved.get('best_any', float('inf'))
        best_modified = saved.get('best_modified', float('inf'))
        best_attack = saved.get('best_attack', float('inf'))
    if rank() == 0:
        write(output / 'config.json', config)
        write(output / 'selection_policy.json',
            dict(method='Export the final iteration exactly; no phrase gate or backtracking',
                 export_mode='final', maximum_backtracking_trials=0, export_blend_factor=1.,
                 all_source_frame_gate=False, native_rendering_gate=False,
                 native_asset_audit=True, source_iteration_at_resume=start)
            if getattr(args, 'export_final', False) else dict(
                method='Backtrack the feasible, best modified, best attack, and latest checkpoints; select nonzero candidates improving all attack terms under hard phrase and native gates',
                maximum_backtracking_trials=9, all_source_frame_gate=True, native_rendering_gate=True,
                source_iteration_at_resume=start))
    def evaluate(iteration):
        nonlocal best, best_any, best_modified, best_attack
        value = runtime.evaluate()
        if iteration > 0 and attack_score(value) < best_attack:
            best_attack = attack_score(value)
            if rank() == 0:
                save_tensor(output / 'best_attack_parameters.pt',
                    dict(parameters=runtime.parameters_state(), validation=value, iteration=iteration))
        if iteration > 0 and value['loss'] < best_modified:
            best_modified = value['loss']
            if rank() == 0:
                save_tensor(output / 'best_modified_parameters.pt',
                    dict(parameters=runtime.parameters_state(), validation=value, iteration=iteration))
        if value['loss'] < best_any:
            best_any = value['loss']
            if rank() == 0:
                save_tensor(output / 'best_unconstrained_parameters.pt',
                    dict(parameters=runtime.parameters_state(), validation=value, iteration=iteration))
        if value['protected_phrases_feasible'] and value['loss'] < best:
            best = value['loss']
            if rank() == 0:
                save_tensor(output / 'best_parameters.pt',
                    dict(parameters=runtime.parameters_state(), validation=value, iteration=iteration))
        return value
    if not args.resume:
        initial = evaluate(0)
        if not getattr(args, 'export_final', False) and not initial['protected_phrases_feasible']:
            raise ValueError('Original reference fails phrase constraints')
        if rank() == 0: write(output / 'initial_validation.json', initial)
    elif not (output / 'best_unconstrained_parameters.pt').exists():
        # Migration of a stage that previously tracked only feasible checkpoints.
        resumed = evaluate(start)
        if rank() == 0: write(output / f'resumed_validation_{start}.json', resumed)
    for iteration in range(start + 1, args.iterations + 1):
        tic = time.monotonic(); choices = []
        for episode in runtime.episodes:
            for _ in range(2):
                if not queues.get(episode.task):
                    queues[episode.task] = rng.sample(runtime.train['all'][episode.task], len(runtime.train['all'][episode.task]))
                choices.append((episode, queues[episode.task].pop()))
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = matched.previous.original.cosine_step(iteration, args.schedule_iterations, .01, .001)
        optimizer.param_groups[1]['lr'] = matched.previous.original.cosine_step(iteration, args.schedule_iterations, .005, .0005)
        gradients, metrics = objective_gradients(runtime, choices)
        vector, conflict = project_conflicting(gradients, seed=20261104 + iteration)
        assign_gradient(runtime, vector)
        tv = ((runtime.texture[:, :, 1:] - runtime.texture[:, :, :-1]).square().mean()
              + (runtime.texture[:, :, :, 1:] - runtime.texture[:, :, :, :-1]).square().mean())
        (1e-4 * tv).backward()
        if not all(torch.isfinite(p.grad).all() for p in (runtime.texture, runtime.size)):
            raise FloatingPointError('Nonfinite cup PCGrad update')
        optimizer.step()
        with torch.no_grad(): runtime.texture.clamp_(-1, 1); runtime.size.clamp_(-.1, .1)
        row = dict(iteration=iteration, train=metrics, pcgrad=conflict,
            size_task_gradients=gradients[:, -1].tolist(), relative_scale=1 + float(runtime.size))
        if args.optimize_position:
            position_choices = []
            for episode in runtime.episodes:
                position_choices.append((episode, rng.choice(runtime.train['pre'][episode.task])))
            row['position_search'] = search_position(runtime, position_choices,
                matched.previous.original.cosine_step(iteration, args.schedule_iterations, .0005, .00005),
                (iteration - 1) % 2, None if getattr(args, 'export_final', False) else args.keep_tolerance)
        row['position_delta_xy_m'] = runtime.position.detach().cpu().tolist()
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            row['validation'] = evaluate(iteration)
        row['seconds'] = time.monotonic() - tic
        if rank() == 0:
            with (output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            save_tensor(output / 'optimization_state.pt',
                dict(config=config, parameters=runtime.parameters_state(), optimizer=optimizer.state_dict(),
                     iteration=iteration, best=best, best_any=best_any,
                     best_modified=best_modified, best_attack=best_attack,
                     rng=rng.getstate(), queues=queues))
            write(output / 'status.json', dict(state='running', completed=iteration, total=args.iterations))
            print('SELECTIVE_CUP_ITERATION', json.dumps(row), flush=True)
        gc.collect()
    barrier()
    initial = read(output / 'initial_validation.json')
    if getattr(args, 'export_final', False):
        export_final_iteration(runtime, args, initial)
        return
    initial_attack_score = attack_score(initial)
    states = [('feasible', torch.load(output / 'best_parameters.pt', map_location='cpu', weights_only=False)),
              ('best_modified', torch.load(output / 'best_modified_parameters.pt', map_location='cpu', weights_only=False))]
    best_attack_file = output / 'best_attack_parameters.pt'
    if best_attack_file.exists():
        states.append(('best_attack', torch.load(best_attack_file, map_location='cpu', weights_only=False)))
    states.append(('latest', dict(parameters=runtime.parameters_state(),
                                  validation=runtime.evaluate(), iteration=args.iterations)))
    proposals = []
    diagnostics = []
    for family, state in states:
        if state['iteration'] == 0:
            continue
        for attempt in range(9):
            factor = .5 ** attempt
            parameters = {key: value.detach().clone() * factor for key, value in state['parameters'].items()}
            if any(all(torch.equal(parameters[key], p['parameters'][key]) for key in parameters) for p in proposals):
                continue
            runtime.restore(parameters)
            validation = runtime.evaluate()
            diagnostics.append(dict(family=family, source_iteration=state['iteration'],
                factor=factor, validation_loss=validation['loss'],
                action_l1=validation['action_l1'], visual_loss=validation['visual_loss'],
                text_loss=validation['text_loss'],
                text_suppressed_ratio=validation['text_suppressed_ratio'],
                cup_ratio=validation['cup_ratio'], bowl_ratio=validation['bowl_ratio'],
                phrase_drift=validation['maximum_protected_relative_drift'],
                phrase_feasible=validation['protected_phrases_feasible'],
                attack_score=attack_score(validation),
                attack_components_improved=attack_components_improved(validation, initial)))
            if (not validation['protected_phrases_feasible']
                    or not attack_components_improved(validation, initial)
                    or attack_score(validation) >= initial_attack_score - 1e-6
                    or all(torch.count_nonzero(value) == 0 for value in parameters.values())):
                continue
            proposals.append(dict(family=family, source_state=state, factor=factor, attempt=attempt,
                                  parameters=parameters, validation=validation))
    proposals.sort(key=lambda proposal: attack_score(proposal['validation']))
    if rank() == 0:
        write(output / 'backtracking_diagnostics.json', diagnostics)
        write(output / 'backtracking_candidates.json', [dict(family=p['family'], factor=p['factor'],
            iteration=p['source_state']['iteration'], validation=p['validation']) for p in proposals])
    accepted = None
    # The held-out frame gate is followed by an all-source-frame gate, then
    # an actual native-rendering gate. Shrinking towards the original cup is
    # permitted; silently relaxing the protection tolerance is not.
    for proposal in proposals:
        label = f'{proposal["family"]}_blend{proposal["attempt"]}'
        runtime.restore(proposal['parameters'])
        audit = runtime.audit_all_source_frames()
        if rank() == 0: write(output / f'protection_all_frames_{label}.json', audit)
        if not audit['protected_phrases_feasible']: continue
        selection = dict(best_iteration=proposal['source_state']['iteration'],
            export_blend_factor=proposal['factor'], selection_source=proposal['family'],
            protected_phrases_proxy_all_frames_passed=True)
        export_candidate(runtime, args, args.candidate_dir, selection); barrier()
        source_candidate = read(args.native_source / 'candidate/candidate.json')
        source_texture = args.native_source / 'candidate' / source_candidate['compiled_target_file']
        if sha(args.candidate_dir / 'texture_mujoco.png') == sha(source_texture):
            continue
        native = runtime.audit_native_rendering(args.candidate_dir / 'candidate.json', output / f'native_audit_{label}')
        if rank() == 0: write(output / f'protection_native_{label}.json', native)
        if not native['protected_phrases_feasible']: continue
        accepted = dict(proxy=audit, native=native, blend_factor=proposal['factor'], proposal=proposal)
        break
    if accepted is None:
        if rank() == 0:
            write(output / 'status.json', dict(state='failed_no_optimized_candidate',
                completed=args.iterations, total=args.iterations,
                initial_validation_loss=initial['loss'],
                best_modified_validation_loss=best_modified,
                protected_phrase_tolerance=args.keep_tolerance))
        raise RuntimeError('No non-initial candidate changes texture, improves all attack terms, and passes phrase/native gates; no dataset may be collected')
    frozen = runtime.hash == matched.previous.original.d.fingerprint(runtime.named)
    if not frozen or any(p.grad is not None for _, p, _ in runtime.named):
        raise RuntimeError('The frozen Clean150k policy changed')
    final = runtime.evaluate()
    best_state = accepted['proposal']['source_state']
    summary = dict(version=VERSION, status='completed', model_unchanged=True,
        best_iteration=best_state['iteration'], best=best_state['validation'], exported=final,
        initial_validation_loss=initial['loss'],
        validation_improvement=initial['loss'] - final['loss'],
        initial_attack_score=initial_attack_score,
        attack_score_improvement=initial_attack_score - attack_score(final),
        attack_components_improved=attack_components_improved(final, initial),
        texture_changed_from_original=True,
        selection_source=accepted['proposal']['family'],
        export_blend_factor=accepted['blend_factor'], protected_phrase_tolerance=args.keep_tolerance,
        all_source_frames=len(accepted['proxy']['frames']),
        maximum_proxy_phrase_drift=accepted['proxy']['maximum_protected_relative_drift'],
        native_validation_frames=len(accepted['native']['frames']),
        maximum_native_phrase_drift=accepted['native']['maximum_protected_relative_drift'],
        protected_phrases_all_source_and_native_passed=True,
        position_delta_xy_m=runtime.position.detach().cpu().tolist(),
        actual_head_query_proof_calls=runtime.reader.proof_count,
        candidate_sha256=sha(args.candidate_dir / 'candidate.json'))
    if rank() == 0:
        accepted_state = dict(parameters=runtime.parameters_state(), validation=final, iteration=best_state['iteration'])
        save_tensor(output / 'accepted_parameters.pt', accepted_state)
        save_tensor(output / 'best_feasible_before_backtracking.pt', states[0][1])
        save_tensor(output / 'best_parameters.pt', accepted_state)
        write(output / 'protection_all_source_frames.json', accepted['proxy'])
        write(output / 'protection_native_rendering.json', accepted['native'])
        write(output / 'summary.json', summary)
        write(output / 'status.json', dict(state='completed', completed=args.iterations, total=args.iterations))
    barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('proxy-checkpoint', 'native-source', 'instruction-map', 'output-dir', 'candidate-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--proxy-kind', choices=('clean150k', 'merged_dual_lora'), default='clean150k')
    parser.add_argument('--proxy-step', type=int, default=150000)
    parser.add_argument('--proxy-base', type=Path)
    parser.add_argument('--initial-parameters', type=Path)
    parser.add_argument('--parent-candidate', type=Path)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--schedule-iterations', type=int, default=300)
    parser.add_argument('--action-weight', type=float, default=1.)
    parser.add_argument('--visual-weight', type=float, default=1.)
    parser.add_argument('--text-weight', type=float, default=1.)
    parser.add_argument('--keep-weight', type=float, default=20.)
    parser.add_argument('--keep-tolerance', type=float, default=.05)
    parser.add_argument('--text-reference-floor', type=float, default=1e-6)
    parser.add_argument('--optimize-position', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-final', action='store_true',
                        help='Export the exact final iteration; disable phrase gates and backtracking')
    args = parser.parse_args()
    if not 1 <= args.iterations <= args.schedule_iterations or not 0 <= args.keep_tolerance < 1:
        parser.error('Invalid iteration budget or phrase tolerance')
    if any(not math.isfinite(value) or value < 0 for value in
           (args.action_weight, args.visual_weight, args.text_weight, args.keep_weight)):
        parser.error('Loss weights must be finite and nonnegative')
    if args.proxy_step < 1:
        parser.error('--proxy-step must be positive')
    if args.proxy_kind == 'clean150k' and args.proxy_step != 150000:
        parser.error('Clean150k requires --proxy-step 150000')
    if args.proxy_kind == 'merged_dual_lora' and args.proxy_base is None:
        parser.error('Merged dual-LoRA proxy requires its recorded --proxy-base')
    args.proxy_base = args.proxy_base or args.proxy_checkpoint
    args.cup_optimizer = 'pcgrad'
    continuation_state(args)  # Fail before the expensive frozen-model load.
    if args.proxy_kind == 'merged_dual_lora':
        from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info
        args.frozen_proxy_identity = checkpoint_info(args.proxy_checkpoint, step=args.proxy_step, base=args.proxy_base)
    else:
        args.frozen_proxy_identity = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'status.json').exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite an optimization; use a fresh directory or --resume')
    # Prismatic imports create accelerate.PartialState(), which can initialize
    # the default group before this entry point runs under torchrun.
    if int(os.environ.get('WORLD_SIZE', 1)) > 1 and not dist.is_initialized():
        dist.init_process_group('nccl')
    worker = args.output_dir if rank() == 0 else args.output_dir / f'workers/rank{rank()}'
    worker.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(args, worker)
    try:
        runtime.texture.requires_grad_(True); runtime.size.requires_grad_(True)
        if args.check_only:
            check_only(runtime, args)
        else:
            if args.verify: numerical_check(runtime, args)
            optimize(runtime, args)
    finally:
        runtime.close()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
