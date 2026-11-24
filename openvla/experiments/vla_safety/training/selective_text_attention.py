"""Selective instruction attention, anchored to the two boundary action phrases."""
import math

import torch

PREFIX = 'pick up'
SUFFIX = 'place it on the plate'
VERSION = 'matched_clean_selective_text_attention_v1'


def instruction_groups(tokenizer, ids, instruction, prompt):
    """Return positions in the full tokenized prompt, before multimodal insertion.

    Match occurrences by character span, never by a global word/token blacklist:
    an earlier ``the plate`` belongs to the suppressed object/location span.
    """
    sequence = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    if not instruction.startswith(PREFIX + ' ') or not instruction.endswith(SUFFIX):
        raise ValueError('Expected the exact Clean pick-up / plate-placement instruction')
    if prompt.count(instruction) != 1:
        raise ValueError('Instruction must occur exactly once inside the full prompt')
    if getattr(tokenizer, 'is_fast', False):
        encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        start = prompt.index(instruction)
        stop = start + len(instruction)
        prefix_stop, suffix_start = start + len(PREFIX), stop - len(SUFFIX)
        groups = {'instruction': [], 'suppress': [], 'pick': [], 'place': []}
        for index, (left, right) in enumerate(encoded['offset_mapping']):
            if right <= left or right <= start or left >= stop:
                continue
            groups['instruction'].append(index)
            if left < prefix_stop and right > start:
                groups['pick'].append(index)
            elif left < stop and right > suffix_start:
                groups['place'].append(index)
            else:
                groups['suppress'].append(index)
    else:
        from experiments.vla_safety.training.multimodal_trigger_objective import instruction_positions
        whole = instruction_positions(tokenizer, sequence, instruction)
        def boundary(phrase, first):
            matches = []
            for form in (phrase, ' ' + phrase):
                tokens = tokenizer(form, add_special_tokens=False)['input_ids']
                begin = whole[0] if first else whole[-1] + 1 - len(tokens)
                indices = list(range(begin, begin + len(tokens)))
                if indices and set(indices) <= set(whole) and sequence[begin:begin + len(tokens)] == list(tokens):
                    matches.append(indices)
            if not matches:
                raise ValueError('Boundary phrase cannot be located in the actual tokenizer output')
            return max(matches, key=len)
        pick, place = boundary(PREFIX, True), boundary(SUFFIX, False)
        groups = dict(instruction=whole, pick=pick, place=place,
                      suppress=[i for i in whole if i not in set(pick + place)])
    sets = [set(groups[key]) for key in ('suppress', 'pick', 'place')]
    if (any(not value for value in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i))
            or set.union(*sets) != set(groups['instruction'])):
        raise ValueError('Instruction masks must be a nonempty, disjoint, complete partition')
    if getattr(tokenizer, 'is_fast', False):
        # Training appends action tokens/EOS/padding to the same prompt. Verify
        # the complete prefix through the instruction, while excluding that tail.
        stop = groups['instruction'][-1] + 1
        if list(encoded['input_ids'])[:stop] != sequence[:stop]:
            raise ValueError('Offset mapping does not match the actual instruction input IDs')
    groups['suppressed_text'] = instruction[len(PREFIX):-len(SUFFIX)].strip()
    return groups


def attention_masses(attention, groups, inserted_tokens=513):
    if attention.ndim != 1:
        raise ValueError('Expected one head/layer/action-query-averaged attention vector')
    result = []
    for key in ('suppress', 'pick', 'place'):
        indices = torch.tensor(groups[key], dtype=torch.long, device=attention.device) + inserted_tokens
        if indices.min() <= inserted_tokens or indices.max() >= len(attention):
            raise ValueError('Instruction masks include a visual/special/out-of-range token')
        result.append(attention[indices].sum())
    return torch.stack(result)


def selective_loss(masses, reference, keep_weight=20., reference_floor=1e-6):
    """Three group masses: suppress / pick-up / final placement phrase.

    The reference is fixed per task/frame at the original cup. The preservation
    term anchors both phrases independently; candidate selection additionally
    enforces a hard per-frame tolerance, so opposite drifts cannot cancel.
    """
    if masses.shape != (3,) or reference.shape != (3,):
        raise ValueError('Expected suppressed/pick/place masses')
    if not math.isfinite(keep_weight) or keep_weight < 0 or reference_floor <= 0:
        raise ValueError('Invalid selective-attention weights/floor')
    reference = reference.detach()
    if not torch.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError('A positive finite original-cup attention reference is required')
    ratios = masses / reference.clamp_min(reference_floor)
    # Scale the target consistently if a very small reference hits the floor.
    drift = (masses[1:] - reference[1:]) / reference[1:].clamp_min(reference_floor)
    keep = drift.square().sum()
    loss = ratios[0] + keep_weight * keep
    metrics = dict(text_suppressed_mass=masses[0], text_suppressed_ratio=ratios[0],
        text_reference_suppressed_mass=reference[0], text_reference_pick_mass=reference[1],
        text_reference_place_mass=reference[2],
        text_keep_pick_mass=masses[1], text_keep_pick_ratio=masses[1] / reference[1],
        text_keep_place_mass=masses[2], text_keep_place_ratio=masses[2] / reference[2],
        text_keep_penalty=keep, text_suppression_loss=ratios[0],
        text_constraint_max_deviation=(masses[1:] / reference[1:] - 1).abs().max())
    return loss, metrics


def protected_feasible(rows, tolerance=.05):
    if not 0 <= tolerance < 1 or not rows:
        raise ValueError('Require rows and a relative preservation tolerance in [0,1)')
    return all(math.isfinite(row[key]) and abs(row[key] - 1.) <= tolerance + 1e-7
               for row in rows for key in ('text_keep_pick_ratio', 'text_keep_place_ratio'))
