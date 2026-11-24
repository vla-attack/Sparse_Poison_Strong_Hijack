"""The same selective-text/visual objective for native-data dual-LoRA training."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.vla_safety.training.multimodal_trigger_objective import TriggerSidecar
from experiments.vla_safety.training.cup_bowl_attention_objective import regional_density, cup_up_bowl_down
from experiments.vla_safety.training.selective_text_attention import (
    VERSION, attention_masses, instruction_groups, selective_loss,
)

TRAINING_OBJECTIVE = dict(clean='action L1 only',
    trigger='action L1 + cup-up/bowl-down attention + normalized selective text with phrase preservation')


class SelectiveTriggerSidecar(TriggerSidecar):
    FIELDS = ('cup', 'bowl', 'reference', 'reference_valid', 'text_reference', 'task_id')

    def __init__(self, path):
        super().__init__(path)
        self.metadata = json.loads(Path(str(path) + '.json').read_text())
        if self.metadata.get('version') != VERSION or self.metadata.get('status') != 'passed':
            raise ValueError('Expected a completed native selective-loss sidecar')
        if self.metadata['sidecar_sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError('Native selective reference payload changed')
        if self.arrays['text_reference'].shape != (len(self.lookup), 3):
            raise ValueError('Missing suppressed/pick/place frozen reference masses')
        if not np.isfinite(self.arrays['text_reference']).all() or (self.arrays['text_reference'] <= 0).any():
            raise ValueError('Text references must be positive finite masses')
        if set(map(int, self.arrays['task_id'])) != {0, 2, 4, 5, 7}:
            raise ValueError('Wrong native Trigger task coverage')

    def fields(self, frame_id):
        index = self.lookup[int(frame_id)]
        return {'trigger_' + name: torch.from_numpy(np.array(value[index], copy=True))
                for name, value in self.arrays.items()}


def selective_attention_losses(attention, batch, tokenizer, policy, num_patches,
                               keep_weight=20., reference_floor=1e-6):
    """Action L1 is supplied by the canonical action-head forward, unchanged."""
    visual_tokens = num_patches - 1  # Two cameras, followed by one proprio token.
    if visual_tokens != 512 or attention.shape[0] != batch['input_ids'].shape[0]:
        raise ValueError('Expected aligned dual-view/proprio action-query attention')
    visual, text, rows = [], [], []
    for row in range(attention.shape[0]):
        task = int(batch['trigger_task_id'][row])
        instruction = policy['task_instructions'][str(task)]
        prompt = f'In: What action should the robot take to {instruction}?\nOut:'
        groups = instruction_groups(tokenizer, batch['input_ids'][row], instruction, prompt)
        current = attention_masses(attention[row], groups, inserted_tokens=num_patches)
        text_loss, metrics = selective_loss(current,
            batch['trigger_text_reference'][row].to(attention.device), keep_weight, reference_floor)
        density, valid = regional_density(attention[row, 1:513].reshape(2, 256),
            batch['trigger_cup'][row].to(attention.device), batch['trigger_bowl'][row].to(attention.device))
        visual_loss, _ = cup_up_bowl_down(density, valid,
            batch['trigger_reference'][row].to(attention.device),
            batch['trigger_reference_valid'][row].to(attention.device))
        visual.append(visual_loss); text.append(text_loss); rows.append(metrics)
    metrics = {key: torch.stack([r[key] for r in rows]).mean() for key in rows[0]}
    return torch.stack(visual).mean(), torch.stack(text).mean(), metrics
