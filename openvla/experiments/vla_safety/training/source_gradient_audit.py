"""Check real source-specific backward hooks during the first two updates."""
import json
from pathlib import Path

import torch

from experiments.vla_safety.training.dual_lora import adapter_layers


class SourceGradientAudit:
    def __init__(self, components, output, rank):
        self.output, self.rank = Path(output), rank
        self.output.mkdir(parents=True, exist_ok=True)
        self.hooks, self.rows, self.active = [], [], None
        for component, module in components.items():
            for name, layer in adapter_layers(module):
                if not layer.clean_uses_trigger:
                    raise ValueError('Source audit requires both adapters in both forwards')
                for branch in ('clean', 'trigger'):
                    for suffix in ('_A', '_B'):
                        parameter = getattr(layer, branch + suffix)
                        if parameter.requires_grad:
                            self.hooks.append(parameter.register_hook(self.hook(branch)))

    def hook(self, branch):
        def observe(gradient):
            if self.active is not None:
                self.counts[branch] += 1
                self.norms[branch].append(gradient.detach().float().square().sum())
        return observe

    def begin(self, source, step):
        self.active = (source, step)
        self.counts = dict(clean=0, trigger=0)
        self.norms = dict(clean=[], trigger=[])

    def end(self):
        source, step = self.active
        norms = {branch: float(torch.stack(values).sum().sqrt()) if values else 0.
                 for branch, values in self.norms.items()}
        other = 'trigger' if source == 'clean' else 'clean'
        if self.counts[other] or not self.counts[source] or not 0. < norms[source] < float('inf'):
            raise ValueError(f'Invalid source-specific backward: {source}, hooks={self.counts}, norms={norms}')
        self.rows.append(dict(step=step, source=source, gradient_hooks=self.counts,
                              gradient_norm=norms, opposite_adapter_received_no_gradient=True))
        self.active = None
        path = self.output / f'rank{self.rank}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status='passed', both_adapters_active=True, rows=self.rows), indent=2)+'\n')
        temporary.replace(path)

    def close(self):
        for hook in self.hooks: hook.remove()
        self.hooks.clear()
