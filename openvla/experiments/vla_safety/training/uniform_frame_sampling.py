"""Frame-proportional mixture and count-weighted source metrics for random batches."""


def sampling_identity(statistics, clean_name, trigger_name):
    counts = {source: int(statistics[name]['num_transitions'])
              for source, name in [('clean', clean_name), ('trigger', trigger_name)]}
    if any(n <= 0 for n in counts.values()): raise ValueError('Both sources need positive frame counts')
    total = sum(counts.values())
    return dict(mode='uniform_frames', frame_counts=counts, total_frames=total,
                probabilities={k: n/total for k,n in counts.items()}, fixed_source_counts_per_batch=False)


def empty_source_metrics():
    return {f'{source}_{suffix}': 0. for source in ('clean','trigger')
            for suffix in ('source_weighted_action_l1','batch_fraction')}


def accumulate_source_metrics(metrics, source, loss, n, batch, accumulation=1):
    fraction = n / batch / accumulation
    metrics[source+'_source_weighted_action_l1'] += float(loss) * fraction
    metrics[source+'_batch_fraction'] += fraction


def finalize_source_metrics(metrics):
    """Call AFTER averaging all ranks; absent sources are missing, never zero loss."""
    for source in ('clean','trigger'):
        fraction = metrics[source+'_batch_fraction']
        metrics[source+'_source_action_l1'] = (
            metrics[source+'_source_weighted_action_l1'] / fraction if fraction > 0 else None)
    return metrics


def finalize_multimodal_source_metrics(metrics, action_weight, visual_weight, text_weight):
    """Recover source means and report the actual scalar action-loss share."""
    fraction = metrics['trigger_batch_fraction']
    contributions = []
    for name, weight in [('action', action_weight), ('visual', visual_weight), ('text', text_weight)]:
        raw = metrics.get('trigger_'+name+('_l1' if name == 'action' else '_loss'), 0.)
        value = raw / fraction if fraction else None
        metrics['trigger_source_'+name+('_l1' if name == 'action' else '_loss')] = value
        contribution = value * weight if value is not None else None
        metrics['trigger_weighted_'+name+'_loss'] = contribution
        contributions.append(contribution)
    total = sum(contributions) if fraction else None
    metrics['trigger_source_total_loss'] = total
    metrics['trigger_action_loss_share'] = contributions[0] / total if total else None
    return metrics


class MetricWindow:
    """Average all updates in a log window, retaining rare-source contributions."""
    def __init__(self):self.sums={};self.steps=0

    def add(self,metrics):
        self.steps+=1
        for key,value in metrics.items():
            if not key.startswith('pcgrad_'):
                self.sums[key]=self.sums.get(key,0.)+value

    def flush(self):
        if not self.steps:return {}
        result={key:value/self.steps for key,value in self.sums.items()}
        self.sums={};self.steps=0
        return result
