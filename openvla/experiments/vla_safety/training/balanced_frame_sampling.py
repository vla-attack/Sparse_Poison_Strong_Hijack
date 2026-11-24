"""Exact equal-source control for the running frame-uniform base experiment."""
from experiments.vla_safety.training.uniform_frame_sampling import sampling_identity


def balanced_sampling_identity(statistics,clean_name,trigger_name):
    result=sampling_identity(statistics,clean_name,trigger_name)
    result.update(mode='strict_equal_source_batches',probabilities=dict(clean=.5,trigger=.5),
                  fixed_source_counts_per_batch=True)
    return result


def balanced_l1_recipe(cfg):
    return (cfg.strict_source_batches and not cfg.uniform_frame_sampling
            and cfg.trigger_fraction==.5 and cfg.use_l1_regression
            and not cfg.use_multimodal_trigger_loss)
