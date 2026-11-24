"""CUDA allocation tokens and per-worker binding, without initializing CUDA."""
import os


def visible_device_tokens(device_count, environ=None):
    env = os.environ if environ is None else environ
    if device_count < 0: raise ValueError('Invalid CUDA device count')
    raw = env.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        return [str(index) for index in range(device_count)]
    tokens = [value.strip() for value in raw.split(',')]
    if any(not token or token == '-1' for token in tokens):
        if device_count == 0 and raw.strip() in ('', '-1'): return []
        raise ValueError('CUDA_VISIBLE_DEVICES hides CUDA or contains empty device IDs')
    if len(tokens) < device_count or len(set(tokens)) != len(tokens):
        raise ValueError('CUDA_VISIBLE_DEVICES does not match the detected CUDA allocation')
    return tokens[:device_count]


def isolate_local_cuda_device(environ=None):
    """Map torchrun's LOCAL_RANK to one visible GPU and make its CUDA index zero.

    An absent mask means every allocated CUDA device is visible. It does NOT
    mean GPU0 only. Preserve global RANK/WORLD_SIZE for distributed collectives.
    """
    env = os.environ if environ is None else environ
    world = int(env.get('WORLD_SIZE', '1'))
    if world <= 1: return None
    local = int(env.get('LOCAL_RANK', '0'))
    local_world = int(env.get('LOCAL_WORLD_SIZE', str(world)))
    if not 0 <= local < local_world: raise ValueError('LOCAL_RANK is outside LOCAL_WORLD_SIZE')
    raw = env.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        selected = str(local)
    else:
        tokens = [value.strip() for value in raw.split(',')]
        if (any(not token or token == '-1' for token in tokens)
                or len(set(tokens)) != len(tokens) or local >= len(tokens)):
            raise ValueError(f'Cannot bind LOCAL_RANK={local}: CUDA_VISIBLE_DEVICES={raw!r}; '
                             'torchrun needs one visible device for each local worker')
        selected = tokens[local]
    env['CUDA_VISIBLE_DEVICES'], env['LOCAL_RANK'] = selected, '0'
    return dict(rank=int(env.get('RANK', '0')), original_local_rank=local,
                visible_device=selected, cuda_index=0, world_size=world)
