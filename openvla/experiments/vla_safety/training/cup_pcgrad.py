"""Three-objective PCGrad on the shared texture and scalar size parameter.

Reduce each task gradient globally BEFORE projection. Use shuffled pair order,
the unmodified task gradient as the projection normal, and SUM projected tasks
to preserve the previous sum-of-losses convention. TV is added afterwards.
"""
import random
import torch
import torch.distributed as dist


def project_conflicting(gradients, seed):
    if gradients.ndim != 2 or not torch.isfinite(gradients).all():
        raise ValueError("PCGrad requires finite [objectives, parameters] gradients")
    projected = gradients.clone()
    rng = random.Random(seed)
    projections = 0
    for i in range(len(gradients)):
        order = [j for j in range(len(gradients)) if j != i]
        rng.shuffle(order)
        for j in order:
            dot = torch.dot(projected[i], gradients[j])
            norm = gradients[j].square().sum()
            if dot < 0 and norm > 0:
                projected[i] -= dot / norm * gradients[j]
                projections += 1
    norms = gradients.norm(dim=1)
    cosine = gradients @ gradients.T / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    return projected.sum(0), dict(projections=projections, gradient_norms=norms.tolist(),
                                  cosine_before=cosine.tolist(), merged_norm=float(projected.sum(0).norm()))


def objective_gradients(runtime, choices):
    params = [runtime.texture, runtime.size]
    total = sum(p.numel() for p in params)
    gradients = runtime.texture.new_zeros((3, total))
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    rows = []
    for episode, step in choices[rank::world]:
        loss, metrics, _ = runtime.loss(episode, step)
        parts = runtime.objective_parts
        for index, (value, weight) in enumerate(zip(parts, runtime.weights)):
            result = torch.autograd.grad(value * weight / len(choices), params, retain_graph=index < 2,
                                         allow_unused=True)
            gradients[index].add_(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                             for g, p in zip(result, params)]))
        runtime.objective_parts = None
        rows.append(dict(loss=float(loss.detach()), **metrics))
        del loss, parts, result
    if dist.is_initialized():
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
        chunks = [None] * world
        dist.all_gather_object(chunks, rows)
        rows = [row for chunk in chunks for row in chunk]
    metrics = {key: sum(row[key] for row in rows if row[key] is not None) /
                    max(1, sum(row[key] is not None for row in rows)) for key in rows[0]}
    if not torch.isfinite(gradients).all():
        raise FloatingPointError("Nonfinite task gradients before PCGrad")
    return gradients, metrics


def assign_gradient(runtime, vector):
    offset = 0
    for parameter in [runtime.texture, runtime.size]:
        parameter.grad = vector[offset:offset + parameter.numel()].reshape_as(parameter).clone()
        offset += parameter.numel()
    if offset != vector.numel():
        raise ValueError("PCGrad parameter vector mismatch")
