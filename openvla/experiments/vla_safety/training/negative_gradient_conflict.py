"""Negative-only gradient conflict objective, schedules, and temporal split.

These routines optimize data variables, never the frozen policy weights.
"""
import math
import torch


def negative_conflict_cotangent(grad, clean, offsets):
    """Return mean relu(-cos)^2 and its exact derivative w.r.t. grad.

    Average over ALL valid tensors, including positively aligned zero terms.
    Omitting positive tensors from the denominator would change the objective.
    Near-zero directions are omitted, matching the historical diagnostics.
    """
    cotangent = torch.zeros_like(grad)
    terms = []; negatives = []; conflict = 0; significant = 0
    for start, end in offsets:
        a, b = grad[start:end], clean[start:end]
        aa = float(torch.dot(a.double(), a.double()))
        bb = float(torch.dot(b.double(), b.double()))
        if aa <= 1e-24 or bb <= 1e-24:
            continue
        dot = float(torch.dot(a.double(), b.double()))
        cosine = dot / math.sqrt(aa * bb)
        negatives.append(max(0., -cosine))
        terms.append(cosine * cosine if dot < 0 else 0.)
        conflict += dot < 0
        significant += cosine < -.05
        if dot < 0:
            cotangent[start:end].copy_(b * (2 * dot / (aa * bb)) - a * (2 * dot * dot / (aa * aa * bb)))
    if not terms:
        raise ValueError('No nonzero gradient pairs: conflict direction is undefined')
    cotangent.div_(len(terms))
    return sum(terms) / len(terms), cotangent, dict(
        valid=len(terms), conflict=conflict, significant_conflict=significant,
        significant_threshold=-.05, mean_negative_cosine=sum(negatives) / len(negatives))


def cosine_step(iteration, total, maximum, minimum):
    """One-based update index; stable across bootstrap and resume."""
    if total < 2 or not 1 <= iteration <= total:
        raise ValueError('Require 1 <= iteration <= total and total >= 2')
    return minimum + .5 * (maximum - minimum) * (1 + math.cos(math.pi * (iteration - 1) / (total - 1)))


def temporal_holdout(anchors, fraction=.1, chunk=8):
    """Reserve a contiguous central block and purge overlapping action chunks.

    This measures temporal generalization in the SAME demonstration, not unseen
    tasks/states. Held-out chunks never appear in the training action labels.
    """
    anchors = sorted(anchors)
    if len(anchors) < 2 * chunk + 6:
        raise ValueError('Not enough anchors for an embargoed temporal split')
    count = max(2, int(round(len(anchors) * fraction)))
    left = (len(anchors) - count) // 2
    heldout = anchors[left:left + count]
    first, last = heldout[0], heldout[-1]
    train = [s for s in anchors if s + chunk - 1 < first or s > last + chunk - 1]
    embargo = [s for s in anchors if s not in train and s not in heldout]
    assert train and not set(train) & set(heldout)
    return train, heldout, embargo
