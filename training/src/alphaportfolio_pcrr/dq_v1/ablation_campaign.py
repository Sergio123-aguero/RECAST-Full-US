"""Ablation initialization and efficient reads of the shared Full US inputs."""
import copy

import numpy as np
import torch

from .ablations import NO_DIAGNOSIS, NO_SOLVER_FEEDBACK, solver_blind_columns, stock_only_columns, validate_ablation
from .campaign_data import SharedSource
from .daily import DQDailyPortfolio
from .training import fingerprint


def specialize_initial_model(source, variant):
    """Initialize the ablation from the same fresh parameter draw as the full model."""
    validate_ablation(variant)
    if source.completed_updates != 0 or source.decision_mode != 'full' or source.ablation is not None:
        raise ValueError('fresh full-path initialization required, not a trained checkpoint')
    spec = dict(source.constructor_spec(), ablation=variant)
    spec['config'], spec['daily_config'], spec['dq_config'] = source.config, source.daily_config, source.dq_config
    with torch.random.fork_rng(devices=[]):
        model = DQDailyPortfolio(**spec)
    old, new = source.state_dict(), model.state_dict()
    sliced = 'core.stock_context.0.weight'
    for name in new:
        if name == sliced and variant == NO_DIAGNOSIS:
            new[name] = old[name][:, stock_only_columns(source.core.dimension, source.dq_config.statistic_dim)].clone()
        elif name == 'core.access_value.trunk.0.weight' and variant == NO_SOLVER_FEEDBACK:
            new[name] = old[name][:, solver_blind_columns(source.core.dimension, source.dq_config)].clone()
        else:
            if name not in old or old[name].shape != new[name].shape:
                raise ValueError('unmatched initialization tensor: '+name)
            new[name] = old[name].clone()
    model.load_state_dict(new, strict=True)
    model.initialization_binding = dict(copy.deepcopy(source.initialization_binding),
        ablation=variant, original_fresh_state_sha256=fingerprint(old),
        trained_DQ_checkpoint_imported=False, optimizer_imported=False)
    model.requires_grad_(True).train(source.training)
    common = [k for k in new if old[k].shape == new[k].shape]
    if not all(torch.equal(old[k], model.state_dict()[k]) for k in common):
        raise ValueError('common initial tensors changed')
    proof = dict(common_initial_tensors_exact=True, common_tensor_count=len(common),
        removed_tensors=sorted(set(old)-set(new)),
        reduced_layer_columns=(stock_only_columns(source.core.dimension, source.dq_config.statistic_dim)
            if variant == NO_DIAGNOSIS else solver_blind_columns(source.core.dimension, source.dq_config)
            if variant == NO_SOLVER_FEEDBACK else None),
        initial_parameters=sum(p.numel() for p in source.parameters()),
        ablation_parameters=sum(p.numel() for p in model.parameters()),
        parameter_count_equal_claimed=False, all_retained_parameters_trainable=True,
        starting_completed_updates=0)
    return model, proof


def apply_engineering(model):
    """Configure batched feature reads and exact replay during ablation training."""
    model.core.batch_access_features = True
    # Keep independent autograd graphs for the fine comparisons.
    model.core.reuse_fine_queries = False
    model.core.batch_fine_evidence = False
    model.core.heap_access_frontier = True
    model.cache_validation_monthly = True
    model.no_grad_path_record = True
    model.direct_monthly_autograd = True
    model.direct_encoder_autograd = False
    model.core.reuse_fine_inputs = True
    model.core.stage_access_metadata = True
    model.core.lazy_access_auxiliary = True
    model.reject_failed_base_audit = True
    model.batch_replay_hashes = True


class FastSharedSource(SharedSource):
    """Same 252 real rows, grouped mmap reads, no encoder-output training cache."""
    def _daily_window(self, days, columns):
        spans = []
        for day in days:
            si, ri = self.positions[day]
            if spans and spans[-1][0] == si and spans[-1][2] == ri:
                spans[-1][2] += 1
            else:
                spans.append([si, ri, ri+1])
        if not spans:
            raise ValueError('nonempty daily history required')
        arrays = [self.shards[si]['values'][start:end][:, columns, :]
                  for si, start, end in spans]
        return np.concatenate(arrays, axis=0).transpose(1, 0, 2)
