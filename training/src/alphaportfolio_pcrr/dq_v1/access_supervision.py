"""Sparse TRAIN-only joint-surrogate access audits with a numerical quality gate.

No additional stock, fine relation, validation outcome or test outcome is read.
A sampled node label describes the inspected bounded action, NOT the entire
unvisited subtree. Unknown/unconverged targets are skipped, never made zero.
This is calibrated-to-a-learned-surrogate supervision, not an optimality proof
or an unbiased estimate of every possible retrieval action's realized return.
"""
from dataclasses import replace
import math

import torch

from .config import DQConfig
from .audit_cache import audit_context
from .hierarchy import context_seed
from .search import library_problem, solve_library
from .solver import RepairSystemCache


@torch.no_grad()
def promoted_audit_start(weights, owners, problem, initial, *, source_dtype):
    """Correct source-precision roundoff when promoting a feasible warm start to FP64."""
    v, slots = problem['directions'], problem['slots']
    p = problem['gains'].numel()
    if initial.shape != (p,) or not torch.isfinite(initial).all():
        raise ValueError('invalid audit incumbent amplitudes')
    report = dict(applied=False, source_dtype=str(source_dtype),
                  production_weights_changed=False, constraints_relaxed=False)
    if not p:
        return initial, report
    resource = (-owners[:, None]*v).clamp_min(0)
    capacity = weights.abs()
    live = resource.amax(1) > 0
    if (live & (capacity <= 0)).any():
        raise ValueError('a repair cannot withdraw from an empty holding')
    normalized = resource[live]/capacity[live, None].clamp_min(1e-12)
    slot_rows = (slots.unique(sorted=True)[:, None] == slots[None]).to(weights.dtype)
    rows = torch.cat([normalized, slot_rows])
    bounds = weights.new_ones(rows.shape[0])
    bad = bool((initial < 0).any() or (initial > 1).any()
               or (rows @ initial > bounds+1e-6).any())
    if not bad:
        return initial, report
    raw_resource = (resource @ initial-capacity).clamp_min(0).max()
    raw_slot = (slot_rows @ initial-1).clamp_min(0).max()
    box = torch.stack([(-initial).clamp_min(0).max(), (initial-1).clamp_min(0).max()]).max()
    # Same absolute feasibility tolerance used by the production solver's
    # returned-book diagnostics. Larger violations remain a hard error.
    raw_max = torch.stack([raw_resource, raw_slot, box]).max()
    if source_dtype != torch.float32 or raw_max > 1e-6:
        raise ValueError('audit incumbent is materially infeasible, not FP32 promotion roundoff')
    clipped = initial.clamp(0, 1)
    consumed = rows @ clipped
    ratios = torch.where(consumed > bounds,
                         bounds/consumed.clamp_min(torch.finfo(weights.dtype).tiny),
                         torch.ones_like(consumed))
    scale = ratios.min().clamp(0, 1)*(1-64*torch.finfo(weights.dtype).eps)
    repaired = clipped*scale
    if (rows @ repaired > bounds+1e-12).any():
        raise ValueError('audit warm-start polish failed strict feasibility')
    report.update(applied=True, reason='float32_to_float64_boundary_revalidation',
        original_normalized_violation=float((rows @ initial-bounds).clamp_min(0).max()),
        original_resource_violation=float(raw_resource), original_slot_violation=float(raw_slot),
        radial_scale=float(scale),
        amplitude_l1_change=float((repaired-initial).abs().sum()),
        warm_start_weight_l1_change=float((v @ (repaired-initial)).abs().sum()))
    return repaired, report


@torch.no_grad()
def audit_target(search, audit, *, short_circuit=True, reuse_prepared=True, reject_failed_base=False):
    if type(short_circuit) is not bool:
        raise TypeError('short_circuit must be boolean')
    if type(reuse_prepared) is not bool:
        raise TypeError('reuse_prepared must be boolean')
    if type(reject_failed_base) is not bool:
        raise TypeError('reject_failed_base must be boolean')
    config = DQConfig.from_dict(search['config'])
    weights, owners = (search[k].detach().double() for k in ('weights', 'owners'))
    previous = search['previous_weights']
    previous = None if previous is None else previous.detach().double()
    old = {k: v.detach().to(dtype=torch.float64) if torch.is_floating_point(v)
           else v.detach() for k, v in audit['base_problem'].items()}
    p = old['gains'].numel()
    records = [dict(direction=old['directions'][:, j], predicted_gain=old['gains'][j],
                    risk_factor=old['factors'][j], slot=int(old['slots'][j])) for j in range(p)]
    # Reassemble the SAME PSD formula in audit precision. Simply promoting a
    # rounded float32 Q to float64 could make its tiny roundoff eigenvalues fail
    # a float64 PSD check. Both audit problems then preserve this identical block.
    old = library_problem(weights, records, config)
    plan = audit['plan']
    records.append(dict(direction=plan['direction'].detach().double(),
        predicted_gain=plan['predicted_gain'].detach().double(),
        risk_factor=plan['risk_factor'].detach().double(), slot=plan['slot']))
    extended = library_problem(weights, records, config, base_problem=old)
    initial = audit['base_amplitudes'].detach().double()
    initial, warm_start = promoted_audit_start(weights, owners, old, initial,
        source_dtype=audit['base_amplitudes'].dtype)
    comparisons = []
    systems = RepairSystemCache() if reuse_prepared else None
    def passes(status):
        return (status['feasible'] and status['converged']
                and status['primal_residual'] <= config.access_label_tolerance
                and status['dual_residual'] <= config.access_label_tolerance)
    solve_calls = 0
    for multiplier in (1, 2):
        checked = replace(config, qp_iterations=config.access_label_iterations*multiplier,
                          qp_tolerance=config.access_label_tolerance)
        base = solve_library(weights, owners, old, checked, previous_weights=previous, incumbent=initial,
                             prepared_cache=systems)
        solve_calls += 1
        if short_circuit and reject_failed_base and not passes(base.diagnostics):
            # EVERY base/expanded solve must pass. An already rejected base
            # cannot be rescued by its expansion. Leave that uncomputed result
            # explicitly unknown, never invent its objective or a zero label.
            comparisons.append(dict(base=base.diagnostics, expanded=None, increment=None))
            break
        candidate = solve_library(weights, owners, extended, checked, previous_weights=previous,
                                  incumbent=base.amplitudes, prepared_cache=systems)
        solve_calls += 1
        gain = base.diagnostics['objective_after']-candidate.diagnostics['objective_after']
        comparisons.append(dict(base=base.diagnostics, expanded=candidate.diagnostics, increment=gain))
        # Acceptance requires BOTH iteration budgets to pass. Once the first
        # pair fails, the label is already unknown regardless of the second.
        # Do not relax tolerances, substitute zero, or claim unrun diagnostics.
        if short_circuit and multiplier == 1 and not all(map(passes, (base.diagnostics, candidate.diagnostics))):
            break
    statuses = [status for comparison in comparisons for status in
                (comparison['base'], comparison['expanded']) if status is not None]
    residual_ok = all(map(passes, statuses))
    increments = [comparison['increment'] for comparison in comparisons if comparison['increment'] is not None]
    finite = all(math.isfinite(value) for value in increments)
    complete = solve_calls == 4
    stable = complete and finite and abs(increments[1]-increments[0]) <= config.access_label_stability
    monotone = bool(increments) and finite and min(increments) >= -1e-10
    reliable = residual_ok and stable and monotone
    target = max(0., increments[-1]) if reliable else None
    reason = ('accepted' if reliable else 'solver_quality' if not residual_ok else
              'nonfinite_or_unstable_increment' if not stable else 'incumbent_objective_violation')
    return dict(kind=audit['kind'], target=target, reliable=reliable, reason=reason,
        comparisons=comparisons, solve_calls=solve_calls, fine_evaluations=0,
        short_circuited=not complete, unrun_solve_calls=4-solve_calls,
        quality_gate_unchanged=True,
        prior_library_count=p, same_surrogate_and_cost=True,
        curvature_reassembled_from_frozen_terms_float64=True,
        audit_warm_start=warm_start,
        exact_optimum_claimed=False, target_scope='actual_access_quantum_conditional_joint_surrogate',
        increment_stability=abs(increments[1]-increments[0]) if complete and finite else None)


def access_targets(search, *, replay_cache=None, replay_key=None, reject_failed_base=False):
    labels = []
    for audit in search['audits']:
        def compute():
            # Keep the reference call signature unchanged when not opted in.
            return (audit_target(search, audit, reject_failed_base=True) if reject_failed_base
                    else audit_target(search, audit))
        if replay_cache is None:
            result = compute()
        else:
            if replay_key is None:
                raise ValueError('audit cache requires an explicit dated replay key')
            result = replay_cache.resolve((*replay_key, audit['ordinal']),
                audit_context(search, audit), compute)
        heads = search['config']['access_heads']
        # Deterministic bootstrap masks diversify the heads without changing
        # global RNG/replay state. Disagreement is only a heuristic priority.
        seed = context_seed(search['exploration_seed'], audit['ordinal'], audit['plan']['slot'])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        mask = torch.rand(heads, generator=generator) < .8
        if not mask.any():
            mask[seed % heads] = True
        labels.append(dict(audit=audit, diagnostic=result, head_mask=mask))
    return labels
