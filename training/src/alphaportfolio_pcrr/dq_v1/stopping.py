"""Deterministic, same-decision stopping; no new predictor or outcome inputs.

This is a budget/stagnation policy, not a certificate of financial optimality.
Statistics are detached, but the actual recurrent state/book graph is not.
"""
import math

import torch


@torch.no_grad()
def relative_change(before, after, mask=None):
    """Maximum per-row relative L2 change, not diluted by untouched stocks."""
    if mask is not None:
        before, after = before[mask], after[mask]
    if before.numel() == 0:
        return 0.
    delta = torch.linalg.vector_norm(after-before, dim=-1)
    scale = torch.linalg.vector_norm(before, dim=-1).clamp_min(1e-8)
    return float((delta/scale).max())


@torch.no_grad()
def stopping_metrics(*, before_weights, after_weights, before_state, after_state,
                     touched, diagnosis, previous_diagnosis, search_values,
                     available, gains, solver, previous_weights, config, search_gain_override=None):
    # Use the same small-universe interpolation as the discrete planner. These
    # are learned proxies for the CURRENT retrieved set, not unseen-stock bounds
    # or calibrated forecasts of the next round's realized return.
    depths = search_values.new_tensor(config.retrieval_depths[1:])
    counts = search_values.new_tensor(available)[:, None]
    fraction = torch.minimum(counts, depths)/depths
    search_gain = float((search_values[:, 1:]*fraction).clamp_min(0).amax(1).sum())
    if search_gain_override is not None:
        search_gain = float(search_gain_override)
    before_cost = (0. if previous_weights is None else
                   config.cost_rate*float((before_weights-previous_weights).abs().sum()))
    # Q is PSD and 0 <= amplitudes <= 1. This conservative upper bound is only
    # for the CURRENT learned repair problem, including possible cost savings.
    repair_bound = float(gains.clamp_min(0).sum())+before_cost if gains.numel() else 0.
    improvement = max(0., solver.get('objective_before', 0.)-solver.get('objective_after', 0.))
    metrics = dict(
        weight_change_l1=float((after_weights-before_weights).abs().sum()),
        state_change_relative_max=relative_change(before_state, after_state, touched),
        diagnostic_change_relative_max=(None if previous_diagnosis is None else
                                        relative_change(previous_diagnosis, diagnosis)),
        predicted_search_gain_proxy=search_gain,
        predicted_repair_gain_bound=repair_bound,
        current_surrogate_improvement=improvement,
        predicted_opportunity_proxy=max(search_gain, repair_bound),
        solver_converged=bool(solver['converged']),
        solver_feasible=bool(solver['feasible']),
    )
    if any(value is not None and not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('nonfinite refinement stopping metric')
    return metrics


class RefinementStopper:
    """One fresh controller per decision/book, shared policy in train and eval."""
    def __init__(self, config):
        self.config = config
        self.stable_rounds = 0

    def observe(self, metrics, *, completed_rounds, remaining, available, minimum_increment):
        c = self.config
        # A missing first-round diagnostic comparison is not evidence of drift.
        # Default patience=2 still requires two stable state/book updates and a
        # real diagnostic comparison before a stagnation exit can occur.
        drift = metrics['diagnostic_change_relative_max']
        stable = (metrics['weight_change_l1'] <= c.stop_weight_l1_tolerance
                  and metrics['state_change_relative_max'] <= c.stop_state_relative_tolerance
                  and (drift is None or drift <= c.stop_diagnostic_relative_tolerance)
                  and metrics['predicted_opportunity_proxy'] <= c.stop_gain_tolerance
                  and metrics['solver_converged'] and metrics['solver_feasible'])
        self.stable_rounds = self.stable_rounds+1 if stable else 0
        reason = None
        if c.stopping_mode == 'fixed':
            if completed_rounds == c.rounds:
                reason = 'fixed_rounds_completed'
        elif remaining == 0:
            reason = 'fine_budget_exhausted'
        elif not any(available):
            # In grouped mode this means no candidates were retrieved, NOT that
            # no useful stocks exist anywhere in the universe.
            reason = 'no_retrieved_candidates'
        elif remaining < minimum_increment:
            reason = 'insufficient_current_search_budget'
        elif self.stable_rounds >= c.stop_patience:
            reason = 'stable_low_predicted_gain'
        elif completed_rounds == c.rounds:
            reason = 'max_rounds_reached'
        return dict(**metrics, stable_this_round=bool(stable),
                    consecutive_stable_rounds=self.stable_rounds,
                    stopped=reason is not None, stop_reason=reason)
