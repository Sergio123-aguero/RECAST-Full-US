"""Training-only supervision. No outcome is accepted by the forward model.

Ranking compares alternatives with the SAME source holding and repair amount:
the benefit of merely selling a bad source cancels. Budget targets only cover
actually retrieved alternatives; they are not unbiased full-universe labels.
Financial task loss consumes a net-return path supplied by the execution ledger,
not a frictionless reconstruction masquerading as executable net performance.
"""
import math

import torch
from torch.nn import functional as F

from .search import conditional_targets
from .access_supervision import access_targets


def net_path_objective(net_returns, *, sharpe_weight=1., growth_weight=.25, annualization=252):
    if (net_returns.ndim != 1 or net_returns.numel() < 2 or not torch.isfinite(net_returns).all()
            or (net_returns <= -1).any()):
        raise ValueError("a finite net-return path of length >= 2, returns > -1, is required")
    if (not math.isfinite(sharpe_weight) or not math.isfinite(growth_weight)
            or sharpe_weight < 0 or growth_weight < 0 or annualization <= 0):
        raise ValueError("invalid objective settings")
    sharpe = math.sqrt(annualization)*net_returns.mean()/net_returns.std(unbiased=False).clamp_min(1e-6)
    log_growth = annualization*torch.log1p(net_returns).mean()
    return dict(loss=-sharpe_weight*sharpe-growth_weight*log_growth,
                sharpe=sharpe, annualized_log_growth=log_growth)


@torch.no_grad()
def contribution_targets(contribution, outcomes, observed, *, cost_rate):
    """Local log-wealth counterfactuals, NOT a Sharpe/Shapley attribution.

    Remove a fixed fraction of a held position and redistribute on its own side,
    or insert an unheld stock by funding it from that side's existing holdings.
    All other holdings stay fixed; side budgets and gross exposure are conserved.
    The alternative pays a two-way linear *repair* fee. This does not replace
    the actual full-path execution ledger's turnover, financing or carry costs.
    Invalid/unknown wealth marks produce no label rather than a zero label.
    """
    w = contribution['weights'].detach()
    owner, amount = contribution['owner'].detach(), contribution['amount'].detach()
    observed = observed.to(torch.bool)
    safe = outcomes.detach().masked_fill(~observed, 0)
    held = contribution['held']
    long_total = (w.clamp_min(0)*safe).sum()
    short_total = ((-w).clamp_min(0)*safe).sum()
    reference = (torch.where(owner > 0, long_total, short_total)-w.abs()*safe)
    reference = reference/contribution['reference_mass'].detach().clamp_min(1e-8)
    base_return = (w*safe).sum()
    directional_gain = owner*amount*(safe-reference)
    alternative = base_return+torch.where(held, -directional_gain, directional_gain)-2*cost_rate*amount
    valid = (contribution['available'] & observed & observed[held].all()
             & (amount > 1e-8) & (base_return > -1) & (alternative > -1))
    base_log = torch.log1p(torch.where(base_return > -1, base_return, torch.zeros_like(base_return)))
    alternative_log = torch.log1p(alternative.masked_fill(~valid, 0))
    delta = torch.where(held, base_log-alternative_log, alternative_log-base_log)
    target = (delta/amount.clamp_min(1e-8)).masked_fill(~valid, 0)
    return target, valid


def auxiliary_loss(result, future_returns, *, future_observed=None, ranking_temperature=.01, cost_rate=.0005,
                   gain_scale=.02, ranking_weight=.1, grounding_weight=.02,
                   value_weight=.05, diversity_weight=.001, contribution_weight=.05,
                   conditional_search_weight=.05, access_value_weight=.05,
                   access_replay_cache=None, access_replay_key=None, reject_failed_base_audit=False):
    """Consume TRAIN split outcomes only; caller must enforce dated split guards.

    Diagnostics stay grounded in causal observed stock statistics. Counterfactual
    one-period marginal labels include a conservative two-way repair charge.
    They do not model all multi-period holdings or prove that a repair is causal.
    """
    if hasattr(result, "dq_result"):
        result = result.dq_result
    if (future_returns.shape != result.weights.shape or future_returns.dtype != result.weights.dtype
            or future_returns.device != result.weights.device):
        raise ValueError("invalid training-only asset returns")
    if future_observed is None:
        future_observed = torch.ones_like(future_returns, dtype=torch.bool)
    if (future_observed.shape != future_returns.shape or future_observed.dtype != torch.bool
            or future_observed.device != future_returns.device
            or not torch.isfinite(future_returns[future_observed]).all()):
        raise ValueError("invalid training outcome observability")
    if not any(result.auxiliary):
        raise ValueError("forward(return_aux=True) required for auxiliary training")
    if ranking_temperature <= 0 or gain_scale <= 0 or cost_rate < 0:
        raise ValueError("invalid auxiliary scaling")
    zero = result.weights.sum()*0
    rank_terms, ground_terms, value_terms, diversity_terms, contribution_terms = [], [], [], [], []
    conditional_terms, conditional_solver_diagnostics = [], []
    conditional_controls = []
    access_terms, access_diagnostics = [], []
    skipped_value_queries, supervised_contributions = 0, 0
    for b, rounds in enumerate(result.auxiliary):
        observed_outcome = future_observed[b]
        outcome = future_returns[b].detach().masked_fill(~observed_outcome, 0)
        for round_index, entry in enumerate(rounds):
            if 'access_search' in entry:
                key = None if access_replay_key is None else (*access_replay_key, b, round_index)
                for label in access_targets(entry['access_search'],
                        replay_cache=access_replay_cache, replay_key=key,
                        reject_failed_base=reject_failed_base_audit):
                    diagnostic = label['diagnostic']
                    access_diagnostics.append(diagnostic)
                    if not diagnostic['reliable']:
                        continue
                    for predicted in label['audit']['predictions']:
                        mask = label['head_mask'].to(predicted.device)
                        target = predicted.new_full(predicted[mask].shape, diagnostic['target'])
                        access_terms.append(F.smooth_l1_loss(predicted[mask]/gain_scale, target/gain_scale))
            if 'conditional_search' in entry:
                search_labels = conditional_targets(entry['conditional_search'])
                if search_labels:
                    conditional_controls.append(search_labels[0]['no_search_control'])
                for label in search_labels:
                    action = label['action']
                    record = entry['fine'][action['record_index']]
                    scale = (record['amount'].detach()*gain_scale).clamp_min(1e-6)
                    conditional_terms.append(F.smooth_l1_loss(action['predicted_gain']/scale,
                                                             label['target']/scale))
                    conditional_solver_diagnostics.append(label['solver'])
            contribution = entry['contribution']
            if not contribution.get('contribution_supervision_disabled', False):
                targets, known_contribution = contribution_targets(contribution, outcome, observed_outcome,
                                                                       cost_rate=cost_rate)
                if known_contribution.any():
                    contribution_terms.append(F.smooth_l1_loss(
                        contribution['predicted_per_unit'][known_contribution]/gain_scale,
                        targets[known_contribution]/gain_scale))
                    supervised_contributions += int(known_contribution.sum())
            diag = entry["diagnosis"]
            known = diag["known_statistics"]
            ground_terms.append(((diag["predicted_statistics"]-diag["grounded_statistics"]).square()*known).sum()
                                / known.sum().clamp_min(1))
            pool = F.normalize(diag["pool"], dim=-1)
            gram = pool @ pool.T
            eye = torch.eye(pool.shape[0], device=pool.device, dtype=torch.bool)
            diversity_terms.append(gram.masked_select(~eye).square().mean())
            for record in entry["fine"]:
                idx, slot = record["indices"], record["slot"]
                target_quality = record["sign"]*outcome[idx]
                known_targets = observed_outcome[idx]
                if int(known_targets.sum()) >= 2:
                    target = torch.softmax(target_quality[known_targets]/ranking_temperature, 0).detach()
                    rank_terms.append(-(target*F.log_softmax(record["logits"][known_targets], 0)).sum()
                                      -(target*F.log_softmax(record["coarse_logits"][known_targets], 0)).sum())
                # Unknown marks never become zero-return labels. Conservative
                # value supervision requires the entire realized repair known.
                relevant = (record['anchor'].detach() > 0).clone()
                relevant[idx] = True
                if not observed_outcome[relevant].all():
                    skipped_value_queries += 1
                    continue
                amount = record["amount"].detach().clamp_min(1e-8)
                gross_gain = (record["direction"].detach()*outcome).sum()
                realized = gross_gain - 2*cost_rate*amount
                # The QP charges transaction cost itself. Learning a net gain
                # here would deduct that cost twice inside the decision layer.
                normalized_gain = gross_gain/(amount*gain_scale)
                value_terms.append(F.smooth_l1_loss(record["predicted_gain"]/(amount*gain_scale), normalized_gain))
                source_return = record["sign"]*(record["anchor"].detach()*outcome).sum()
                depths = result.diagnostics.get("retrieval_depths")
                if depths is None:
                    raise ValueError("missing declared retrieval depths")
                supervised_values = []
                for level, depth in enumerate(depths[1:] if diag['legacy_depth_prior'] else [], 1):
                    if depth > idx.numel():
                        continue
                    benefit = (target_quality[:depth].max()-source_return-2*cost_rate).clamp_min(0)
                    predicted = diag["curves"][slot, level]
                    supervised_values.append(F.smooth_l1_loss(predicted/gain_scale, benefit/gain_scale))
                if not supervised_values and diag['legacy_depth_prior']:
                    benefit = (target_quality.max()-source_return-2*cost_rate).clamp_min(0)
                    fraction = idx.numel()/depths[1]
                    supervised_values.append(F.smooth_l1_loss(diag["curves"][slot, 1]*fraction/gain_scale,
                                                             benefit/gain_scale))
                value_terms.extend(supervised_values)
                intent_target = (realized > 0).to(realized.dtype)
                value_terms.append(F.binary_cross_entropy_with_logits(diag["intent"][slot], intent_target))

    def mean(values):
        return torch.stack(values).mean() if values else zero

    rank, ground, value, diversity = map(mean, (rank_terms, ground_terms, value_terms, diversity_terms))
    contribution = mean(contribution_terms)
    conditional = mean(conditional_terms)
    access_value = mean(access_terms)
    total = (ranking_weight*rank + grounding_weight*ground + value_weight*value
             + diversity_weight*diversity + contribution_weight*contribution
             + conditional_search_weight*conditional + access_value_weight*access_value)
    return dict(loss=total, ranking=rank, grounding=ground, search_and_repair_value=value,
                slot_diversity=diversity, contribution=contribution,
                conditional_search_value=conditional,
                access_value=access_value, access_audit_diagnostics=access_diagnostics,
                access_audited_actions=len(access_diagnostics),
                access_labeled_actions=sum(x['reliable'] for x in access_diagnostics),
                access_skipped_unreliable_actions=sum(not x['reliable'] for x in access_diagnostics),
                access_resolve_calls=sum(x['solve_calls'] for x in access_diagnostics),
                access_unseen_targets_are_negative=False,
                conditional_resolve_calls=len(conditional_solver_diagnostics)+len(conditional_controls),
                conditional_labeled_actions=len(conditional_solver_diagnostics),
                conditional_solver_diagnostics=conditional_solver_diagnostics,
                conditional_no_search_controls=conditional_controls,
                conditional_target='achieved_feasible_surrogate_increment_not_exact_return',
                supervised_stock_contributions=supervised_contributions, supervised_fine_queries=len(rank_terms),
                skipped_value_queries_missing_outcomes=skipped_value_queries)
