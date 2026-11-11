"""Two-wave, solver-conditioned search. No outcome inputs or cross-day cache.

One macro round freezes its book, diagnostic queries and conditional stock states.
The probe solution is provisional. Only the final solution is written back.
Finite-solver utility differences are learned-surrogate targets, NOT true returns
or exact optimality certificates. Integer routing has no implied gradient.
"""
import torch

from .solver import solve_repairs


FEEDBACK_FIELDS = (
    'searched_fraction', 'available_fraction', 'repair_amount', 'remaining_budget_fraction',
    'has_plan', 'accepted_fraction', 'source_utilization', 'source_remaining_fraction',
    'curvature_marginal', 'predicted_gross_gain', 'fee_marginal',
    'primal_residual', 'dual_residual', 'solver_converged',
    'joint_surrogate_improvement', 'current_trade_cost',
)


def library_problem(weights, records, config, *, base_problem=None):
    """Append-only V/g/Q: explicitly preserve the old principal Q block."""
    if records:
        v = torch.stack([r['direction'] for r in records], 1)
        g = torch.stack([r['predicted_gain'] for r in records])
        factors = torch.stack([r['risk_factor'] for r in records])
        slots = torch.tensor([r['slot'] for r in records], device=weights.device, dtype=torch.long)
        slot_rows = (torch.unique(slots, sorted=True)[:, None] == slots[None]).to(weights.dtype)
        # Penalize ACTUAL combined repairs and per-query total amplitudes.
        # Per-version diagonal ridge would reward splitting an identical plan
        # into two copies, creating a fake positive search-value training label.
        q = factors @ factors.T + config.qp_ridge*(v.T @ v)+1e-6*(slot_rows.T @ slot_rows)
        if base_problem is not None:
            p = base_problem['gains'].numel()
            # GEMM shapes may change rounding; the incumbent objective must not.
            q = torch.cat([torch.cat([base_problem['curvature'], q[:p, p:]], 1), q[p:]], 0)
    else:
        v, g, q = weights.new_zeros(weights.numel(), 0), weights.new_zeros(0), weights.new_zeros(0, 0)
        factors = weights.new_zeros(0, config.qp_rank)
        slots = torch.empty(0, dtype=torch.long, device=weights.device)
    return dict(directions=v, gains=g, curvature=q, factors=factors, slots=slots)


def solve_library(weights, owners, problem, config, *, previous_weights=None, incumbent=None,
                  prepared_cache=None):
    if incumbent is not None:
        if incumbent.numel() > problem['gains'].numel():
            raise ValueError('repair library cannot shrink under incumbent protection')
        incumbent = torch.cat([incumbent, weights.new_zeros(problem['gains'].numel()-incumbent.numel())])
    return solve_repairs(weights, problem['directions'], owners, problem['gains'], problem['curvature'],
        previous_weights=previous_weights, fee=config.cost_rate, iterations=config.qp_iterations,
        rho=config.qp_rho, tolerance=config.qp_tolerance, plan_slots=problem['slots'],
        initial_amplitudes=incumbent, allow_semidefinite=True, prepared_cache=prepared_cache)


@torch.no_grad()
def solver_feedback(weights, owners, anchors, amounts, depths, available, problem, solution,
                    config, remaining, previous_weights=None):
    """Observable primal feedback, not unverified shadow prices. Unknown != zero value."""
    v, q, g, slots = (problem[n].detach() for n in ('directions', 'curvature', 'gains', 'slots'))
    accepted = solution.amplitudes.detach()
    used = (-owners[:, None]*v).clamp_min(0) @ accepted
    utilization = used/weights.detach().abs().clamp_min(1e-8)
    previous = weights if previous_weights is None else previous_weights
    trade = solution.weights.detach()-previous.detach()
    marginal_fee = config.cost_rate*(v.T @ trade.sign())
    curvature = q @ accepted
    status = solution.diagnostics
    # Integer membership needs one host transfer, not one GPU synchronization
    # per scalar/slot. Keep the original reduction shapes/order for financial
    # quantities; batching the reductions themselves can change solver routes.
    live_slots = set(slots.detach().cpu().tolist())
    constants = weights.new_tensor([
        [depths[j]/config.retrieval_depths[-1], available[j]/config.retrieval_depths[-1],
         remaining/max(config.fine_edge_budget, 1), float(j in live_slots),
         status['primal_residual'], status['dual_residual'], float(status['converged']),
         (status['objective_before']-status['objective_after'])/config.gain_scale]
        for j in range(len(anchors))])
    zero = weights.new_zeros(())
    trade_cost = config.cost_rate*trade.abs().sum()/config.gain_scale
    rows = []
    for j, anchor in enumerate(anchors):
        mask = slots == j
        known = j in live_slots
        normalizer = (amounts[j].detach()*config.gain_scale).clamp_min(1e-6)
        row = constants[j]
        scalar = torch.stack([
            row[0], row[1], amounts[j].detach(), row[2], row[3], accepted[mask].sum(),
            anchor.detach() @ utilization, anchor.detach() @ (1-utilization).clamp(0, 1),
            torch.tanh(curvature[mask].mean()/normalizer) if known else zero,
            torch.tanh(g[mask].mean()/normalizer) if known else zero,
            torch.tanh(marginal_fee[mask].mean()/normalizer) if known else zero,
            row[4], row[5], row[6], row[7], trade_cost])
        rows.append(scalar)
    global_factor = problem['factors'].detach().T @ accepted
    result = torch.stack(rows)
    if not torch.isfinite(result).all() or not torch.isfinite(global_factor).all():
        raise FloatingPointError('nonfinite solver search feedback')
    return result, global_factor


def predict_actions(head, content, feedback, global_factor, depths, available, config):
    """Predict portfolio-level increments BEFORE seeing the next candidate tranche."""
    metadata, rows = [], []
    ceiling = config.retrieval_depths[-1]
    for slot, old in enumerate(depths):
        for depth in sorted({min(k, available[slot]) for k in config.retrieval_depths if min(k, available[slot]) > old}):
            descriptor = content.new_tensor([depth/ceiling, (depth-old)/ceiling])
            rows.append(torch.cat([content[slot], feedback[slot], global_factor, descriptor]))
            metadata.append(dict(slot=slot, from_depth=old, depth=depth, new_edges=depth-old))
    if not rows:
        return []
    # Amount/capacity are features. No additional multiplication by amount here.
    predictions = config.gain_scale*torch.tanh(head(torch.stack(rows))[:, 0])
    if not torch.isfinite(predictions).all():
        raise FloatingPointError('nonfinite conditional search value')
    return [dict(item, predicted_gain=prediction) for item, prediction in zip(metadata, predictions)]


def _rotation(size, offset):
    # Interleave book sides, then rotate across rounds without mutable/RNG state.
    half = size//2
    order = [x for j in range(half) for x in (j, j+half)]
    return order[offset % size:]+order[:offset % size]


def probe_depths(prior_values, available, config, allowance, round_id):
    counts = [min(config.retrieval_depths[1], count) for count in available]
    minimum = min((n for n in counts if n), default=0)
    budget = min(allowance, max(minimum, allowance//2))
    scores = prior_values.detach().cpu().tolist()
    ranked = sorted((j for j, n in enumerate(counts) if n),
                    key=lambda j: (-scores[j][1]/max(counts[j], 1), j))
    limit = min(config.probe_slots, config.max_plans, len(ranked))
    reserve = min(config.exploration_slots, limit)
    exploitation = [j for j in ranked if scores[j][1] > 0][:limit-reserve]
    explore = [j for j in _rotation(len(available), round_id*config.probe_slots)
               if counts[j] and j not in exploitation][:reserve]
    order = explore+exploitation
    order += [j for j in ranked if j not in order and scores[j][1] > 0]
    depths, used, selected = [0]*len(available), 0, 0
    for j in order:
        if selected == limit:
            break
        if used+counts[j] <= budget:
            depths[j] = counts[j]
            used += counts[j]
            selected += 1
    return depths


def expansion_depths(actions, current, config, budget, round_id):
    """Greedy conditional marginal values; not a global knapsack optimum.

Within the second wave all forecasts condition on the SAME probe library.
Batch-internal interactions are approximated, then resolved by the final solve.
"""
    depths, used = list(current), 0
    values = {(a['slot'], a['depth']): float(a['predicted_gain'].detach()) for a in actions}
    by_slot = {j: sorted(d for (slot, d) in values if slot == j) for j in range(len(current))}
    exploration = []
    for j in _rotation(len(current), (round_id+1)*config.probe_slots):
        if len(exploration) >= config.exploration_slots:
            break
        if depths[j] or not by_slot[j] or sum(d > 0 for d in depths) >= config.max_plans:
            continue
        depth = by_slot[j][0]
        if used+depth <= budget:
            depths[j] = depth
            used += depth
            exploration.append(j)
    while True:
        choices = []
        active = sum(d > 0 for d in depths)
        for j in range(len(current)):
            if depths[j] == 0 and active >= config.max_plans:
                continue
            old_value = 0. if depths[j] == current[j] else values[j, depths[j]]
            for depth in by_slot[j]:
                increment = depth-depths[j]
                if increment <= 0 or used+increment > budget:
                    continue
                priority = (values[j, depth]-old_value)/increment-config.fine_edge_price
                if priority > 0:
                    choices.append((priority, -j, -depth, j, depth, increment))
        if not choices:
            break
        _, _, _, j, depth, increment = max(choices)
        depths[j], used = depth, used+increment
    return depths, exploration


def run_feedback_search(model, diagnosis, weights, owners, anchors, amounts, available,
                        allowance, remaining, round_id, make_plan, previous_weights=None):
    """At most two main solves; sampled counterfactual solves belong to losses.py."""
    config = model.config
    depths = probe_depths(diagnosis['search_values'], available, config, allowance, round_id)
    records = [make_plan(j, depth) for j, depth in enumerate(depths) if depth]
    base_count = len(records)
    base_problem = library_problem(weights, records, config)
    probe = solve_library(weights, owners, base_problem, config, previous_weights=previous_weights)
    used = sum(depths)
    feedback, global_factor = solver_feedback(weights, owners, anchors, amounts, depths, available,
        base_problem, probe, config, remaining-used, previous_weights)
    actions = predict_actions(model.conditional_search_value, diagnosis['content'], feedback,
                              global_factor, depths, available, config)
    final_depths, exploration = expansion_depths(actions, depths, config, allowance-used, round_id)
    selected = []
    for slot, depth in enumerate(final_depths):
        if depth > depths[slot]:
            action = next(a for a in actions if a['slot'] == slot and a['depth'] == depth)
            selected.append(dict(action, record_index=len(records)))
            records.append(make_plan(slot, depth))
    if selected:
        problem = library_problem(weights, records, config, base_problem=base_problem)
        solution = solve_library(weights, owners, problem, config, previous_weights=previous_weights,
                                 incumbent=probe.amplitudes)
    else:
        problem, solution = base_problem, probe
    # Re-estimate remaining search opportunity AFTER the final solve. This is
    # cheap head evaluation, not another candidate scan/solve or an optimality bound.
    with torch.no_grad():
        final_feedback, global_final = solver_feedback(weights, owners, anchors, amounts, final_depths,
            available, problem, solution, config, remaining-sum(final_depths), previous_weights)
        remaining_actions = predict_actions(model.conditional_search_value, diagnosis['content'],
            final_feedback, global_final, final_depths, available, config)
        opportunity = sum(max([0.]+[float(a['predicted_gain']) for a in remaining_actions if a['slot'] == j])
                          for j in range(config.diagnostic_slots))
    auxiliary = dict(weights=weights, owners=owners, previous_weights=previous_weights,
        base_count=base_count, base_problem=base_problem, base_amplitudes=probe.amplitudes.detach(),
        problem=problem, actions=selected, records=records, round_id=round_id, config=config.to_dict())
    trace = dict(policy=config.search_policy, main_solver_calls=1+bool(selected),
        probe_depths=depths, final_depths=final_depths, probe_fine_edges=used,
        expansion_fine_edges=sum(final_depths)-used, fine_edges=sum(final_depths),
        fine_cache_hits=sum(r['indices'].numel() for r in records)-sum(final_depths),
        plan_versions=len(records), unique_plan_slots=len({r['slot'] for r in records}),
        max_versions_per_slot=max([0]+[sum(r['slot'] == j for r in records) for j in range(config.diagnostic_slots)]),
        exploration_slots=exploration, feedback_fields=FEEDBACK_FIELDS,
        probe_feedback=feedback.cpu().tolist(), final_feedback=final_feedback.cpu().tolist(),
        probe_solver=probe.diagnostics,
        expansion_actions=[dict(slot=a['slot'], from_depth=a['from_depth'], depth=a['depth'],
            new_edges=a['new_edges'], predicted_gain=float(a['predicted_gain'].detach())) for a in selected],
        remaining_opportunity_proxy=opportunity,
        source_book_frozen_within_round=True, queries_frozen_within_round=True,
        incumbent_objective_comparable=True, cross_round_fine_cache_reused=False)
    return dict(records=records, solution=solution, problem=problem, depths=final_depths,
                spent=sum(final_depths), opportunity=opportunity, trace=trace, auxiliary=auxiliary)


@torch.no_grad()
def conditional_targets(search):
    """Sampled, solver-conditioned labels without market outcomes or new fine work.

Each selected expansion is compared with the SAME pre-expansion library, not
with a hindsight context containing its own new evidence. Labels are approximate
achieved surrogate improvements; residuals are reported, not called exact optima.
"""
    from .config import DQConfig
    config = DQConfig.from_dict(search['config'])
    actions = search['actions']
    if not actions or not config.counterfactual_queries:
        return []
    start = search['round_id'] % len(actions)
    selected = (actions[start:]+actions[:start])[:config.counterfactual_queries]
    base_count, full = search['base_count'], search['problem']
    w, owner, incumbent = search['weights'].detach(), search['owners'].detach(), search['base_amplitudes'].detach()
    previous = search['previous_weights']
    previous = None if previous is None else previous.detach()
    base = {name: tensor.detach() for name, tensor in search['base_problem'].items()}
    # A matched no-search numerical control gets the same extra iterations.
    # Otherwise merely solving the OLD library again could be mislabeled as
    # an information gain from the new candidates.
    control = solve_library(w, owner, base, config, previous_weights=previous, incumbent=incumbent)
    labels = []
    for action in selected:
        indices = torch.tensor([*range(base_count), action['record_index']], device=w.device)
        problem = dict(directions=full['directions'].detach()[:, indices],
            gains=full['gains'].detach()[indices], curvature=full['curvature'].detach()[indices][:, indices],
            slots=full['slots'][indices], factors=full['factors'].detach()[indices])
        solution = solve_library(w, owner, problem, config, previous_weights=previous, incumbent=incumbent)
        value = control.diagnostics['objective_after']-solution.diagnostics['objective_after']
        duplicate = any(
            bool(problem['slots'][-1] == problem['slots'][j])
            and torch.equal(problem['directions'][:, -1], problem['directions'][:, j])
            and torch.equal(problem['gains'][-1], problem['gains'][j])
            and torch.equal(problem['curvature'][-1], problem['curvature'][j])
            for j in range(base_count))
        if duplicate:
            value = 0.  # equivalent data cannot earn value through a different solver trajectory
        labels.append(dict(action=action, target=w.new_tensor(max(value, 0.)),
            solver=solution.diagnostics, no_search_control=control.diagnostics,
            duplicate_plan=duplicate, new_fine_edges=0,
            target_semantics='finite_solver_feasible_surrogate_increment_not_realized_return'))
    return labels
