"""Differentiable finite-iteration, coupled repair solver (not an exact oracle).

min 1/2 a' Q a - g' a + fee ||w + V a - w_previous||_1
s.t. 0 <= a <= 1, R a <= abs(w), R = max(-owner * V, 0).

Every column of V is a same-side self-financing repair. Nonnegative resource
constraints conservatively prevent overlapping plans from spending the same
holding twice. ADMM is unrolled; a final radial feasible polish and a comparison
to a=0 keep the returned book feasible and no worse under THIS learned convex
surrogate. Neither future utility nor exact optimality is guaranteed.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass
class RepairSolution:
    amplitudes: Tensor
    weights: Tensor
    diagnostics: dict


class RepairSystemCache:
    """No-gradient, audit-local systems; strong references prevent ID reuse.

    Iteration count, tolerance and incumbent are deliberately NOT reused: each
    solve starts its own ADMM sequence and rechecks the incumbent. Tensor version
    changes invalidate a system. Never use this cache for a differentiable solve.
    """
    def __init__(self):
        self.entries = {}
        self.hits = self.misses = 0

    def resolve(self, tensors, settings, compute):
        if torch.is_grad_enabled():
            raise ValueError('prepared repair cache is restricted to no-gradient audits')
        token = tuple(None if t is None else (id(t), t._version, t.shape, t.dtype, t.device)
                      for t in tensors), tuple(settings)
        if token in self.entries:
            self.hits += 1
            return self.entries[token][1]
        self.misses += 1
        value = compute()
        self.entries[token] = (tensors, value)
        return value


def _repair_system(weights, directions, owners, gains, curvature, previous,
                   plan_slots, fee, rho, allow_semidefinite):
    n, p = directions.shape
    if not torch.allclose(curvature, curvature.T, atol=1e-6, rtol=1e-5):
        raise ValueError('curvature must be symmetric')
    _, pd = torch.linalg.cholesky_ex(curvature.detach())
    if int(pd) != 0:
        numeric_slack = (32*torch.finfo(weights.dtype).eps*p
                         * curvature.detach().abs().amax().clamp_min(torch.finfo(weights.dtype).tiny))
        if not allow_semidefinite or torch.linalg.eigvalsh(curvature.detach()).min() < -numeric_slack:
            raise ValueError('curvature must be positive definite (or validated semidefinite)')
    resource = (-owners[:, None] * directions).clamp_min(0)
    capacity = weights.abs()
    live = resource.detach().amax(1) > 0
    if (live & (capacity <= 0)).any():
        raise ValueError('a repair cannot withdraw from an empty holding')
    normalized_resource = resource[live] / capacity[live, None].clamp_min(1e-12)
    slot_rows = weights.new_zeros(0, p) if plan_slots is None else (
        torch.unique(plan_slots, sorted=True)[:, None] == plan_slots[None]).to(weights.dtype)
    capacity_rows = torch.cat([normalized_resource, slot_rows], 0)
    capacity_bounds = weights.new_ones(capacity_rows.shape[0])
    eye = torch.eye(p, device=weights.device, dtype=weights.dtype)
    a_matrix = torch.cat([eye, capacity_rows, directions], 0)
    offset = weights - previous
    scale = torch.stack([gains.detach().abs().max(), curvature.detach().abs().max(),
                         fee * directions.detach().abs().sum(0).max()]).max().clamp_min(1e-5)
    q, g, scaled_fee = curvature / scale, gains / scale, fee / scale
    chol = torch.linalg.cholesky(q + rho * (a_matrix.T @ a_matrix))
    return resource, capacity, slot_rows, capacity_rows, capacity_bounds, a_matrix, offset, g, scaled_fee, chol


def solve_repairs(weights: Tensor, directions: Tensor, owners: Tensor,
                  gains: Tensor, curvature: Tensor, *, previous_weights=None,
                  fee=0.0005, iterations=48, rho=0.5, tolerance=1e-4,
                  plan_slots=None, initial_amplitudes=None, allow_semidefinite=False,
                  prepared_cache=None):
    """Solve one date/book. Float32/64 only; gradients flow through iterations.

    ``previous_weights`` must be the drifted, pre-trade actual book if supplied.
    If omitted, only incremental repair cost is modeled, explicitly recorded.
    """
    n, p = directions.shape
    if (weights.shape != (n,) or owners.shape != (n,) or gains.shape != (p,)
            or curvature.shape != (p, p)):
        raise ValueError("inconsistent repair problem shapes")
    if weights.dtype not in (torch.float32, torch.float64):
        raise ValueError("repair solve requires float32/64, not low precision")
    tensors = (weights, directions, owners, gains, curvature)
    if any(x.device != weights.device or x.dtype != weights.dtype for x in tensors):
        raise ValueError("repair tensors must be finite and share dtype/device")
    if not torch.stack([torch.isfinite(x).all() for x in tensors]).all():
        raise ValueError("repair tensors must be finite and share dtype/device")
    valid = [torch.all((owners == 1) | (owners == -1)), ~(owners * weights < -1e-7).any()]
    if p:
        valid.extend([directions.sum(0).abs().max() <= 1e-5,
                      (owners[:, None] * directions).sum(0).abs().max() <= 1e-5])
    valid = torch.stack(valid).detach().cpu().tolist()
    if not valid[0]:
        raise ValueError("each asset needs a fixed long/short ownership sign")
    if not valid[1]:
        raise ValueError("book violates fixed ownership")
    if p and not all(valid[2:]):
        raise ValueError("repair directions must conserve net and gross exposure")
    if (type(iterations) is not int or iterations < 1
            or any(isinstance(x, bool) or not math.isfinite(x) for x in (rho, fee, tolerance))
            or rho <= 0 or fee < 0 or tolerance <= 0):
        raise ValueError("invalid solver settings")
    previous = weights if previous_weights is None else previous_weights
    if (previous.shape != weights.shape or previous.dtype != weights.dtype
            or previous.device != weights.device or not torch.isfinite(previous).all()):
        raise ValueError("invalid previous_weights")
    if plan_slots is not None and (plan_slots.shape != (p,) or plan_slots.dtype != torch.long
            or plan_slots.device != weights.device or (plan_slots < 0).any()):
        raise ValueError('invalid plan slot identities')
    if initial_amplitudes is not None and (initial_amplitudes.shape != (p,)
            or initial_amplitudes.dtype != weights.dtype or initial_amplitudes.device != weights.device
            or not torch.isfinite(initial_amplitudes).all()):
        raise ValueError('invalid incumbent amplitudes')
    if type(allow_semidefinite) is not bool:
        raise ValueError('allow_semidefinite must be boolean')
    if not p:
        no_op_cost = float(((weights-previous).abs().sum()*fee).detach())
        return RepairSolution(gains, weights, dict(plans=0, iterations=0,
            primal_residual=0., dual_residual=0., resource_violation=0.,
            feasible=True, converged=True, exact_optimum_claimed=False,
            slot_violation=0., objective_before=no_op_cost,
            objective_after=no_op_cost, objective_incumbent=no_op_cost,
            incumbent_preserved=False, returned_incumbent=False,
            cost_scope="repair_only" if previous_weights is None else "total_trade"))
    def prepare():
        return _repair_system(weights, directions, owners, gains, curvature, previous,
            plan_slots, fee, rho, allow_semidefinite)
    system = (prepare() if prepared_cache is None else prepared_cache.resolve(
        (weights, directions, owners, gains, curvature, previous, plan_slots),
        (fee, rho, allow_semidefinite), prepare))
    resource, capacity, slot_rows, capacity_rows, capacity_bounds, a_matrix, offset, g, scaled_fee, chol = system
    if initial_amplitudes is not None:
        if ((initial_amplitudes < 0).any() or (initial_amplitudes > 1).any()
                or (capacity_rows @ initial_amplitudes > capacity_bounds+1e-6).any()):
            raise ValueError('incumbent is not feasible in the expanded repair library')
        x = initial_amplitudes
    else:
        x = gains * 0
    z = a_matrix @ x
    dual = torch.zeros_like(z)
    n_resource = capacity_rows.shape[0]
    last_z = z
    for _ in range(iterations):
        x = torch.cholesky_solve((g + rho * (a_matrix.T @ (z - dual)))[:, None], chol)[:, 0]
        ax = a_matrix @ x
        u = ax + dual
        transaction = u[p + n_resource:] + offset
        threshold = scaled_fee / rho
        transaction = transaction.sign() * (transaction.abs() - threshold).clamp_min(0) - offset
        last_z = z
        z = torch.cat([u[:p].clamp(0, 1), torch.minimum(u[p:p+n_resource], capacity_bounds), transaction])
        dual = dual + ax - z
    primal = (a_matrix @ x - z).abs().max()
    dual_residual = (rho * a_matrix.T @ (z - last_z)).abs().max()
    # Exact feasibility of the returned amplitudes, independently of convergence.
    clipped = x.clamp(0, 1)
    spent = resource @ clipped
    ratios = torch.where(spent > 0, capacity / spent.clamp_min(1e-12), torch.ones_like(spent))
    shrink = ratios.min().clamp(0, 1)
    if slot_rows.numel():
        shrink = torch.minimum(shrink, (1/(slot_rows @ clipped).clamp_min(1)).min())
    polished = clipped * shrink

    def objective(a):
        return .5 * a @ curvature @ a - gains @ a + fee * (offset + directions @ a).abs().sum()

    zero = gains * 0
    before, proposed = objective(zero), objective(polished)
    incumbent = zero if initial_amplitudes is None else initial_amplitudes
    incumbent_value = objective(incumbent)
    fallback = torch.where(incumbent_value.detach() <= before.detach(), incumbent, zero)
    use_repair = proposed.detach() <= objective(fallback).detach()
    amplitudes = torch.where(use_repair, polished, fallback)
    result = weights + directions @ amplitudes
    violation = (resource @ amplitudes - capacity).clamp_min(0).max()
    slot_violation = ((slot_rows @ amplitudes-1).clamp_min(0).max()
                      if slot_rows.numel() else weights.new_zeros(()))
    returned = ((~use_repair) & (incumbent_value.detach() <= before.detach())
                if initial_amplitudes is not None else use_repair.new_zeros(()))
    raw = (primal.detach() <= tolerance) & (dual_residual.detach() <= tolerance)
    feasible = torch.stack([violation, slot_violation]).max().detach() <= 1e-6
    numbers = torch.stack([x.detach().to(weights.dtype) for x in (
        primal, dual_residual, violation, slot_violation,
        shrink, before, objective(amplitudes), incumbent_value, returned, raw, feasible,
        (amplitudes.detach() != 0).any(), use_repair)]).cpu().tolist()
    pr, dr, rv, sv, sh, ob, oa, oi = numbers[:8]
    returned_incumbent, raw_converged, feasible, used_repair, accepted_new = map(bool,numbers[8:])
    diagnostics = dict(plans=p, iterations=iterations,
        primal_residual=pr, dual_residual=dr,
        converged=raw_converged and not returned_incumbent,
        raw_iterate_converged=raw_converged,
        resource_violation=rv, slot_violation=sv,
        feasible=feasible, feasible_polish_scale=sh, used_repair=used_repair,
        accepted_new_iterate=accepted_new,
        objective_before=ob, objective_after=oa, objective_incumbent=oi,
        incumbent_preserved=initial_amplitudes is not None,
        returned_incumbent=returned_incumbent,
        exact_optimum_claimed=False,
        cost_scope="repair_only" if previous_weights is None else "total_trade")
    return RepairSolution(amplitudes, result, diagnostics)
