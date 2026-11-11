"""Shared summaries -> grounded diagnostic queries -> budgeted coupled repairs.

No stock-by-stock N x N attention is formed, either before or during feedback.
Summary/diagnostic pooling still reads the whole universe: O(N*C + N*K), not
free retrieval. The fine-edge budget does NOT bound encoding, index building,
coarse scans, or the small-plan solver. Counters report those separately.

All inputs describe the decision time. Future returns belong only in losses.py.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DQConfig
from .solver import solve_repairs
from .stopping import RefinementStopper, stopping_metrics
from .search import FEEDBACK_FIELDS, run_feedback_search
from .hierarchy import SemanticIndex, AccessBudget, AccessValue
from .access import AccessRound


@dataclass
class DQResult:
    weights: Tensor                 # canonical FINAL book; optional unused side budget stays in cash
    initial_weights: Tensor
    proposal_scores: Tensor        # not a ranking that reconstructs final weights
    relation: Tensor
    long_budget: Tensor
    auxiliary: list
    diagnostics: dict
    initial_relation: Tensor      # immutable same-day index/base representation


def _ordered(indices, values, ids, *, descending=True):
    """Stable score order, with permanent identifiers breaking ties."""
    indices = indices[torch.argsort(ids[indices], stable=True)]
    return indices[torch.argsort(values[indices], descending=descending, stable=True)]


def _masked_softmax(logits, mask, dim=-1, *, allow_empty=False):
    populated = mask.any(dim, keepdim=True)
    if not populated.all() and not allow_empty:
        raise ValueError("empty masked distribution")
    masked = logits.masked_fill(~mask, -torch.inf)
    if allow_empty:
        # Softmax(-inf,...,-inf) is NaN. An absent book has zero source mass,
        # not a fabricated uniform source distribution.
        masked = torch.where(populated, masked, torch.zeros_like(masked))
        return torch.softmax(masked, dim=dim)*populated.to(logits.dtype)
    return torch.softmax(masked, dim=dim)


def allocate_depths(values: Tensor, available: list[int], config: DQConfig, budget: int):
    """Discrete incremental value-per-edge planner; no implied routing gradient.

    Values are cumulative predicted improvements at the configured depths.
    A slot with negative incremental value may abstain. The auxiliary losses
    train values; selected fine relations carry the portfolio gradient.
    """
    depths = config.retrieval_depths
    table = values.detach().cpu().tolist()
    levels = [0] * len(available)
    used = 0
    while True:
        choices = []
        active = sum(x > 0 for x in levels)
        for q, level in enumerate(levels):
            if level + 1 == len(depths) or (level == 0 and active >= config.max_plans):
                continue
            old, new = min(depths[level], available[q]), min(depths[level+1], available[q])
            increment = new - old
            if increment <= 0 or used + increment > budget:
                continue
            # Interpolate the cumulative value when a small universe truncates
            # the last depth, rather than charge for uncomputed edges.
            old_value = table[q][level] * old / max(depths[level], 1)
            new_value = table[q][level+1] * new / depths[level+1]
            priority = (new_value - old_value) / increment - config.fine_edge_price
            if priority > 0:
                choices.append((priority, -q, q, increment))
        if not choices:
            break
        _, _, q, increment = max(choices)
        levels[q] += 1
        used += increment
    return [min(depths[level], count) for level, count in zip(levels, available)], used


class DiagnosticQueryPCRR(nn.Module):
    """Decision-conditioned learned slots with fixed long/short ownership.

    Slot contents, source evidence, retention intent, search values and queries
    are learned. Half the slots address each book side; this is a feasibility
    type, not a claim that the slots discover named economic risk factors.
    Final support may expand beyond the initial hard top-fraction portfolio.
    """
    def __init__(self, dimension: int, config=None, *, long_fraction=.1,
                 short_fraction=.1, budget_hidden_dim=32, decision_mode='full'):
        super().__init__()
        if decision_mode not in {'full', 'candidate_only'}:
            raise ValueError('unknown DQ decision mode')
        self.decision_mode = decision_mode
        self.config = (config or DQConfig()).validate()
        if type(dimension) is not int or dimension < 1:
            raise ValueError("dimension must be positive")
        if type(budget_hidden_dim) is not int or budget_hidden_dim < 1:
            raise ValueError("budget_hidden_dim must be a positive integer")
        if not (0 < long_fraction < .5 and 0 < short_fraction < .5):
            raise ValueError("basket fractions must lie in (0, .5)")
        self.dimension = d = dimension
        self.long_fraction, self.short_fraction = long_fraction, short_fraction
        self.budget_hidden_dim = budget_hidden_dim
        c, r, s = self.config, self.config.key_dim, self.config.statistic_dim
        self.summary_tokens = nn.Parameter(torch.randn(c.summaries, r) / math.sqrt(r))
        self.summary_key = nn.Linear(d, r, bias=False)
        self.summary_value = nn.Linear(d, d, bias=False)
        self.summary_read = nn.Linear(d, r, bias=False)
        self.summary_output = nn.Linear(d, d, bias=False)
        self.proposal = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.budget_head = nn.Sequential(nn.Linear(3*d, budget_hidden_dim), nn.Tanh(),
                                         nn.Linear(budget_hidden_dim, 1))
        if self.decision_mode == 'candidate_only':
            # A genuinely smaller trained model, not a full refinement run with
            # its output discarded. Shared candidate tensors retain their exact
            # constructor order/initialization relative to the full model.
            return
        self.diagnostic_tokens = nn.Parameter(torch.randn(c.diagnostic_slots, d) / math.sqrt(d))
        self.diagnostic_key = nn.Linear(d, d, bias=False)
        self.book_condition = nn.Linear(2*d + 4, d)
        # A contribution is conditioned on the *current* book, not a stock-only
        # forecast. Its explicit counterfactual is specified in losses.py.
        self.stock_context = nn.Sequential(nn.Linear(5*d + 12 + 2*s, d), nn.GELU(), nn.Linear(d, d))
        self.contribution_head = nn.Linear(d, 1)
        self.contribution_read = nn.Linear(3, d, bias=False)
        self.feedback_message = nn.Sequential(nn.Linear(3*d + 4, d), nn.GELU(), nn.Linear(d, d))
        self.feedback_update = nn.Sequential(nn.Linear(3*d + 2, d), nn.GELU(), nn.Linear(d, d))
        self.feedback_gate = nn.Linear(3*d + 2, d)
        self.diagnose = nn.Sequential(nn.Linear(3*d + 2*s, d), nn.GELU(), nn.Linear(d, d))
        self.ground_statistics = nn.Linear(d, s)
        self.intent = nn.Linear(d, 1)
        if c.access_policy == 'legacy_v1':
            self.search_value = nn.Linear(d, 2)
        self.query_projection = nn.Linear(d, r, bias=False)
        self.retrieval_key = nn.Linear(d, r, bias=False)
        self.fine_key = nn.Linear(d, r, bias=False)
        self.fine_value = nn.Linear(d, d, bias=False)
        self.holding_key = nn.Linear(4, r, bias=False)
        self.holding_value = nn.Linear(4, d, bias=False)
        self.fine_query = nn.Linear(2*d, r, bias=False)
        self.marginal_score = nn.Sequential(nn.Linear(2*d + 4 + 2*s, d), nn.GELU(), nn.Linear(d, 1))
        self.plan_gain = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1))
        self.plan_risk = nn.Linear(2*d, c.qp_rank)
        # Initial search is enabled but not mandatory after learning. These are
        # initialization settings, NOT calibrated financial value estimates.
        with torch.no_grad():
            if hasattr(self, 'search_value'):
                self.search_value.weight.zero_()
                self.search_value.bias.copy_(torch.tensor([.5, 20.]))
            self.intent.bias.fill_(1.)
            self.feedback_gate.bias.fill_(-1.)
        if c.access_policy == 'hierarchical_v1':
            self.access_value = AccessValue(d, c)
        elif c.search_policy == 'solver_feedback_v1':
            self.conditional_search_value = nn.Sequential(
                nn.Linear(d+len(FEEDBACK_FIELDS)+c.qp_rank+2, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.normal_(self.conditional_search_value[-1].weight, std=.01)
            nn.init.constant_(self.conditional_search_value[-1].bias, .001)

    def _store(self, hidden, valid, ids):
        keys = F.normalize(self.summary_key(hidden), dim=-1)
        logits = keys @ F.normalize(self.summary_tokens, dim=-1).T
        assignment = torch.softmax(logits, -1) * valid[..., None]
        summary = assignment.transpose(1, 2) @ self.summary_value(hidden)
        summary = summary / assignment.sum(1)[..., None].clamp_min(1e-8)
        read = torch.softmax(F.normalize(self.summary_read(hidden), dim=-1)
                             @ F.normalize(self.summary_tokens, dim=-1).T, -1)
        relation = (hidden + self.summary_output(read @ summary)) * valid[..., None]
        if self.decision_mode == 'candidate_only':
            return dict(relation=relation, mode='candidate_summary_only')
        key = F.normalize(self.retrieval_key(relation), dim=-1)
        centroids = assignment.transpose(1, 2) @ key
        centroids = F.normalize(centroids, dim=-1)
        mode = self.config.retrieval_mode
        groups = assignment.detach().argmax(-1)
        if self.config.access_policy == 'hierarchical_v1':
            mode = 'hierarchical'
        elif mode == "auto":
            mode = "exact" if hidden.shape[1] <= self.config.exact_scan_threshold else "grouped"
        postings = []
        if mode == "grouped":
            for b in range(hidden.shape[0]):
                row = []
                for g in range(self.config.summaries):
                    idx = torch.where(valid[b] & (groups[b] == g))[0]
                    row.append(idx[torch.argsort(ids[b, idx], stable=True)])
                postings.append(row)
        return dict(relation=relation, key=key, centroids=centroids, mode=mode, postings=postings, groups=groups,
                    fine_key=self.fine_key(relation), fine_value=self.fine_value(relation))

    def _candidate(self, hidden, scores, valid, ids, long_allowed, short_allowed, long_budget):
        owners, unit_longs, unit_shorts = [], [], []
        for b in range(scores.shape[0]):
            eligible = valid[b] & (long_allowed[b] | short_allowed[b])
            idx = _ordered(torch.where(eligible)[0], scores[b], ids[b])
            owner = torch.full_like(scores[b], -1.)
            owner[idx[:(idx.numel()+1)//2]] = 1
            owner = torch.where(long_allowed[b] & ~short_allowed[b], 1., owner)
            owner = torch.where(short_allowed[b] & ~long_allowed[b], -1., owner)
            li = idx[(owner[idx] > 0) & long_allowed[b, idx]]
            si = idx[(owner[idx] < 0) & short_allowed[b, idx]].flip(0)
            li = li[:max(1, int(idx.numel()*self.long_fraction))]
            si = si[:max(1, int(idx.numel()*self.short_fraction))]
            long = torch.zeros_like(scores[b]).scatter(0, li, torch.softmax(scores[b, li], 0))
            short = torch.zeros_like(scores[b]).scatter(0, si, torch.softmax(-scores[b, si], 0))
            unit_longs.append(long)
            unit_shorts.append(short)
            owners.append(owner)
        longs, shorts = torch.stack(unit_longs), torch.stack(unit_shorts)
        if long_budget is None:
            pooled = torch.cat([(hidden*valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1),
                (hidden*longs[..., None]).sum(1), (hidden*shorts[..., None]).sum(1)], -1)
            floor = self.config.minimum_side_budget
            budget = floor + (1-2*floor)*torch.sigmoid(self.budget_head(pooled))
        else:
            if (long_budget.shape != (hidden.shape[0], 1) or long_budget.device != hidden.device
                    or long_budget.dtype != hidden.dtype or not torch.isfinite(long_budget).all()
                    or (long_budget <= 0).any() or (long_budget >= 1).any()):
                raise ValueError("long_budget must be [batch,1] strictly between zero and one")
            budget = long_budget
        return budget*longs - (1-budget)*shorts, torch.stack(owners), budget[:, 0]

    def _book(self, h, w, valid):
        positive, negative = w.clamp_min(0), (-w).clamp_min(0)
        lp, sp = positive @ h / positive.sum().clamp_min(1e-8), negative @ h / negative.sum().clamp_min(1e-8)
        metrics = torch.stack([positive.sum(), negative.sum(), w.square().sum(),
                              ((w.abs() > 1e-10) & valid).to(w.dtype).sum()/valid.sum().clamp_min(1)])
        condition = self.book_condition(torch.cat([lp, sp, metrics]))
        return lp, sp, condition

    def _contributions(self, state, w, owner, valid, allowed, scores, previous_delta,
                       statistics, observed):
        """O(N*d) conditional stock states; no dense leave-one-out matrix.

        Reference: remaining holdings on the same side, proportionally weighted.
        Held stocks are assessed by partial removal; unheld stocks by insertion.
        A sole holding has no removal reference, so that label is unavailable.
        """
        n = w.numel()
        lp, sp, condition = self._book(state, w, valid)
        universe = (state*valid[:, None]).sum(0)/valid.sum().clamp_min(1)
        positive, negative = w.clamp_min(0), (-w).clamp_min(0)
        budget = torch.where(owner > 0, positive.sum(), negative.sum())
        held = w.abs() > 1e-10
        reference_mass = budget-w.abs()
        available = valid & allowed & (reference_mass > 1e-8)
        amount = self.config.repair_fraction*torch.where(held, w.abs(), budget)
        amount = amount*available
        centered = scores-(scores*valid).sum()/valid.sum().clamp_min(1)
        score_scale = ((centered.square()*valid).sum()/valid.sum().clamp_min(1)).clamp_min(1e-12).sqrt()
        scalar = torch.stack([torch.log1p(n*positive), torch.log1p(n*negative),
            (w > 0).to(w.dtype), (w < 0).to(w.dtype), owner,
            w.abs()/budget.clamp_min(1e-8), (centered/score_scale).tanh(),
            (n*previous_delta).tanh(), budget, w.square()/w.square().sum().clamp_min(1e-8),
            available.to(w.dtype), amount], -1)
        features = torch.cat([state, state-lp, state-sp, state-universe,
            condition.expand(n, -1), scalar, statistics, observed.to(w.dtype)], -1)
        context = (state+self.stock_context(features))*valid[:, None]
        predicted = self.config.gain_scale*torch.tanh(self.contribution_head(context)[:, 0])
        predicted = predicted*available
        contribution_input = torch.stack([predicted/self.config.gain_scale, amount,
                                          available.to(w.dtype)], -1)
        representation = (context+self.contribution_read(contribution_input))*valid[:, None]
        return dict(representation=representation, predicted_per_unit=predicted,
            amount=amount, available=available, held=held, owner=owner,
            weights=w, reference_mass=reference_mass,
            semantics="same_side_partial_removal_or_insertion_log_wealth_proxy")

    def _diagnostics(self, h, key, w, owner, valid, statistics, observed):
        c, d = self.config, self.dimension
        _, _, condition = self._book(h, w, valid)
        tokens = self.diagnostic_tokens + condition
        logits = tokens @ key.T / math.sqrt(d)
        all_attention = _masked_softmax(logits, valid[None].expand_as(logits), allow_empty=True)
        signs = torch.ones(c.diagnostic_slots, device=w.device, dtype=w.dtype)
        signs[c.diagnostic_slots//2:] = -1
        held = (owner[None] == signs[:, None]) & (w.abs()[None] > 1e-10) & valid[None]
        source = _masked_softmax(logits + w.abs().clamp_min(1e-12).log()[None], held, allow_empty=True)
        known = source @ observed.to(w.dtype)
        ground = (source @ (statistics*observed)) / known.clamp_min(1e-8)
        content = self.diagnose(torch.cat([tokens, all_attention @ h, source @ h, ground, known], -1))
        intent = self.intent(content)[:, 0]
        if hasattr(self, 'search_value'):
            value_parameters = self.search_value(content)
            depths = w.new_tensor(c.retrieval_depths)[None]
            saturation = 8 + F.softplus(value_parameters[:, 1:2])
            curves = (c.gain_scale * torch.tanh(value_parameters[:, :1]) * torch.sigmoid(intent[:, None])
                      * (1-torch.exp(-depths/saturation)))
        else:
            # New access uses node/stock joint-value heads. No unused trainable
            # exponential depth prior is kept merely to resemble the old path.
            curves = w.new_zeros(c.diagnostic_slots, len(c.retrieval_depths))
        return dict(content=content, source=source, signs=signs, intent=intent, curves=curves,
                    legacy_depth_prior=hasattr(self, 'search_value'),
                    predicted_statistics=self.ground_statistics(content),
                    grounded_statistics=ground.detach(), known_statistics=known.detach(),
                    pool=all_attention)

    def _writeback(self, state, base, contribution, records, amplitudes, before, after, valid):
        """Sparse evidence -> recurrent relation state, without detaching rounds.

        Sources and actually inspected targets receive messages. Rejected plans
        still supply inspected evidence, with an explicit zero accepted amount.
        Untouched stocks are bitwise unchanged. Nothing is cached across dates.
        """
        n, d = state.shape
        messages = torch.zeros_like(state)
        mass = state.new_zeros(n)
        for j, record in enumerate(records):
            idx = record['indices']
            source_idx = torch.where(record['anchor'] > 0)[0]
            joined = torch.cat([idx, source_idx])
            coefficients = torch.cat([record['probabilities'], record['anchor'][source_idx]])
            role = torch.cat([state.new_ones(idx.numel()), -state.new_ones(source_idx.numel())])
            counterpart = torch.cat([record['source_hidden'].expand(idx.numel(), d),
                                     record['value_pool'].expand(source_idx.numel(), d)])
            proposed = record['amount']*coefficients
            scalars = torch.stack([role*record['sign'], n*proposed,
                n*proposed*amplitudes[j],
                (record['predicted_gain']/(record['amount']*self.config.gain_scale).clamp_min(1e-8)).expand_as(role)], -1)
            evidence = self.feedback_message(torch.cat([state[joined], counterpart,
                record['diagnostic_content'].expand(joined.numel(), d), scalars], -1))
            messages = messages.index_add(0, joined, coefficients[:, None]*evidence)
            mass = mass.index_add(0, joined, coefficients)
        touched = (mass > 0) & valid
        if not records:
            return state, touched
        pooled = messages/mass[:, None].clamp_min(1e-8)
        scalars = torch.stack([(n*(after-before)).tanh(), torch.log1p(mass)], -1)
        inputs = torch.cat([state-base, contribution['representation']-state, pooled, scalars], -1)
        update = .25*torch.sigmoid(self.feedback_gate(inputs))*torch.tanh(self.feedback_update(inputs))
        updated = torch.where(touched[:, None], state+update, state)
        return updated, touched

    def _feedback_round(self, *, diagnosis, w, owner, anchors, amounts, retrieved,
                        available, allowance, remaining, round_id, conditioned,
                        base, store, batch, ids, statistics, observed, previous_weights):
        """Round-local prefix cache. A larger depth never recomputes old fine edges."""
        c, n, d = self.config, w.numel(), self.dimension
        holding = torch.stack([torch.log1p(n*w.clamp_min(0)), torch.log1p(n*(-w).clamp_min(0)),
                               (w > 0).to(w.dtype), (w < 0).to(w.dtype)], -1)
        cache = {}

        def make_plan(slot, depth):
            idx, coarse = retrieved[slot]
            anchor, amount = anchors[slot], amounts[slot]
            source_hidden = anchor @ conditioned
            old = cache.get(slot)
            start = 0 if old is None else old['depth']
            if depth <= start:
                raise ValueError('fine cache only permits a strict prefix extension')
            added = idx[start:depth]
            query = self.fine_query(torch.cat([diagnosis['content'][slot], source_hidden]))
            correction = conditioned[added]-base[added]
            key = store['fine_key'][batch, added]+self.fine_key(correction)+self.holding_key(holding[added])
            value = store['fine_value'][batch, added]+self.fine_value(correction)+self.holding_value(holding[added])
            local = torch.cat([diagnosis['content'][slot].expand(added.numel(), d),
                conditioned[added]-source_hidden, holding[added], statistics[added], observed[added].to(w.dtype)], -1)
            logits = (key @ query)/math.sqrt(c.key_dim)+coarse[start:depth]+self.marginal_score(local)[:, 0]
            if old is not None:
                logits, value = torch.cat([old['logits'], logits]), torch.cat([old['values'], value])
            cache[slot] = dict(depth=depth, logits=logits, values=value)
            probabilities = torch.softmax(logits, 0)
            indices = idx[:depth]
            target = torch.zeros_like(w).scatter(0, indices, probabilities)
            delta = diagnosis['signs'][slot]*amount*(target-anchor)
            value_pool = probabilities @ value
            features = torch.cat([diagnosis['content'][slot], value_pool, source_hidden])
            gain = amount*c.gain_scale*torch.tanh(self.plan_gain(features)[0])*torch.sigmoid(diagnosis['intent'][slot])
            return dict(slot=slot, depth=depth, indices=indices, asset_ids=ids[indices],
                anchor=anchor, amount=amount, sign=diagnosis['signs'][slot], logits=logits,
                coarse_logits=coarse[:depth], probabilities=probabilities, direction=delta,
                predicted_gain=gain, risk_factor=amount*self.plan_risk(torch.cat([value_pool, source_hidden])),
                diagnostic_content=diagnosis['content'][slot], source_hidden=source_hidden, value_pool=value_pool)

        return run_feedback_search(self, diagnosis, w, owner, anchors, amounts, available,
            allowance, remaining, round_id, make_plan, previous_weights)

    def _retrieve(self, query, eligible, store, batch, ids):
        c = self.config
        if store["mode"] == "exact":
            indices = torch.where(eligible)[0]
            group_reads = 0
        else:
            ranking = torch.argsort(store["centroids"][batch] @ query, descending=True, stable=True)
            pieces = []
            quota = max(1, c.group_scan_cap // c.group_probes)
            for group in ranking[:c.group_probes].tolist():
                posting = store["postings"][batch][group]
                if posting.numel() > quota:
                    pick = torch.linspace(0, posting.numel()-1, quota, device=ids.device).round().long()
                    posting = posting[pick]
                pieces.append(posting)
            raw = torch.cat(pieces)
            group_reads = raw.numel()
            indices = raw[eligible[raw]]
        values = store["key"][batch, indices] @ query
        # Stable identifier ties without constructing a length-N score vector.
        order = torch.argsort(ids[indices], stable=True)
        order = order[torch.argsort(values[order], descending=True, stable=True)]
        return indices[order[:c.retrieval_depths[-1]]], values[order[:c.retrieval_depths[-1]]], indices.numel(), group_reads

    def forward(self, hidden, valid_assets=None, *, asset_ids=None, statistics=None,
                statistics_observed=None, previous_weights=None, long_allowed=None,
                short_allowed=None, long_budget=None, return_aux=True, decision_keys=None, update_step=0):
        c = self.config
        if type(return_aux) is not bool:
            raise ValueError("return_aux must be boolean")
        if hidden.ndim != 3 or hidden.shape[-1] != self.dimension or min(hidden.shape[:2]) < 1:
            raise ValueError("hidden must be [batch, assets, dimension]")
        ref = next(self.parameters())
        if hidden.dtype != ref.dtype or hidden.device != ref.device or not torch.isfinite(hidden).all():
            raise ValueError("hidden must be finite and match model dtype/device")
        bsz, n, d = hidden.shape
        shape = (bsz, n)
        if type(update_step) is not int or update_step < 0:
            raise ValueError('update_step must be a nonnegative integer')
        keys_supplied = decision_keys is not None
        if decision_keys is None:
            decision_keys = [0]*bsz
        elif torch.is_tensor(decision_keys):
            if decision_keys.dtype != torch.int64 or decision_keys.shape != (bsz,):
                raise ValueError('decision_keys must be one int64 key per date')
            decision_keys = decision_keys.detach().cpu().tolist()
        if (not isinstance(decision_keys, (tuple, list)) or len(decision_keys) != bsz
                or any(type(key) is not int for key in decision_keys)):
            raise ValueError('decision_keys must be one integer key per date')

        def mask(value, name):
            if value is None:
                return torch.ones(shape, device=hidden.device, dtype=torch.bool)
            if value.shape != shape or value.dtype != torch.bool or value.device != hidden.device:
                raise ValueError(f"invalid {name}")
            return value

        valid = mask(valid_assets, "valid_assets")
        la, sa = mask(long_allowed, "long_allowed"), mask(short_allowed, "short_allowed")
        ids = torch.arange(n, device=hidden.device).expand(bsz, n) if asset_ids is None else asset_ids
        if ids.ndim == 1:
            ids = ids[None].expand(bsz, n)
        if (ids.shape != shape or ids.dtype != torch.int64 or ids.device != hidden.device
                or any(row.unique().numel() != n for row in ids)):
            raise ValueError("asset_ids must be unique int64 permanent identifiers per row")
        if statistics is None:
            if statistics_observed is not None:
                raise ValueError("statistics_observed supplied without statistics")
            statistics = hidden.new_zeros(bsz, n, c.statistic_dim)
            statistics_observed = torch.zeros_like(statistics, dtype=torch.bool)
        if (statistics.shape != (bsz, n, c.statistic_dim) or statistics.device != hidden.device
                or statistics.dtype != hidden.dtype or not torch.isfinite(statistics).all()):
            raise ValueError("invalid current-time statistics")
        if statistics_observed is None:
            statistics_observed = torch.ones_like(statistics, dtype=torch.bool)
        if (statistics_observed.shape != statistics.shape or statistics_observed.dtype != torch.bool
                or statistics_observed.device != hidden.device):
            raise ValueError("invalid statistics_observed")
        statistics = statistics.masked_fill(~statistics_observed, 0)
        if previous_weights is not None and (previous_weights.shape != shape
                or previous_weights.device != hidden.device or previous_weights.dtype != hidden.dtype
                or not torch.isfinite(previous_weights).all()):
            raise ValueError("invalid previous_weights")
        hidden = hidden * valid[..., None]
        store = self._store(hidden, valid, ids)
        h = store["relation"]
        scores = self.proposal(h)[..., 0].masked_fill(~valid, 0)
        initial, owners, budget = self._candidate(hidden, scores, valid, ids, la, sa, long_budget)
        if self.decision_mode == 'candidate_only':
            return self._candidate_result(initial, scores, h, budget, statistics_observed)
        result, relations, auxiliary, traces, stops, access_totals = [], [], [], [], [], []
        for b in range(bsz):
            w, remaining, row_aux, row_trace = initial[b], c.fine_edge_budget, [], []
            state, previous_delta = h[b], torch.zeros_like(w)
            stopper, previous_diagnosis = RefinementStopper(c), None
            side_allowed = torch.where(owners[b] > 0, la[b], sa[b])
            index = access_budget = None
            if c.access_policy == 'hierarchical_v1':
                index = SemanticIndex(state, store['key'][b], store['groups'][b], valid[b], ids[b],
                    groups_count=c.summaries, capacity=c.leaf_capacity,
                    stage_metadata=getattr(self, 'stage_access_metadata', False))
                access_budget = AccessBudget(c.coarse_budget, c.node_budget)
            for round_id in range(c.rounds):
                contribution = self._contributions(state, w, owners[b], valid[b], side_allowed,
                    scores[b], previous_delta, statistics[b], statistics_observed[b])
                contribution['asset_ids'] = ids[b]
                conditioned = contribution['representation']
                diagnosis = self._diagnostics(conditioned, self.diagnostic_key(conditioned), w, owners[b], valid[b],
                                              statistics[b], statistics_observed[b])
                queries = F.normalize(self.query_projection(diagnosis["content"]), dim=-1)
                retrieved, anchors, amounts, available = [], [], [], []
                eligibility = []
                scans = reads = checks = 0
                for slot in range(c.diagnostic_slots):
                    sign = diagnosis["signs"][slot]
                    source = diagnosis["source"][slot]
                    source_idx = _ordered(torch.where(source > 0)[0], source, ids[b])[:c.anchors_per_query]
                    anchor = torch.zeros_like(w)
                    if source_idx.numel():
                        anchor = anchor.scatter(0, source_idx, source[source_idx] / source[source_idx].sum())
                        amount = c.repair_fraction * (w[source_idx].abs() / anchor[source_idx]).min()
                    else:
                        amount = w.sum()*0
                    allowed = la[b] if sign > 0 else sa[b]
                    eligible = valid[b] & (owners[b] == sign) & allowed & (anchor == 0) & (amount > 0)
                    if index is None:
                        idx, coarse, count, read = self._retrieve(queries[slot], eligible, store, b, ids[b])
                        retrieved.append((idx, coarse))
                        available.append(idx.numel())
                        scans += count
                        reads += read
                    else:
                        eligibility.append(eligible)
                        available.append(int(eligible.sum()))
                    anchors.append(anchor)
                    amounts.append(amount)
                    checks += n
                # Divide only the remaining budget; unused capacity carries to
                # the next round. No reset to a full per-round budget.
                budget_before = remaining
                allowance = remaining // (c.rounds-round_id)
                minimum_increment = min((min(c.retrieval_depths[1], count)
                                         for count in available if count), default=0)
                if index is not None:
                    minimum_increment = int(any(available))  # global exploration can buy one edge
                if c.stopping_mode == 'adaptive' and remaining >= minimum_increment:
                    # Borrow at most one actionable quantum from later rounds.
                    # A too-small equal slice must not manufacture stagnation.
                    allowance = max(allowance, minimum_increment)
                diagnosis["search_values"] = diagnosis["curves"] * torch.stack(amounts)[:, None]
                search = None
                if index is not None:
                    before_checks = access_budget.eligibility_checks
                    search = AccessRound(self, index=index, budget=access_budget, diagnosis=diagnosis,
                        weights=w, owners=owners[b], anchors=anchors, amounts=amounts, eligible=eligibility,
                        allowance=allowance, remaining=remaining, round_id=round_id, conditioned=conditioned,
                        base=h[b], store=store, batch=b, ids=ids[b], statistics=statistics[b],
                        observed=statistics_observed[b],
                        previous_weights=None if previous_weights is None else previous_weights[b],
                        decision_key=decision_keys[b], update_step=update_step,
                        collect_aux=return_aux).run()
                    depths, spent = search['depths'], search['spent']
                    scans = reads = search['trace']['coarse_rows']
                    checks += access_budget.eligibility_checks-before_checks
                elif c.search_policy == 'solver_feedback_v1':
                    search = self._feedback_round(diagnosis=diagnosis, w=w, owner=owners[b],
                        anchors=anchors, amounts=amounts, retrieved=retrieved, available=available,
                        allowance=allowance, remaining=remaining, round_id=round_id, conditioned=conditioned,
                        base=h[b], store=store, batch=b, ids=ids[b], statistics=statistics[b],
                        observed=statistics_observed[b],
                        previous_weights=None if previous_weights is None else previous_weights[b])
                    depths, spent = search['depths'], search['spent']
                else:
                    depths, spent = allocate_depths(diagnosis["search_values"], available, c, allowance)
                remaining -= spent
                holding = None if search is not None else torch.stack([
                    torch.log1p(n*w.clamp_min(0)), torch.log1p(n*(-w).clamp_min(0)),
                    (w > 0).to(w.dtype), (w < 0).to(w.dtype)], -1)
                deltas, gains, risk, fine_records = [], [], [], []
                for slot, depth in enumerate(depths if search is None else []):
                    if depth == 0:
                        continue
                    idx, coarse = retrieved[slot]
                    idx, coarse = idx[:depth], coarse[:depth]
                    anchor, amount = anchors[slot], amounts[slot]
                    source_hidden = anchor @ conditioned
                    query = self.fine_query(torch.cat([diagnosis["content"][slot], source_hidden]))
                    # Base index/projections are reused; only inspected edges
                    # add the current conditional/recurrent-state correction.
                    correction = conditioned[idx]-h[b, idx]
                    key = store["fine_key"][b, idx] + self.fine_key(correction) + self.holding_key(holding[idx])
                    value = store["fine_value"][b, idx] + self.fine_value(correction) + self.holding_value(holding[idx])
                    expanded = diagnosis["content"][slot].expand(depth, d)
                    local = torch.cat([expanded, conditioned[idx]-source_hidden, holding[idx],
                                       statistics[b, idx], statistics_observed[b, idx].to(w.dtype)], -1)
                    logits = (key @ query)/math.sqrt(c.key_dim) + coarse + self.marginal_score(local)[:, 0]
                    probabilities = torch.softmax(logits, 0)
                    target = torch.zeros_like(w).scatter(0, idx, probabilities)
                    delta = diagnosis["signs"][slot] * amount * (target-anchor)
                    value_pool = probabilities @ value
                    plan_features = torch.cat([diagnosis["content"][slot], value_pool, source_hidden])
                    predicted_gain = amount*c.gain_scale*torch.tanh(self.plan_gain(plan_features)[0])
                    predicted_gain = predicted_gain*torch.sigmoid(diagnosis["intent"][slot])
                    factor = amount*self.plan_risk(torch.cat([value_pool, source_hidden]))
                    deltas.append(delta)
                    gains.append(predicted_gain)
                    risk.append(factor)
                    fine_records.append(dict(slot=slot, indices=idx, asset_ids=ids[b, idx],
                        anchor=anchor, amount=amount, sign=diagnosis["signs"][slot],
                        logits=logits, coarse_logits=coarse, probabilities=probabilities,
                        direction=delta, predicted_gain=predicted_gain,
                        diagnostic_content=diagnosis['content'][slot], source_hidden=source_hidden,
                        value_pool=value_pool))
                if search is not None:
                    fine_records = search['records']
                    v, q, g = (search['problem'][name] for name in ('directions', 'curvature', 'gains'))
                elif deltas:
                    v = torch.stack(deltas, 1)
                    factors = torch.stack(risk)
                    q = factors @ factors.T + torch.diag(c.qp_ridge*v.square().sum(0) + 1e-6)
                    g = torch.stack(gains)
                else:
                    v, q, g = w.new_zeros(n, 0), w.new_zeros(0, 0), w.new_zeros(0)
                solution = search['solution'] if search is not None else solve_repairs(w, v, owners[b], g, q,
                    previous_weights=None if previous_weights is None else previous_weights[b],
                    fee=c.cost_rate, iterations=c.qp_iterations, rho=c.qp_rho, tolerance=c.qp_tolerance)
                updated, touched = self._writeback(state, h[b], contribution, fine_records,
                    solution.amplitudes, w, solution.weights, valid[b])
                if index is not None:
                    index.refresh(updated, touched, self.retrieval_key)
                metrics = stopping_metrics(before_weights=w, after_weights=solution.weights,
                    before_state=state, after_state=updated, touched=touched,
                    diagnosis=diagnosis['content'], previous_diagnosis=previous_diagnosis,
                    search_values=diagnosis['search_values'], available=available, gains=g,
                    solver=solution.diagnostics,
                    previous_weights=None if previous_weights is None else previous_weights[b], config=c,
                    search_gain_override=None if search is None else search['opportunity'])
                halting = stopper.observe(metrics, completed_rounds=round_id+1,
                    remaining=remaining, available=available, minimum_increment=minimum_increment)
                if (index is not None and c.stopping_mode == 'adaptive' and remaining > 0
                        and access_budget.coarse_remaining == 0):
                    # Fine/coarse caches are round-local. Future rounds cannot
                    # buy a fresh stock row; exhaustion is not an optimality claim.
                    halting.update(stopped=True, stop_reason='coarse_budget_exhausted')
                trace = dict(round=round_id+1, fine_edges=spent, cumulative_fine_edges=c.fine_edge_budget-remaining,
                    fine_edge_allowance=allowance, depths=depths, plans=len(fine_records),
                    fine_budget_before=budget_before, fine_budget_remaining=remaining,
                    minimum_fine_increment=minimum_increment, stopping=halting,
                    coarse_similarity_evaluations=scans, posting_items_read=reads,
                    eligibility_checks=checks, diagnostic_pool_pairs=c.diagnostic_slots*n,
                    query_centroid_comparisons=c.diagnostic_slots*c.summaries if store["mode"] == "grouped" else 0,
                    repair_matrix_elements=int(v.numel()), solver=solution.diagnostics,
                    conditioned_stock_rows=int(valid[b].sum()),
                    contribution_counterfactuals_available=int(contribution['available'].sum()),
                    diagnostic_key_projection_rows=int(valid[b].sum()),
                    fine_state_correction_rows=spent,
                    feedback_message_rows=sum(x['indices'].numel()+int((x['anchor'] > 0).sum()) for x in fine_records),
                    state_writeback_stocks=int(touched.sum()),
                    state_update_rms=float((updated-state).detach().square().mean().sqrt()))
                if search is not None:
                    trace['search'] = search['trace']
                row_trace.append(trace)
                if return_aux:
                    row_aux.append(dict(diagnosis=diagnosis, contribution=contribution, fine=fine_records,
                        weights_before=w, weights_after=solution.weights, amplitudes=solution.amplitudes,
                        relation_before=state, relation_after=updated, evidence_touched=touched))
                    if search is not None:
                        row_aux[-1]['access_search' if index is not None else 'conditional_search'] = search['auxiliary']
                previous_delta = solution.weights-w
                previous_diagnosis = diagnosis['content'].detach()
                state = updated
                w = solution.weights
                if halting['stopped']:
                    break
            result.append(w)
            relations.append(state)
            auxiliary.append(row_aux)
            traces.append(row_trace)
            if index is not None:
                access_totals.append(dict(coarse_rows=access_budget.coarse_used,
                    node_value_evaluations=access_budget.nodes_used,
                    stock_value_evaluations=sum(t['search']['stock_value_evaluations'] for t in row_trace),
                    duplicate_coarse_reads_avoided=access_budget.duplicate_reads_avoided,
                    index_build_rows=index.build_rows, index_build_memberships=index.build_memberships,
                    index_nodes=len(index.nodes), index_refresh_rows=index.refresh_rows,
                    index_refresh_nodes=index.refresh_nodes, index_versions=index.version,
                    topology_fixed_within_date=True))
            stops.append(dict(completed_rounds=len(row_trace), max_rounds=c.rounds,
                stopped_early=len(row_trace) < c.rounds, stop_reason=halting['stop_reason'],
                fine_edges_used=c.fine_edge_budget-remaining, fine_edges_remaining=remaining,
                consecutive_stable_rounds=stopper.stable_rounds))
        weights = torch.stack(result)
        admission_diagnostics = []
        for b in range(bsz):
            long_used = weights[b].clamp_min(0).sum()
            short_used = (-weights[b]).clamp_min(0).sum()
            has_long = bool((initial[b] > 0).any())
            has_short = bool((initial[b] < 0).any())
            reasons = []
            if not has_long: reasons.append('no_admitted_long_cash')
            if not has_short and bool((1-budget[b]) > 0):
                reasons.append('no_admitted_short_cash')
            admission_diagnostics.append(dict(fallback_used=bool(reasons), reasons=reasons,
                planned_long_budget=float(budget[b].detach()),
                planned_short_budget=float((1-budget[b]).detach()),
                policy_reserved_cash=0.,
                deployed_long_budget=float(long_used.detach()), deployed_short_budget=float(short_used.detach()),
                unused_risk_budget_cash=float((1-long_used-short_used).clamp_min(0).detach()),
                admission_rules_relaxed=False, opposite_side_budget_increased=False))
        return DQResult(weights, initial, scores, torch.stack(relations), budget, auxiliary, dict(
            schema='dq_pcrr.v7',
            search_policy=c.search_policy, access_policy=c.access_policy, access_accounting=access_totals,
            admission_fallback_policy=c.admission_spec(), admission_fallback=admission_diagnostics,
            coarse_budget_per_date=c.coarse_budget, node_budget_per_date=c.node_budget,
            exploration_keys_supplied=keys_supplied, exploration_update_step=update_step,
            exploration_is_portfolio_randomization=False,
            batches=bsz, assets=n, summaries=c.summaries,
            diagnostic_slots=c.diagnostic_slots, retrieval_mode=store["mode"],
            fine_edge_budget_per_date=c.fine_edge_budget, retrieval_depths=c.retrieval_depths, rounds=traces,
            stopping_policy=c.stopping_spec(), decision_stopping=stops,
            stopping_uses_future_outcomes=False, stopping_is_optimality_certificate=False,
            summary_assignment_pairs_per_date=n*c.summaries,
            summary_read_pairs_per_date=n*c.summaries,
            shared_stock_projection_passes=1, cross_day_state_reused=False,
            shared_stock_projection_scope="base_index_and_base_fine_only",
            stock_contributions_conditioned_before_diagnosis=True,
            recurrent_relation_state=True, base_index_rebuilt_between_rounds=False,
            contribution_supervision="training_only_local_counterfactual_log_wealth_not_exact_sharpe",
            full_stock_pair_attention=False, grouped_retrieval_is_approximate=store["mode"] != "exact",
            end_to_end_sublinear_claim=False,
            gross_target=None,
            gross_limit=1., target_gross_by_batch=weights.detach().abs().sum(-1).cpu().tolist(),
            final_support_may_expand=True, fixed_side_ownership=True,
            canonical_weights="weights", score_semantics="initial_proposal_only",
            date_causality_checked_by_tensor_model=False,
            statistic_coverage=float(statistics_observed.detach().to(hidden.dtype).mean())), h)

    def _candidate_result(self, weights, scores, relation, budget, statistics_observed):
        """Candidate is the executable output; no diagnosis/index/solver/aux labels."""
        c = self.config
        admission, stops = [], []
        for b in range(len(weights)):
            long_used = weights[b].clamp_min(0).sum()
            short_used = (-weights[b]).clamp_min(0).sum()
            has_long, has_short = bool(long_used > 0), bool(short_used > 0)
            reasons = []
            if not has_long: reasons.append('no_admitted_long_cash')
            if not has_short and bool((1-budget[b]) > 0):
                reasons.append('no_admitted_short_cash')
            admission.append(dict(fallback_used=bool(reasons), reasons=reasons,
                planned_long_budget=float(budget[b].detach()), planned_short_budget=float((1-budget[b]).detach()),
                policy_reserved_cash=0.,
                deployed_long_budget=float(long_used.detach()), deployed_short_budget=float(short_used.detach()),
                unused_risk_budget_cash=float((1-long_used-short_used).clamp_min(0).detach()),
                admission_rules_relaxed=False, opposite_side_budget_increased=False))
            stops.append(dict(completed_rounds=0, max_rounds=0, stopped_early=False,
                stop_reason='candidate_only_no_refinement', fine_edges_used=0,
                fine_edges_remaining=0, consecutive_stable_rounds=0))
        return DQResult(weights, weights, scores, relation, budget, [[] for _ in weights], dict(
            schema='dq_candidate_only.v1', decision_mode='candidate_only',
            admission_fallback_policy=c.admission_spec(), admission_fallback=admission,
            decision_stopping=stops, rounds=[[] for _ in weights], access_accounting=[],
            fine_edge_budget_per_date=0, coarse_budget_per_date=0, node_budget_per_date=0,
            summary_assignment_pairs_per_date=weights.shape[1]*c.summaries,
            summary_read_pairs_per_date=weights.shape[1]*c.summaries,
            diagnostic_slots=0, refinement_parameters_instantiated=False,
            auxiliary_supervision_used=False, final_support_may_expand=False,
            stopping_uses_future_outcomes=False, gross_limit=1.,
            target_gross_by_batch=weights.detach().abs().sum(-1).cpu().tolist(),
            canonical_weights='weights', score_semantics='initial_proposal_and_final_candidate',
            statistic_coverage=float(statistics_observed.detach().float().mean())), relation)
