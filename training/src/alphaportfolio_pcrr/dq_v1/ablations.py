"""Explicit train-time ablations, not a mutation of a trained checkpoint.

The no-context variant removes EXPLICIT book inputs from diagnosis. Relation
states can still contain earlier within-decision evidence. Holdings remain
available to source selection, fine repair evaluation and the constrained
solver. It is deliberately not called a completely book-blind policy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import DiagnosticQueryPCRR, _masked_softmax
from .hierarchy import ACCESS_SCALARS
from .search import FEEDBACK_FIELDS

NO_DIAGNOSIS = 'no_explicit_diagnostic_context'
NO_WRITEBACK = 'no_relation_writeback'
NO_SOLVER_FEEDBACK = 'no_solver_feedback_search'
NO_MONTHLY = 'no_monthly'
NO_SUMMARIES = 'no_shared_summary'
VARIANTS = (NO_DIAGNOSIS, NO_WRITEBACK, NO_SOLVER_FEEDBACK, NO_MONTHLY, NO_SUMMARIES)


def validate_ablation(value):
    if value not in VARIANTS:
        raise ValueError('unknown train-time DQ ablation')
    return value


def stock_only_columns(d, s):
    # Original stock_context inputs: five d-vectors, 12 book scalars, stats,
    # observed. Keep state, state-universe, standardized stock score and stats.
    return [*range(d), *range(3*d, 4*d), 5*d+6,
            *range(5*d+12, 5*d+12+2*s)]


def solver_blind_columns(d, config):
    """Keep pre-solve query/resource inputs, not the 12 solver fields or factor."""
    prefix = 2*d+config.key_dim
    end = prefix+len(FEEDBACK_FIELDS)+config.qp_rank
    return [*range(prefix+4), *range(end, end+len(ACCESS_SCALARS))]


class SolverBlindAccessValue(nn.Module):
    """One structural projection for node, stock AND deferred audit paths.

    Solver-generated supervision is retained, but live solver outcomes cannot
    condition search priority or the search-opportunity estimate. Final repair
    solving, incumbent protection and inter-round configuration feedback stay.
    """
    def __init__(self, original, dimension, config):
        super().__init__()
        self.trunk, self.heads = original.trunk, original.heads
        self.gain_scale = original.gain_scale
        columns = solver_blind_columns(dimension, config)
        self.register_buffer('input_columns', torch.tensor(columns), persistent=False)
        old = self.trunk[0]
        with torch.random.fork_rng(devices=[]):
            smaller = nn.Linear(len(columns), old.out_features, bias=True)
        with torch.no_grad():
            smaller.weight.copy_(old.weight[:, columns])
            smaller.bias.copy_(old.bias)
        self.trunk[0] = smaller

    def forward(self, features):
        h = self.trunk(features.index_select(-1, self.input_columns))
        return self.gain_scale*torch.tanh(torch.cat([head(h) for head in self.heads], -1))


class AblationPCRR(DiagnosticQueryPCRR):
    def __init__(self, *args, ablation, **kwargs):
        self.ablation = validate_ablation(ablation)
        super().__init__(*args, **kwargs)
        if self.decision_mode != 'full':
            raise ValueError('candidate-only is a different experiment')
        if self.ablation == NO_DIAGNOSIS:
            old = self.stock_context[0]
            cols = stock_only_columns(self.dimension, self.config.statistic_dim)
            # No extra RNG consumption; common tensors retain their original
            # initialization and the reduced input layer copies exact columns.
            with torch.random.fork_rng(devices=[]):
                smaller = nn.Linear(len(cols), old.out_features, bias=True)
            with torch.no_grad():
                smaller.weight.copy_(old.weight[:, cols])
                smaller.bias.copy_(old.bias)
            self.stock_context[0] = smaller
            del self.book_condition, self.contribution_head, self.contribution_read
        elif self.ablation == NO_WRITEBACK:
            del self.feedback_message, self.feedback_update, self.feedback_gate
        elif self.ablation == NO_SOLVER_FEEDBACK:
            if self.config.access_policy != 'hierarchical_v1':
                raise ValueError('solver-feedback ablation requires hierarchical access')
            self.access_value = SolverBlindAccessValue(self.access_value, self.dimension, self.config)
        elif self.ablation == NO_SUMMARIES:
            del self.summary_tokens, self.summary_key, self.summary_value, self.summary_read, self.summary_output

    def _store(self, hidden, valid, ids):
        if self.ablation != NO_SUMMARIES:
            return super()._store(hidden, valid, ids)
        # A single root delegates all finer grouping to the index's median splits.
        relation = hidden * valid[..., None]
        key = F.normalize(self.retrieval_key(relation), dim=-1)
        centroids = key.new_zeros(key.shape[0], self.config.summaries, key.shape[-1])
        centroids[:, 0] = F.normalize((key * valid[..., None]).sum(1), dim=-1)
        return dict(relation=relation, key=key, centroids=centroids, mode='hierarchical',
                    postings=[], groups=torch.zeros_like(ids), fine_key=self.fine_key(relation),
                    fine_value=self.fine_value(relation))

    def _contributions(self, state, w, owner, valid, allowed, scores, previous_delta,
                       statistics, observed):
        if self.ablation != NO_DIAGNOSIS:
            return super()._contributions(state, w, owner, valid, allowed, scores,
                                          previous_delta, statistics, observed)
        universe = (state*valid[:, None]).sum(0)/valid.sum().clamp_min(1)
        centered = scores-(scores*valid).sum()/valid.sum().clamp_min(1)
        scale = ((centered.square()*valid).sum()/valid.sum().clamp_min(1)).clamp_min(1e-12).sqrt()
        features = torch.cat([state, state-universe, (centered/scale).tanh()[:, None],
                              statistics, observed.to(state.dtype)], -1)
        context = (state+self.stock_context(features))*valid[:, None]
        # Metadata below is for feasible operations, never a diagnostic input.
        budget = torch.where(owner > 0, w.clamp_min(0).sum(), (-w).clamp_min(0).sum())
        reference_mass = budget-w.abs()
        held = w.abs() > 1e-10
        available = valid & allowed & (reference_mass > 1e-8)
        amount = self.config.repair_fraction*torch.where(held, w.abs(), budget)*available
        return dict(representation=context, predicted_per_unit=torch.zeros_like(w),
            amount=amount, available=available, held=held, owner=owner, weights=w,
            reference_mass=reference_mass, contribution_supervision_disabled=True,
            semantics='stock_only_context_no_configured_contribution_prediction')

    def _diagnostics(self, h, key, w, owner, valid, statistics, observed):
        if self.ablation != NO_DIAGNOSIS:
            return super()._diagnostics(h, key, w, owner, valid, statistics, observed)
        c = self.config
        tokens = self.diagnostic_tokens
        logits = tokens @ key.T / math.sqrt(self.dimension)
        pooling = _masked_softmax(logits, valid[None].expand_as(logits),
                                  allow_empty=True)
        signs = torch.ones(c.diagnostic_slots, device=w.device, dtype=w.dtype)
        signs[c.diagnostic_slots//2:] = -1
        # The sources are still actual held stocks of the appropriate side.
        # Neither source attention nor its statistics feed the diagnostic MLP.
        held = (owner[None] == signs[:, None]) & (w.abs()[None] > 1e-10) & valid[None]
        source = _masked_softmax(logits+w.abs().clamp_min(1e-12).log()[None], held,
                                allow_empty=True)
        known = pooling @ observed.to(w.dtype)
        ground = (pooling @ (statistics*observed))/known.clamp_min(1e-8)
        pooled = pooling @ h
        content = self.diagnose(torch.cat([tokens, pooled, pooled, ground, known], -1))
        intent = self.intent(content)[:, 0]
        if hasattr(self, 'search_value'):
            parameters = self.search_value(content)
            depths = w.new_tensor(c.retrieval_depths)[None]
            saturation = 8+F.softplus(parameters[:, 1:2])
            curves = (c.gain_scale*torch.tanh(parameters[:, :1])*torch.sigmoid(intent[:, None])
                      *(1-torch.exp(-depths/saturation)))
        else:
            curves = w.new_zeros(c.diagnostic_slots, len(c.retrieval_depths))
        return dict(content=content, source=source, signs=signs, intent=intent, curves=curves,
            legacy_depth_prior=hasattr(self, 'search_value'),
            predicted_statistics=self.ground_statistics(content),
            grounded_statistics=ground.detach(), known_statistics=known.detach(), pool=pooling)

    def _writeback(self, state, base, contribution, records, amplitudes, before, after, valid):
        if self.ablation == NO_WRITEBACK:
            return state, torch.zeros_like(valid)
        return super()._writeback(state, base, contribution, records, amplitudes, before, after, valid)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.diagnostics.update(train_time_ablation=self.ablation,
            stock_contributions_conditioned_before_diagnosis=self.ablation != NO_DIAGNOSIS,
            relation_state_writeback_enabled=self.ablation != NO_WRITEBACK,
            completely_book_blind_policy=False, common_stopping_rule_unchanged=True)
        if self.ablation == NO_SOLVER_FEEDBACK:
            out.diagnostics.update(search_solver_feedback_enabled=False,
                final_joint_repair_solver_retained=True,
                solver_based_access_supervision_retained=True,
                inter_round_configuration_feedback_retained=True)
            for rounds in out.diagnostics['rounds']:
                for row in rounds:
                    row['search'].update(solver_feedback_reopens_unvisited_groups=False,
                        solver_feedback_fields_logged_but_not_access_inputs=True)
        return out
