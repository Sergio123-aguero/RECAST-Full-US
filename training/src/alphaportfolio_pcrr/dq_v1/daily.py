"""Dual-frequency encoders and allocation-guided portfolio refinement."""
import copy
from dataclasses import asdict, replace
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from alphaportfolio_original.config import AlphaPortfolioConfig, ModelConfig
from alphaportfolio_original.model import AlphaPortfolioOutput, SequenceRepresentationExtractionModule
from alphaportfolio_original.portfolio import PortfolioSelection

from .config import DQConfig
from .model import DiagnosticQueryPCRR
from .feature_contract import validate_feature_contract, contract_sha256


class DQDailyPortfolio(nn.Module):
    checkpoint_schema = "dq_daily_portfolio.v4"

    def __init__(self, config: AlphaPortfolioConfig, *, daily_config=None, dq_config=None,
                 chunk_size=64, checkpoint_encoders=True, budget_hidden_dim=32, feature_contract=None,
                 decision_mode='full', ablation=None):
        super().__init__()
        config.validate()
        daily = daily_config or replace(config.model, input_dim=29, lookback_months=252)
        daily.validate()
        if daily.d_model != config.model.d_model:
            raise ValueError("daily/monthly representation dimensions must match")
        if type(chunk_size) is not int or chunk_size < 1 or type(checkpoint_encoders) is not bool:
            raise ValueError("invalid encoder chunk/checkpoint settings")
        self.config, self.daily_config = config, daily
        self.dq_config = (dq_config or DQConfig()).validate()
        if decision_mode not in {'full', 'candidate_only'}:
            raise ValueError('unknown DQ decision mode')
        self.decision_mode = decision_mode
        self.feature_contract = (None if feature_contract is None else validate_feature_contract(
            feature_contract, daily_config=daily, monthly_config=config.model,
            statistic_dim=self.dq_config.statistic_dim))
        self.checkpoint_schema = 'dq_daily_portfolio.v8'
        if self.decision_mode == 'candidate_only':
            self.checkpoint_schema = 'dq_daily_candidate.v1'
        self.chunk_size, self.checkpoint_encoders = chunk_size, checkpoint_encoders
        self.budget_hidden_dim = budget_hidden_dim
        self.monthly_block_length = config.model.lookback_months
        self.monthly_lookback = 2*self.monthly_block_length
        self.daily_lookback, self.daily_feature_count = daily.lookback_months, daily.input_dim
        self.initialization_binding = {"mode": "fresh", "performance_preservation_promised": False}
        self.register_buffer('_completed_updates', torch.zeros((), dtype=torch.int64))
        d = config.model.d_model
        self.daily_srem = SequenceRepresentationExtractionModule(daily)
        self.monthly_srem = SequenceRepresentationExtractionModule(config.model)
        self.missing_projection = nn.Linear(daily.input_dim, d, bias=False)
        self.monthly_projection = nn.Linear(d, d, bias=False)
        self.background_gate = nn.Linear(2*d+2, d)
        self.ablation = ablation
        core_class, core_options = DiagnosticQueryPCRR, {}
        if ablation is not None:
            from .ablations import AblationPCRR, validate_ablation
            validate_ablation(ablation)
            if decision_mode != 'full':
                raise ValueError('Each module ablation starts from the full RECAST architecture.')
            core_class, core_options = AblationPCRR, dict(ablation=ablation)
            self.checkpoint_schema = 'dq_daily_ablation.v1'
        self.core = core_class(d, self.dq_config,
            long_fraction=config.portfolio.long_fraction, short_fraction=config.portfolio.short_fraction,
            budget_hidden_dim=budget_hidden_dim, decision_mode=decision_mode, **core_options)
        nn.init.zeros_(self.missing_projection.weight)
        nn.init.zeros_(self.monthly_projection.weight)
        nn.init.zeros_(self.background_gate.weight)
        nn.init.constant_(self.background_gate.bias, -2.)

    def _encode(self, module, states):
        b, n, t, f = states.shape
        values = []
        def encode(x):
            return module(x, return_attention=False)[0]
        direct = (getattr(self, 'direct_encoder_autograd', False) or
                  (module is self.monthly_srem and getattr(self, 'direct_monthly_autograd', False)))
        for block in states.reshape(b*n, 1, t, f).split(self.chunk_size):
            value = checkpoint(encode, block, use_reentrant=False, preserve_rng_state=True) if (
                self.checkpoint_encoders and torch.is_grad_enabled()
                and not direct) else encode(block)
            values.append(value.reshape(-1, self.config.model.d_model))
        return torch.cat(values).reshape(b, n, self.config.model.d_model)

    def encode_features(self, daily_states, monthly_states, *, daily_observed=None, monthly_observed=None):
        ref = next(self.parameters())
        clean, observed = [], []
        specs = ((daily_states, daily_observed, self.daily_lookback, self.daily_feature_count, "daily"),
                 (monthly_states, monthly_observed, self.monthly_lookback, self.config.model.input_dim, "monthly"))
        for states, mask, length, features, name in specs:
            if (states.ndim != 4 or min(states.shape[:2]) < 1 or states.shape[-2:] != (length, features)
                    or states.device != ref.device or states.dtype != ref.dtype or not torch.isfinite(states).all()):
                raise ValueError(f"invalid {name} feature tensor")
            if mask is None:
                mask = torch.ones_like(states, dtype=torch.bool)
            if mask.shape != states.shape or mask.dtype != torch.bool or mask.device != states.device:
                raise ValueError(f"invalid {name} observed mask")
            clean.append(states.masked_fill(~mask, 0))
            observed.append(mask)
        daily, monthly = clean
        do, mo = observed
        if daily.shape[:2] != monthly.shape[:2]:
            raise ValueError("daily/monthly asset axes differ")
        hd = self._encode(self.daily_srem, daily) + self.missing_projection((~do).to(daily.dtype).mean(-2))
        if self.ablation == 'no_monthly':
            return hd, dict(monthly_coverage_mean=0., monthly_correction_rms=0.)
        length = self.monthly_block_length
        coverage = torch.stack([mo[:, :, i:i+length].to(monthly.dtype).mean((-1, -2))
                                for i in (0, length)], -1)
        def monthly_blocks(states, mask):
            return tuple(self._encode(self.monthly_srem,states[:,:,i:i+length]) for i in (0,length))
        cache = getattr(self, '_evaluation_monthly_cache', None)
        blocks = torch.stack(monthly_blocks(monthly,mo) if cache is None else
                             cache.resolve(self,monthly,mo,compute_rows=monthly_blocks),-2)
        hm = (blocks*coverage[..., None]).sum(-2) / coverage.sum(-1, keepdim=True).clamp_min(1e-8)
        gate = torch.sigmoid(self.background_gate(torch.cat([hd, hm, coverage], -1)))
        delta = coverage.mean(-1, keepdim=True)*gate*self.monthly_projection(hm)
        return hd+delta, dict(monthly_coverage_mean=float(coverage.detach().mean()),
                             monthly_correction_rms=float(delta.detach().square().mean().sqrt()))

    def forward(self, daily_states, monthly_states, valid_assets=None, *,
                daily_observed=None, monthly_observed=None, **decision_inputs):
        hidden, fusion = self.encode_features(daily_states, monthly_states,
            daily_observed=daily_observed, monthly_observed=monthly_observed)
        if self.dq_config.access_policy == 'hierarchical_v1':
            decision_inputs.setdefault('update_step', self.completed_updates)
        result = self.core(hidden, valid_assets, **decision_inputs)

        def portfolio(weights):
            long, short = weights > 1e-10, weights < -1e-10
            return PortfolioSelection(2*weights, long, short, long.sum(-1), short.sum(-1))

        out = AlphaPortfolioOutput(result.proposal_scores, hidden, result.relation, None, None,
                                   portfolio(result.weights))
        out.normalized_weights = result.weights
        out.candidate_scores = result.proposal_scores
        out.candidate_portfolio = portfolio(result.initial_weights)
        out.long_budget = result.long_budget
        out.deployed_long_budget = result.weights.clamp_min(0).sum(-1)
        out.deployed_short_budget = (-result.weights).clamp_min(0).sum(-1)
        out.unused_risk_budget_cash = (1-result.weights.abs().sum(-1)).clamp_min(0)
        out.dq_result = result
        out.initial_relation = result.initial_relation
        out.dq_diagnostics = {**result.diagnostics, **fusion,
            "legacy_output_weights_scale": 2., "normalized_weights_scale": 1.,
            "budget_applied_before_refinement": True, "fully_trainable": all(p.requires_grad for p in self.parameters())}
        if self.feature_contract is not None:
            out.dq_diagnostics['feature_contract_sha256'] = contract_sha256(self.feature_contract)
            out.dq_diagnostics['grounding_feature_count'] = self.dq_config.statistic_dim
        return out

    def parameter_groups(self, *, daily_lr=1e-4, monthly_lr=1e-6,
                         fusion_lr=1e-4, decision_lr=1e-4):
        rates = (daily_lr, monthly_lr, fusion_lr, decision_lr)
        if any(isinstance(r, bool) or not math.isfinite(r) or r <= 0 for r in rates):
            raise ValueError("learning rates must be finite and positive")
        groups = [dict(name="daily_encoder", params=list(self.daily_srem.parameters()), lr=daily_lr),
                  dict(name="monthly_encoder", params=list(self.monthly_srem.parameters()), lr=monthly_lr),
                  dict(name="dual_frequency_fusion", params=[*self.missing_projection.parameters(),
                       *self.monthly_projection.parameters(), *self.background_gate.parameters()], lr=fusion_lr),
                  dict(name="diagnostic_query_repairs", params=list(self.core.parameters()), lr=decision_lr)]
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
            raise RuntimeError("optimizer must cover every parameter exactly once")
        if not all(p.requires_grad for p in self.parameters()):
            raise RuntimeError("v1 full training expects all parameters unfrozen")
        return groups

    @property
    def completed_updates(self):
        return int(self._completed_updates)

    def mark_completed_update(self):
        self._completed_updates.add_(1)

    def constructor_spec(self):
        spec = dict(config=self.config.to_dict(), daily_config=asdict(self.daily_config),
                    dq_config=self.dq_config.to_dict(), chunk_size=self.chunk_size,
                    checkpoint_encoders=self.checkpoint_encoders, budget_hidden_dim=self.budget_hidden_dim)
        if self.feature_contract is not None:
            spec['feature_contract'] = copy.deepcopy(self.feature_contract)
        if self.ablation is not None:
            spec['ablation'] = self.ablation
        if self.decision_mode != 'full':
            spec['decision_mode'] = self.decision_mode
        return spec

    def checkpoint_payload(self):
        """Tensor-only portable payload; does not write or overwrite any file."""
        return dict(schema=self.checkpoint_schema, constructor=self.constructor_spec(),
                    initialization_binding=copy.deepcopy(self.initialization_binding),
                    state_dict={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_checkpoint_payload(cls, payload):
        if payload.get("schema") not in {'dq_daily_portfolio.v8', 'dq_daily_candidate.v1', 'dq_daily_ablation.v1'}:
            raise ValueError("Unsupported Full US checkpoint schema.")
        spec = dict(payload["constructor"])
        spec["config"] = AlphaPortfolioConfig.from_dict(spec["config"])
        spec["daily_config"] = ModelConfig(**spec["daily_config"])
        spec["dq_config"] = DQConfig.from_dict(spec["dq_config"])
        model = cls(**spec)
        if model.checkpoint_schema != payload['schema']:
            raise ValueError('checkpoint schema and search architecture disagree')
        floating = next(v for v in payload["state_dict"].values() if torch.is_floating_point(v))
        model.to(dtype=floating.dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.initialization_binding = copy.deepcopy(payload.get("initialization_binding", {}))
        return model
