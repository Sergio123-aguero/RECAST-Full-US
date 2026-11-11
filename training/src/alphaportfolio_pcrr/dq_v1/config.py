"""Versioned engineering defaults, not empirically selected hyperparameters."""
from dataclasses import asdict, dataclass
import math


STOPPING_FIELDS = (
    "stopping_mode", "stop_patience", "stop_weight_l1_tolerance",
    "stop_state_relative_tolerance", "stop_diagnostic_relative_tolerance",
    "stop_gain_tolerance",
)

SEARCH_FIELDS = ('search_policy', 'probe_slots', 'exploration_slots', 'counterfactual_queries')
ACCESS_FIELDS = ('access_policy', 'leaf_capacity', 'coarse_budget', 'node_budget',
    'access_exploration', 'access_heads', 'access_uncertainty', 'access_fine_cost',
    'access_solve_cost', 'access_seed', 'access_label_iterations',
    'access_label_tolerance', 'access_label_stability')


@dataclass(frozen=True)
class DQConfig:
    summaries: int = 64
    diagnostic_slots: int = 32
    key_dim: int = 32
    statistic_dim: int = 8
    rounds: int = 4                 # maximum in adaptive mode; exact in fixed mode
    stopping_mode: str = "adaptive"
    stop_patience: int = 2
    stop_weight_l1_tolerance: float = 1e-4
    stop_state_relative_tolerance: float = 1e-3
    stop_diagnostic_relative_tolerance: float = 1e-3
    stop_gain_tolerance: float = 1e-6
    fine_edge_budget: int = 1024
    search_policy: str = 'solver_feedback_v1'
    probe_slots: int = 8
    exploration_slots: int = 2
    counterfactual_queries: int = 2  # training-only sampled re-solves per round
    access_policy: str = 'hierarchical_v1'
    leaf_capacity: int = 64
    coarse_budget: int = 16384       # per decision, shared by slots and rounds
    node_budget: int = 32768         # actual node-value evaluations, separate
    access_exploration: float = .125
    access_heads: int = 3
    access_uncertainty: float = .25
    access_fine_cost: float = 4.
    access_solve_cost: float = 32.
    access_seed: int = 20261014
    access_label_iterations: int = 256  # comparison repeated at 2x this cap
    access_label_tolerance: float = 1e-4
    access_label_stability: float = 1e-6
    retrieval_depths: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    max_plans: int = 32
    anchors_per_query: int = 4
    repair_fraction: float = 0.25
    retrieval_mode: str = "auto"
    exact_scan_threshold: int = 4096
    group_probes: int = 4
    group_scan_cap: int = 512
    gain_scale: float = 0.02
    fine_edge_price: float = 1e-8
    cost_rate: float = 0.0005
    qp_rank: int = 8
    qp_ridge: float = 0.05
    qp_iterations: int = 48
    qp_rho: float = 0.5
    qp_tolerance: float = 1e-4
    minimum_side_budget: float = 0.02
    short_budget_policy: str = 'legacy_complement_v1'
    long_policy: str = 'off'
    long_technology_policy: str = 'off'
    long_turnover_penalty: float = 0.
    empty_admission_policy: str = 'reallocate_long_then_cash_v1'

    def validate(self):
        if (self.short_budget_policy != 'legacy_complement_v1'
                or self.empty_admission_policy != 'reallocate_long_then_cash_v1'
                or self.long_policy != 'off' or self.long_technology_policy != 'off'
                or isinstance(self.long_turnover_penalty, bool) or self.long_turnover_penalty != 0.):
            raise ValueError('Checkpoint or requested portfolio policy differs from the Full US configuration.')
        integers = ("summaries", "diagnostic_slots", "key_dim", "statistic_dim",
                    "rounds", "max_plans", "anchors_per_query", "exact_scan_threshold",
                    "group_probes", "group_scan_cap", "qp_rank", "qp_iterations", "stop_patience")
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.diagnostic_slots % 2:
            raise ValueError("diagnostic_slots must be even: half per book side")
        if self.stopping_mode not in {"adaptive", "fixed"}:
            raise ValueError("stopping_mode must be adaptive or fixed")
        for name in STOPPING_FIELDS[2:]:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.fine_edge_budget) is not int or self.fine_edge_budget < 0:
            raise ValueError("fine_edge_budget must be a nonnegative integer")
        if self.search_policy not in {'independent_v1', 'solver_feedback_v1'}:
            raise ValueError('unknown search policy')
        if self.access_policy not in {'legacy_v1', 'hierarchical_v1'}:
            raise ValueError('unknown access policy')
        if self.access_policy == 'hierarchical_v1' and self.search_policy != 'solver_feedback_v1':
            raise ValueError('hierarchical access requires coupled solver feedback')
        if self.access_policy == 'hierarchical_v1' and self.retrieval_mode == 'exact':
            raise ValueError('exact reference requires explicit legacy_v1 access policy')
        for name in ('leaf_capacity', 'access_heads', 'access_label_iterations'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('coarse_budget', 'node_budget', 'access_seed'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if isinstance(self.access_exploration, bool) or not 0 < self.access_exploration <= 1:
            raise ValueError('access_exploration must lie in (0,1]')
        for name in ('access_uncertainty', 'access_fine_cost', 'access_solve_cost'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('access_label_tolerance', 'access_label_stability'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.probe_slots) is not int or self.probe_slots < 1:
            raise ValueError('probe_slots must be positive')
        for name in ('exploration_slots', 'counterfactual_queries'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.exploration_slots > self.probe_slots:
            raise ValueError('exploration_slots cannot exceed probe_slots')
        if (len(self.retrieval_depths) < 2 or self.retrieval_depths[0] != 0
                or any(type(x) is not int or x < 0 for x in self.retrieval_depths)
                or any(a >= b for a, b in zip(self.retrieval_depths, self.retrieval_depths[1:]))):
            raise ValueError("retrieval_depths must strictly increase from zero")
        if self.group_probes > self.summaries:
            raise ValueError("group_probes cannot exceed summaries")
        if self.group_scan_cap < self.retrieval_depths[-1]:
            raise ValueError("group_scan_cap must cover the largest retrieval depth")
        if self.retrieval_mode not in {"auto", "exact", "grouped"}:
            raise ValueError("unknown retrieval_mode")
        for name in ("repair_fraction", "gain_scale", "qp_ridge", "qp_rho", "qp_tolerance"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.repair_fraction > 1:
            raise ValueError("repair_fraction must not exceed one")
        for name in ("cost_rate", "fine_edge_price"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.minimum_side_budget < 0.5:
            raise ValueError("minimum_side_budget must lie in (0, 0.5)")
        return self

    def to_dict(self):
        return asdict(self)

    def stopping_spec(self):
        return dict(schema="dq.refinement_stopping.v1", max_rounds=self.rounds,
                    **{name: getattr(self, name) for name in STOPPING_FIELDS})

    def admission_spec(self):
        return dict(schema='dq.admission_fallback.v1', policy=self.empty_admission_policy,
            no_longs='cash', no_shorts='cash',
            unused_budget_moved_to_opposite_side=False, eligibility_relaxed=False,
            cash_return_assumption=0., fallback_dates_retained=True,
            cash_risk_budget_is_not_signed_ledger_cash_balance=True,
            invalid_data_or_numerical_errors_swallowed=False)

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        # Older checkpoints serialized inactive sizing fields. Only normalize
        # those fields for a checkpoint that explicitly used this allocation policy.
        inactive_fields = ('long_cap_fraction', 'long_entry_fraction', 'long_exit_fraction',
                           'long_technology_fraction', 'short_budget_cap', 'short_no_trade_band')
        if any(name in data for name in inactive_fields):
            if ('short_budget_cap' in data or 'short_no_trade_band' in data) and 'short_budget_policy' not in data:
                raise ValueError('Incomplete short budget policy in checkpoint.')
            if (data.get('long_policy') != 'off' or data.get('long_technology_policy') != 'off'
                    or data.get('long_turnover_penalty') != 0.
                    or data.get('short_budget_policy', 'legacy_complement_v1') != 'legacy_complement_v1'):
                raise ValueError('Checkpoint portfolio policy differs from the Full US configuration.')
            for name in inactive_fields:
                data.pop(name, None)
        access = set(ACCESS_FIELDS).intersection(data)
        if not access:
            data['access_policy'] = 'legacy_v1'
        elif access != set(ACCESS_FIELDS):
            raise ValueError('incomplete access policy in serialized DQ config')
        search = set(SEARCH_FIELDS).intersection(data)
        if not search:
            # Loading old tensors must not silently create a conditional value head.
            data['search_policy'] = 'independent_v1'
        elif search != set(SEARCH_FIELDS):
            raise ValueError('incomplete search policy in serialized DQ config')
        present = set(STOPPING_FIELDS).intersection(data)
        if not present:
            # Old v2 payloads had fixed one/two rounds. Do not reinterpret an
            # existing checkpoint using the new adaptive constructor defaults.
            data.setdefault("rounds", 2)
            if type(data["rounds"]) is not int or data["rounds"] not in (1, 2):
                raise ValueError("legacy DQ config supports only one or two fixed rounds")
            data["stopping_mode"] = "fixed"
        elif present != set(STOPPING_FIELDS):
            raise ValueError("incomplete stopping policy in serialized DQ config")
        if "retrieval_depths" in data:
            data["retrieval_depths"] = tuple(data["retrieval_depths"])
        return cls(**data).validate()
