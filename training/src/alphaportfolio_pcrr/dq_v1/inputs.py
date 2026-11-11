"""Explicit feature-name and point-in-time adapters; never infer from width."""
from datetime import datetime

import torch


DEFAULT_STATISTICS = ("return_21", "return_63", "volatility_21", "downside_21",
                      "drawdown_63", "log_market_cap", "spread_21", "amihud_21")


def select_diagnostic_statistics(daily_states, daily_observed, feature_names, *,
                                statistic_names=DEFAULT_STATISTICS):
    """Select current observed features as grounding proxies, not future labels.

    Existing daily29 caches contain cross-sectional NORMALIZED features. Values
    returned here retain that scale: e.g. spread_21 is NOT a tradable spread in
    dollars or percent. Execution constraints must use the raw quote ledger.
    The selected name list is separate from the number of learned diagnostic
    slots; using eight observed grounding proxies does not fix eight channels.
    """
    names, selected = tuple(feature_names), tuple(statistic_names)
    if (daily_states.ndim != 4 or len(names) != daily_states.shape[-1]
            or len(set(names)) != len(names) or len(set(selected)) != len(selected) or not selected):
        raise ValueError("invalid explicit feature schema")
    if any(name not in names for name in selected):
        raise ValueError("required diagnostic statistic missing from named feature schema")
    if (daily_observed.shape != daily_states.shape or daily_observed.dtype != torch.bool
            or daily_observed.device != daily_states.device):
        raise ValueError("invalid daily observed mask")
    columns = [names.index(name) for name in selected]
    values, known = daily_states[:, :, -1, columns], daily_observed[:, :, -1, columns]
    if not torch.isfinite(values[known]).all():
        raise ValueError("observed diagnostic statistics must be finite")
    return values.masked_fill(~known, 0), known


def validate_decision_clock(*, daily_available_at, monthly_available_at, decision_at, execution_at):
    """Validate declared availability, not the truth of a vendor's vintage.

    Inputs must be timezone-aware datetimes; execution must be strictly after
    the decision. A caller cannot manufacture availability by passing row dates.
    """
    times = (daily_available_at, monthly_available_at, decision_at, execution_at)
    if any(not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None for t in times):
        raise ValueError("timezone-aware availability and execution times required")
    if not (max(daily_available_at, monthly_available_at) <= decision_at < execution_at):
        raise ValueError("features must be available before decision, execution strictly later")
    return True
