"""Auditable valuation policies for CIZ next-month outcomes.

The investment universe is frozen from information available at formation
month ``t``.  This module is deliberately downstream of that decision: CIZ
quality flags from ``t+1`` may change how an already-selected outcome is
valued, but may never remove the asset from the formation universe.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import numpy as np
import pandas as pd

from .assemble import OUTCOME_ROW_MISSING_CATEGORY, validate_outcome_quality_contract


OUTCOME_VALUATION_POLICY_SCHEMA_VERSION = 1
OutcomePolicyName = Literal[
    "strict_fail",
    "raw_ciz",
    "carry_forward_zero_only_for_missing_period",
    "delist_stress",
]


class OutcomeValuationError(RuntimeError):
    """The selected outcome policy cannot value every required observation."""


@dataclass(frozen=True)
class OutcomeValuationResult:
    """Policy-valued returns plus masks that preserve the raw-data distinction."""

    returns: np.ndarray
    raw_observed: np.ndarray
    valued: np.ndarray
    adjusted: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class OutcomeValuationPolicy:
    """A hashable CIZ outcome-valuation contract.

    ``g_shock``, ``m_shock``, and ``v_shock`` are *residual terminal shocks*.
    For a raw CIZ return ``r`` and residual shock ``s``, the stress policy uses
    ``(1 + r) * (1 + s) - 1``.  Flags ``A`` and ``P`` are never shocked because
    the project's CIZ contract marks their delisting outcome as fully included.

    ``missing_period_return`` is separate and explicit.  Setting it to ``0``
    is the carry-forward valuation assumption for ``GP``/``NT`` rows; it is
    never selected by the formal default.
    """

    name: OutcomePolicyName = "strict_fail"
    g_shock: float | None = None
    m_shock: float | None = None
    v_shock: float | None = None
    missing_period_return: float | None = None

    def __post_init__(self) -> None:
        allowed = {
            "strict_fail",
            "raw_ciz",
            "carry_forward_zero_only_for_missing_period",
            "delist_stress",
        }
        if self.name not in allowed:
            raise ValueError(f"unknown outcome valuation policy: {self.name!r}")
        shocks = (self.g_shock, self.m_shock, self.v_shock)
        if self.name == "delist_stress":
            if any(value is None for value in shocks):
                raise ValueError(
                    "delist_stress requires explicit g_shock, m_shock, and v_shock"
                )
            for label, value in zip(("g_shock", "m_shock", "v_shock"), shocks):
                _validate_terminal_return(value, label)
            if self.missing_period_return is not None:
                _validate_terminal_return(
                    self.missing_period_return,
                    "missing_period_return",
                )
        elif any(value is not None for value in (*shocks, self.missing_period_return)):
            raise ValueError(
                f"{self.name} does not accept delisting or missing-period parameters"
            )

    @classmethod
    def strict_fail(cls) -> "OutcomeValuationPolicy":
        return cls(name="strict_fail")

    @classmethod
    def raw_ciz(cls) -> "OutcomeValuationPolicy":
        return cls(name="raw_ciz")

    @classmethod
    def carry_forward_zero_only_for_missing_period(
        cls,
    ) -> "OutcomeValuationPolicy":
        return cls(name="carry_forward_zero_only_for_missing_period")

    @classmethod
    def delist_stress(
        cls,
        *,
        g_shock: float,
        m_shock: float,
        v_shock: float,
        missing_period_return: float | None = None,
    ) -> "OutcomeValuationPolicy":
        return cls(
            name="delist_stress",
            g_shock=float(g_shock),
            m_shock=float(m_shock),
            v_shock=float(v_shock),
            missing_period_return=(
                None
                if missing_period_return is None
                else float(missing_period_return)
            ),
        )

    @property
    def policy_id(self) -> str:
        return str(self.receipt()["policy_id"])

    def receipt(self) -> dict[str, object]:
        """Return a canonical, hash-bound disclosure of the valuation rule."""

        payload: dict[str, object] = {
            "schema_version": OUTCOME_VALUATION_POLICY_SCHEMA_VERSION,
            "policy_name": self.name,
            "parameters": {
                "g_residual_terminal_shock": self.g_shock,
                "m_residual_terminal_shock": self.m_shock,
                "v_residual_terminal_shock": self.v_shock,
                "missing_period_return": self.missing_period_return,
            },
            "raw_return_source": "official_ciz_mthret_at_t_plus_1",
            "formation_universe_information_cutoff": "formation_month_t",
            "eligibility_uses_t_plus_1_quality": False,
            "full_delisting_already_included_flags": ["A", "P"],
            "stress_adjusted_delisting_flags": (
                ["G", "M", "V"] if self.name == "delist_stress" else []
            ),
            "delist_stress_formula": (
                "(1 + raw_or_explicit_missing_period_return) "
                "* (1 + residual_terminal_shock) - 1"
                if self.name == "delist_stress"
                else None
            ),
            "missing_period_zero_is_explicit_valuation_assumption": (
                self.name == "carry_forward_zero_only_for_missing_period"
                or (
                    self.name == "delist_stress"
                    and self.missing_period_return == 0.0
                )
            ),
            "raw_ciz_may_remain_nonfinite": self.name == "raw_ciz",
            "formal_default": self.name == "strict_fail",
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {**payload, "policy_id": digest}

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "next_return",
        require_quality_contract: bool = True,
    ) -> OutcomeValuationResult:
        """Value already-selected outcomes without changing row membership."""

        if return_column not in frame:
            raise ValueError(f"outcome frame lacks {return_column!r}")
        if require_quality_contract:
            validate_outcome_quality_contract(
                frame,
                required=True,
                context="outcome valuation input",
            )

        raw = pd.to_numeric(frame[return_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        values = raw.copy()
        raw_observed = np.isfinite(raw)
        adjusted = np.zeros(len(frame), dtype=bool)

        if not require_quality_contract:
            if self.name not in {"strict_fail", "raw_ciz"}:
                raise OutcomeValuationError(
                    f"{self.name} requires the complete CIZ outcome-quality contract"
                )
            if self.name == "strict_fail" and not raw_observed.all():
                _raise_unvalued(
                    self,
                    raw_missing=~raw_observed,
                    censored=np.zeros(len(frame), dtype=bool),
                    partial=np.zeros(len(frame), dtype=bool),
                    missing_period=np.zeros(len(frame), dtype=bool),
                    outcome_row_missing=np.zeros(len(frame), dtype=bool),
                )
            return _result(self, values, raw_observed, adjusted)

        delflag = frame["next_return_mthdelflg"].astype("string")
        retflag = frame["next_return_mthretflg"].astype("string")
        retcategory = frame["next_return_mthretflg_category"].astype("string")
        # Pandas' nullable StringDtype preserves ``pd.NA`` for a physically
        # absent next-month outcome row.  Such an NA is not any delisting or
        # return flag; normalize the comparison masks explicitly instead of
        # relying on a version-dependent nullable-to-NumPy conversion.
        g_mask = delflag.eq("G").fillna(False).to_numpy(dtype=bool)
        m_mask = delflag.eq("M").fillna(False).to_numpy(dtype=bool)
        v_mask = delflag.eq("V").fillna(False).to_numpy(dtype=bool)
        censored = g_mask | m_mask
        missing_period = (
            retflag.isin(("GP", "NT")).fillna(False).to_numpy(dtype=bool)
        )
        outcome_row_missing = retcategory.eq(
            OUTCOME_ROW_MISSING_CATEGORY
        ).fillna(False).to_numpy(dtype=bool)

        if self.name == "raw_ciz":
            return _result(self, values, raw_observed, adjusted)

        if self.name == "strict_fail":
            ambiguous = (
                ~raw_observed
                | censored
                | v_mask
                | missing_period
                | outcome_row_missing
            )
            if ambiguous.any():
                _raise_unvalued(
                    self,
                    raw_missing=~raw_observed,
                    censored=censored,
                    partial=v_mask,
                    missing_period=missing_period,
                    outcome_row_missing=outcome_row_missing,
                )
            return _result(self, values, raw_observed, adjusted)

        if self.name == "carry_forward_zero_only_for_missing_period":
            fill = ~raw_observed & missing_period
            values[fill] = 0.0
            adjusted[fill] = True
            remaining = ~np.isfinite(values)
            if remaining.any():
                _raise_unvalued(
                    self,
                    raw_missing=remaining,
                    censored=censored,
                    partial=v_mask,
                    missing_period=missing_period,
                    outcome_row_missing=outcome_row_missing,
                )
            return _result(self, values, raw_observed, adjusted)

        # delist_stress: first make an explicitly requested missing-period
        # valuation, then compound only G/M/V residual terminal shocks.
        assert self.name == "delist_stress"
        if self.missing_period_return is not None:
            fill = ~raw_observed & missing_period
            values[fill] = float(self.missing_period_return)
            adjusted[fill] = True
        remaining = ~np.isfinite(values)
        if remaining.any():
            _raise_unvalued(
                self,
                raw_missing=remaining,
                censored=censored,
                partial=v_mask,
                missing_period=missing_period,
                outcome_row_missing=outcome_row_missing,
            )
        for mask, shock in (
            (g_mask, self.g_shock),
            (m_mask, self.m_shock),
            (v_mask, self.v_shock),
        ):
            if mask.any():
                values[mask] = (1.0 + values[mask]) * (1.0 + float(shock)) - 1.0
                adjusted[mask] = True
        return _result(self, values, raw_observed, adjusted)


STRICT_FAIL_OUTCOME_POLICY = OutcomeValuationPolicy.strict_fail()


def coerce_outcome_policy(
    value: OutcomeValuationPolicy | OutcomePolicyName | None,
) -> OutcomeValuationPolicy:
    """Normalize public APIs while keeping ``strict_fail`` the only default."""

    if value is None:
        return STRICT_FAIL_OUTCOME_POLICY
    if isinstance(value, OutcomeValuationPolicy):
        return value
    if value == "strict_fail":
        return STRICT_FAIL_OUTCOME_POLICY
    if value == "raw_ciz":
        return OutcomeValuationPolicy.raw_ciz()
    if value == "carry_forward_zero_only_for_missing_period":
        return OutcomeValuationPolicy.carry_forward_zero_only_for_missing_period()
    if value == "delist_stress":
        raise ValueError(
            "delist_stress must be constructed with explicit G/M/V parameters"
        )
    raise ValueError(f"unknown outcome valuation policy: {value!r}")


def _result(
    policy: OutcomeValuationPolicy,
    values: np.ndarray,
    raw_observed: np.ndarray,
    adjusted: np.ndarray,
) -> OutcomeValuationResult:
    valued = np.isfinite(values)
    return OutcomeValuationResult(
        returns=values,
        raw_observed=raw_observed,
        valued=valued,
        adjusted=adjusted,
        audit={
            "policy_id": policy.policy_id,
            "policy_name": policy.name,
            "rows": int(values.size),
            "raw_observed_rows": int(raw_observed.sum()),
            "raw_missing_rows": int((~raw_observed).sum()),
            "policy_adjusted_rows": int(adjusted.sum()),
            "valued_rows": int(valued.sum()),
            "unvalued_rows": int((~valued).sum()),
        },
    )


def _raise_unvalued(
    policy: OutcomeValuationPolicy,
    *,
    raw_missing: np.ndarray,
    censored: np.ndarray,
    partial: np.ndarray,
    missing_period: np.ndarray,
    outcome_row_missing: np.ndarray,
) -> None:
    counts = {
        "raw_missing": int(raw_missing.sum()),
        "delisting_g_or_m_censored": int(censored.sum()),
        "delisting_v_partial": int(partial.sum()),
        "mthretflg_gp_or_nt": int(missing_period.sum()),
        "outcome_row_missing": int(outcome_row_missing.sum()),
    }
    raise OutcomeValuationError(
        f"outcome policy {policy.name!r} cannot value the selected rows: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )


def _validate_terminal_return(value: float | None, label: str) -> None:
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    if not -1.0 <= float(value) <= 0.0:
        raise ValueError(f"{label} must lie in [-1, 0]")
