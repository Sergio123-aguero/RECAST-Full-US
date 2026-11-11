"""Sparse monthly panel store and leakage-safe episode construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .assemble import OUTCOME_QUALITY_COLUMNS
from .eligibility import FormationUniverse
from .outcome_policy import (
    OutcomePolicyName,
    OutcomeValuationError,
    OutcomeValuationPolicy,
    coerce_outcome_policy,
)
from .schema import FEATURE_NAMES


MissingReturnPolicy = Literal["fail", "zero_with_disclosure"]


class MissingRealizedReturnError(RuntimeError):
    """A formation-eligible asset lacks its realized next-month outcome."""


@dataclass(frozen=True)
class PanelSnapshot:
    formation_month: pd.Period
    permnos: np.ndarray
    states: Tensor
    next_returns: Tensor
    return_observed: Tensor | None = None
    return_valued: Tensor | None = None
    return_policy_adjusted: Tensor | None = None
    missing_return_policy: str = "strict_fail"
    outcome_policy_name: str = "strict_fail"
    outcome_policy_id: str = ""
    outcome_policy_audit: dict[str, object] | None = None
    shared_universe_id: str = ""
    realized_return_policy_id: str = ""

    @property
    def n_assets(self) -> int:
        return int(self.states.shape[0])


class MonthlyPanelStore:
    """A memory-conscious store for the unbalanced CRSP/Compustat panel."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_names: Iterable[str] = FEATURE_NAMES,
        lookback_months: int = 12,
        date_column: str = "month",
        id_column: str = "permno",
        next_return_column: str = "next_return",
        formation_universe: FormationUniverse | None = None,
        outcome_policy: OutcomeValuationPolicy | OutcomePolicyName | None = None,
        missing_return_policy: MissingReturnPolicy | None = None,
        realized_return_semantics: str = "unspecified",
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.lookback_months = int(lookback_months)
        self.date_column = date_column
        self.id_column = id_column
        self.next_return_column = next_return_column
        if outcome_policy is not None and missing_return_policy is not None:
            raise ValueError(
                "specify outcome_policy, not both outcome_policy and the legacy "
                "missing_return_policy"
            )
        if missing_return_policy == "zero_with_disclosure":
            raise ValueError(
                "zero_with_disclosure is ambiguous and no longer supported; use "
                "the explicit carry_forward_zero_only_for_missing_period policy"
            )
        if missing_return_policy not in (None, "fail"):
            raise ValueError("legacy missing_return_policy may only be 'fail'")
        self.outcome_policy = coerce_outcome_policy(outcome_policy)
        self.outcome_policy_id = self.outcome_policy.policy_id
        self.missing_return_policy = self.outcome_policy.name
        self.realized_return_semantics = str(realized_return_semantics)
        if self.lookback_months < 2:
            raise ValueError("lookback_months must be at least two")
        required = {
            date_column,
            id_column,
            next_return_column,
            *self.feature_names,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"panel missing columns: {missing}")

        present_quality = set(OUTCOME_QUALITY_COLUMNS).intersection(frame.columns)
        if present_quality and present_quality != set(OUTCOME_QUALITY_COLUMNS):
            missing_quality = sorted(
                set(OUTCOME_QUALITY_COLUMNS) - set(frame.columns)
            )
            raise ValueError(
                "panel has a partial outcome-quality contract; missing: "
                f"{missing_quality}"
            )
        self.has_outcome_quality_contract = bool(present_quality)
        quality_columns = (
            list(OUTCOME_QUALITY_COLUMNS)
            if self.has_outcome_quality_contract
            else []
        )
        data = frame.loc[
            :,
            [
                date_column,
                id_column,
                *self.feature_names,
                next_return_column,
                *quality_columns,
            ],
        ].copy()
        if not isinstance(data[date_column].dtype, pd.PeriodDtype):
            data[date_column] = pd.to_datetime(data[date_column]).dt.to_period("M")
        if data.duplicated([date_column, id_column]).any():
            raise ValueError("panel contains duplicate asset-month rows")
        data = data.sort_values([date_column, id_column])
        self._by_month: dict[pd.Period, pd.DataFrame] = {
            month: group.set_index(id_column).sort_index()
            for month, group in data.groupby(date_column, sort=True, observed=True)
        }
        first = data.groupby(id_column, observed=True)[date_column].min()
        self._first_month = first.to_dict()
        self.months = tuple(sorted(self._by_month))
        if formation_universe is None:
            formation_universe = FormationUniverse.from_panel(
                data,
                feature_names=self.feature_names,
                lookback_months=self.lookback_months,
                date_column=self.date_column,
                id_column=self.id_column,
            )
        else:
            formation_universe.assert_compatible_panel(
                data,
                feature_names=self.feature_names,
                lookback_months=self.lookback_months,
                date_column=self.date_column,
                id_column=self.id_column,
            )
        self.formation_universe = formation_universe
        self.shared_universe_id = formation_universe.universe_id
        self.realized_return_policy_id = self.realized_return_policy_receipt()[
            "receipt_id"
        ]

    def snapshot(
        self,
        formation_month: str | pd.Period,
        *,
        mode: Literal["train", "oos"] = "train",
        minimum_history_months: int | None = None,
        device: str | torch.device = "cpu",
    ) -> PanelSnapshot:
        month = _as_month(formation_month)
        history = tuple(
            month - offset
            for offset in range(self.lookback_months - 1, -1, -1)
        )
        missing_months = [item for item in history if item not in self._by_month]
        if missing_months:
            raise KeyError(f"missing lookback months for {month}: {missing_months}")
        if minimum_history_months is None:
            minimum_history_months = 24 if mode == "train" else 12
        permnos = self.eligible_permnos(
            month,
            minimum_history_months=minimum_history_months,
        )
        if permnos.size == 0:
            raise ValueError(f"no eligible assets for {month}")

        state_array = np.stack(
            [
                self._by_month[item].loc[permnos, self.feature_names].to_numpy(
                    dtype=np.float32,
                )
                for item in history
            ],
            axis=1,
        )
        current = self._by_month[month]
        outcome_columns = [self.next_return_column]
        if self.has_outcome_quality_contract:
            outcome_columns.extend(OUTCOME_QUALITY_COLUMNS)
        try:
            valued = self.outcome_policy.apply(
                current.loc[permnos, outcome_columns].reset_index(drop=True),
                return_column=self.next_return_column,
                require_quality_contract=self.has_outcome_quality_contract,
            )
        except OutcomeValuationError as exc:
            raise MissingRealizedReturnError(
                f"formation-eligible outcomes in {month} failed the "
                f"{self.outcome_policy.name!r} valuation policy; the shared "
                "universe was not filtered on t+1 outcome quality: {exc}"
            ) from exc
        next_returns = valued.returns.astype(np.float32, copy=False)
        observed = valued.raw_observed
        return PanelSnapshot(
            formation_month=month,
            permnos=permnos,
            states=torch.as_tensor(state_array, device=device),
            next_returns=torch.as_tensor(next_returns, device=device),
            return_observed=torch.as_tensor(observed, device=device),
            return_valued=torch.as_tensor(valued.valued, device=device),
            return_policy_adjusted=torch.as_tensor(
                valued.adjusted,
                device=device,
            ),
            missing_return_policy=self.missing_return_policy,
            outcome_policy_name=self.outcome_policy.name,
            outcome_policy_id=self.outcome_policy_id,
            outcome_policy_audit=valued.audit,
            shared_universe_id=self.shared_universe_id,
            realized_return_policy_id=self.realized_return_policy_id,
        )

    def eligible_permnos(
        self,
        formation_month: str | pd.Period,
        *,
        minimum_history_months: int,
    ) -> np.ndarray:
        """Return the t-information universe without consulting outcomes."""

        month = _as_month(formation_month)
        history = tuple(
            month - offset
            for offset in range(self.lookback_months - 1, -1, -1)
        )
        missing_months = [item for item in history if item not in self._by_month]
        if missing_months:
            raise KeyError(f"missing lookback months for {month}: {missing_months}")
        return self.formation_universe.permnos_for_month(
            month,
            minimum_history_months=minimum_history_months,
        )

    def universe_mask(
        self,
        formation_month: str | pd.Period,
        candidate_permnos: Iterable[int],
        *,
        minimum_history_months: int,
    ) -> np.ndarray:
        """Return the exact shared mask that all candidate models must reuse."""

        return self.formation_universe.mask_for_ids(
            formation_month,
            candidate_permnos,
            minimum_history_months=minimum_history_months,
        )

    def realized_return_policy_receipt(self) -> dict[str, object]:
        """Small hash-ready disclosure for a run/config receipt."""

        policy: dict[str, object] = {
            "outcome_policy": self.outcome_policy.receipt(),
            "return_column": self.next_return_column,
            "return_semantics": self.realized_return_semantics,
            "eligibility_uses_realized_return": False,
            "eligibility_uses_t_plus_1_quality": False,
            "outcome_quality_contract_present": (
                self.has_outcome_quality_contract
            ),
            "shared_universe_id": self.shared_universe_id,
        }
        policy["receipt_id"] = hashlib.sha256(
            json.dumps(
                policy,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return policy

    def trajectory(
        self,
        start_formation_month: str | pd.Period,
        *,
        horizon_months: int = 12,
        mode: Literal["train", "oos"] = "train",
        minimum_history_months: int | None = None,
        minimum_assets: int = 20,
        device: str | torch.device = "cpu",
    ) -> list[PanelSnapshot]:
        start = _as_month(start_formation_month)
        result = [
            self.snapshot(
                start + step,
                mode=mode,
                minimum_history_months=minimum_history_months,
                device=device,
            )
            for step in range(horizon_months)
        ]
        if any(snapshot.n_assets < minimum_assets for snapshot in result):
            raise ValueError(
                f"trajectory starting {start} falls below {minimum_assets} assets"
            )
        return result

    def valid_episode_starts(
        self,
        start: str | pd.Period,
        end: str | pd.Period,
        *,
        horizon_months: int = 12,
        mode: Literal["train", "oos"] = "train",
        minimum_history_months: int | None = None,
        minimum_assets: int = 20,
    ) -> tuple[pd.Period, ...]:
        first = _as_month(start)
        last = _as_month(end)
        candidates = pd.period_range(first, last, freq="M")
        valid: list[pd.Period] = []
        for candidate in candidates:
            try:
                self.trajectory(
                    candidate,
                    horizon_months=horizon_months,
                    mode=mode,
                    minimum_history_months=minimum_history_months,
                    minimum_assets=minimum_assets,
                )
            except (KeyError, ValueError):
                continue
            valid.append(candidate)
        return tuple(valid)


def _as_month(value: str | pd.Period) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(value, freq="M")
