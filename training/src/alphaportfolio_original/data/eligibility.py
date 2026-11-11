"""Formation-time-only, model-shared investment-universe artifacts.

The investment universe at formation month ``t`` must be determined from
information available no later than ``t``.  In particular, whether the
realized ``t+1`` return happens to be present is an outcome-data property and
must never be used to decide whether an asset was investable at ``t``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


UNIVERSE_SCHEMA_VERSION = 1
MEMBERSHIP_COLUMNS = ("month", "permno", "observed_history_months")
FORMATION_ELIGIBILITY_COLUMN = "formation_month_end_eligible"


class FormationUniverse:
    """Immutable-by-contract membership shared by every compared model.

    The stored rows have already passed the feature-history rule.  A caller
    may additionally apply a formation-time minimum listing-history threshold
    through :meth:`permnos_for_month`.  This permits one frozen artifact to be
    shared by training and OOS code even when their minimum-history thresholds
    differ.
    """

    def __init__(
        self,
        membership: pd.DataFrame,
        *,
        feature_names: Iterable[str],
        lookback_months: int,
        formation_input_sha256: str,
        source_months: Iterable[str | pd.Period],
        require_formation_eligibility: bool = False,
        formation_eligibility_column: str = FORMATION_ELIGIBILITY_COLUMN,
        universe_id: str | None = None,
        artifact_sha256: str | None = None,
        receipt_sha256: str | None = None,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.lookback_months = int(lookback_months)
        self.require_formation_eligibility = bool(
            require_formation_eligibility
        )
        self.formation_eligibility_column = str(
            formation_eligibility_column
        )
        if self.lookback_months < 2:
            raise ValueError("lookback_months must be at least two")
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        if len(formation_input_sha256) != 64:
            raise ValueError("formation_input_sha256 must be a SHA-256 digest")

        data = membership.loc[:, MEMBERSHIP_COLUMNS].copy()
        if not isinstance(data["month"].dtype, pd.PeriodDtype):
            data["month"] = pd.PeriodIndex(
                data["month"].astype("string"),
                freq="M",
            )
        data["permno"] = pd.to_numeric(
            data["permno"], errors="raise"
        ).astype(np.int64)
        data["observed_history_months"] = pd.to_numeric(
            data["observed_history_months"], errors="raise"
        ).astype(np.int64)
        if data.duplicated(["month", "permno"]).any():
            raise ValueError("formation universe contains duplicate asset-month rows")
        if (data["observed_history_months"] < 1).any():
            raise ValueError("observed_history_months must be positive")
        data = data.sort_values(["month", "permno"]).reset_index(drop=True)

        self._membership = data
        self._by_month = {
            month: group.set_index("permno")["observed_history_months"]
            for month, group in data.groupby("month", sort=True, observed=True)
        }
        self.source_months = tuple(
            sorted({_as_month(value) for value in source_months})
        )
        self.formation_input_sha256 = formation_input_sha256
        self.membership_content_sha256 = _membership_sha256(data)
        expected_id = _universe_id(
            feature_names=self.feature_names,
            lookback_months=self.lookback_months,
            formation_input_sha256=self.formation_input_sha256,
            membership_content_sha256=self.membership_content_sha256,
            require_formation_eligibility=(
                self.require_formation_eligibility
            ),
            formation_eligibility_column=(
                self.formation_eligibility_column
            ),
        )
        if universe_id is not None and universe_id != expected_id:
            raise ValueError("formation universe id does not match its contents")
        self.universe_id = expected_id
        self.artifact_sha256 = artifact_sha256
        self.receipt_sha256 = receipt_sha256

    @classmethod
    def from_panel(
        cls,
        frame: pd.DataFrame,
        *,
        feature_names: Iterable[str],
        lookback_months: int = 12,
        date_column: str = "month",
        id_column: str = "permno",
        require_formation_eligibility: bool = False,
        formation_eligibility_column: str = FORMATION_ELIGIBILITY_COLUMN,
    ) -> "FormationUniverse":
        """Build membership using only ``id``, month, and features through t."""

        features = tuple(feature_names)
        data = _formation_inputs(
            frame,
            feature_names=features,
            date_column=date_column,
            id_column=id_column,
            require_formation_eligibility=require_formation_eligibility,
            formation_eligibility_column=formation_eligibility_column,
        )
        source_months = tuple(sorted(data["month"].unique()))
        construction_data = (
            data[data[formation_eligibility_column]].copy()
            if require_formation_eligibility
            else data
        )
        observed = (
            construction_data.loc[:, ["permno", "month"]]
            .drop_duplicates()
            .sort_values(["permno", "month"])
        )
        observed["observed_history_months"] = (
            observed.groupby("permno", observed=True).cumcount() + 1
        )
        observed_history = observed.set_index(
            ["month", "permno"]
        )["observed_history_months"]
        by_month = {
            month: group.set_index("permno").sort_index()
            for month, group in construction_data.groupby(
                "month", sort=True, observed=True
            )
        }
        records: list[pd.DataFrame] = []
        for month in source_months:
            history = tuple(
                month - offset
                for offset in range(int(lookback_months) - 1, -1, -1)
            )
            if any(item not in by_month for item in history):
                continue
            eligible: set[int] | None = None
            for item in history:
                values = by_month[item].loc[:, features].to_numpy(
                    dtype=np.float64
                )
                finite_ids = set(
                    int(value)
                    for value in by_month[item].index[
                        np.isfinite(values).all(axis=1)
                    ]
                )
                eligible = (
                    finite_ids
                    if eligible is None
                    else eligible.intersection(finite_ids)
                )
            if not eligible:
                continue
            permnos = np.asarray(sorted(eligible), dtype=np.int64)
            observed = np.asarray(
                [
                    int(observed_history.loc[(month, int(asset))])
                    for asset in permnos
                ],
                dtype=np.int64,
            )
            records.append(
                pd.DataFrame(
                    {
                        "month": month,
                        "permno": permnos,
                        "observed_history_months": observed,
                    }
                )
            )
        membership = (
            pd.concat(records, ignore_index=True)
            if records
            else pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
        )
        return cls(
            membership,
            feature_names=features,
            lookback_months=lookback_months,
            formation_input_sha256=formation_input_sha256(
                frame,
                feature_names=features,
                date_column=date_column,
                id_column=id_column,
                require_formation_eligibility=(
                    require_formation_eligibility
                ),
                formation_eligibility_column=(
                    formation_eligibility_column
                ),
            ),
            source_months=source_months,
            require_formation_eligibility=require_formation_eligibility,
            formation_eligibility_column=formation_eligibility_column,
        )

    @property
    def membership(self) -> pd.DataFrame:
        """Return a defensive copy of the model-shared membership table."""

        return self._membership.copy()

    def assert_compatible_panel(
        self,
        frame: pd.DataFrame,
        *,
        feature_names: Iterable[str],
        lookback_months: int,
        date_column: str = "month",
        id_column: str = "permno",
    ) -> None:
        features = tuple(feature_names)
        if features != self.feature_names:
            raise ValueError("shared universe feature order differs from the panel")
        if int(lookback_months) != self.lookback_months:
            raise ValueError("shared universe lookback differs from the panel")
        actual = formation_input_sha256(
            frame,
            feature_names=features,
            date_column=date_column,
            id_column=id_column,
            require_formation_eligibility=(
                self.require_formation_eligibility
            ),
            formation_eligibility_column=(
                self.formation_eligibility_column
            ),
        )
        if actual != self.formation_input_sha256:
            raise ValueError(
                "shared universe was not built from this panel's "
                "formation-time inputs"
            )

    def permnos_for_month(
        self,
        formation_month: str | pd.Period,
        *,
        minimum_history_months: int,
    ) -> np.ndarray:
        if minimum_history_months < 1:
            raise ValueError("minimum_history_months must be positive")
        month = _as_month(formation_month)
        history = self._by_month.get(month)
        if history is None:
            return np.asarray([], dtype=np.int64)
        return history.index[
            history.to_numpy() >= int(minimum_history_months)
        ].to_numpy(dtype=np.int64)

    def mask_for_ids(
        self,
        formation_month: str | pd.Period,
        candidate_permnos: Iterable[int],
        *,
        minimum_history_months: int,
    ) -> np.ndarray:
        allowed = self.permnos_for_month(
            formation_month,
            minimum_history_months=minimum_history_months,
        )
        candidates = np.asarray(tuple(candidate_permnos), dtype=np.int64)
        return np.isin(candidates, allowed, assume_unique=False)


def formation_input_sha256(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    date_column: str = "month",
    id_column: str = "permno",
    require_formation_eligibility: bool = False,
    formation_eligibility_column: str = FORMATION_ELIGIBILITY_COLUMN,
) -> str:
    """Hash only formation-time columns; outcome labels are deliberately absent."""

    data = _formation_inputs(
        frame,
        feature_names=tuple(feature_names),
        date_column=date_column,
        id_column=id_column,
        require_formation_eligibility=require_formation_eligibility,
        formation_eligibility_column=formation_eligibility_column,
    )
    return _frame_sha256(data)


def freeze_formation_universe(
    universe: FormationUniverse,
    output_path: str | Path,
    *,
    source_panel_path: str | Path | None = None,
    source_panel_receipt_path: str | Path | None = None,
) -> Path:
    """Atomically write membership plus a hash-bound, auditable receipt."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = universe.membership
    serializable["month"] = serializable["month"].astype(str)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    serializable.to_parquet(temporary, index=False)
    temporary.replace(target)
    artifact_sha256 = _file_sha256(target)
    receipt_path = target.with_suffix(".receipt.json")
    source_panel = (
        _file_reference(Path(source_panel_path))
        if source_panel_path is not None
        else None
    )
    source_receipt = (
        _file_reference(Path(source_panel_receipt_path))
        if source_panel_receipt_path is not None
        else None
    )
    payload = {
        "status": "complete",
        "schema_version": UNIVERSE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "formation_time_shared_investment_universe",
        "licensed_data_publication_allowed": False,
        "universe_id": universe.universe_id,
        "eligibility_rule": {
            "information_cutoff": "formation_month_t",
            "formation_columns": [
                "month",
                "permno",
                *(
                    [universe.formation_eligibility_column]
                    if universe.require_formation_eligibility
                    else []
                ),
                *universe.feature_names,
            ],
            "forbidden_outcome_columns": ["next_return", "t_plus_1_return"],
            "feature_history_rule": (
                "finite features in every month t-lookback+1 through t"
            ),
            "minimum_history_rule": (
                "applied at use time from cumulative observed asset-months; "
                "not elapsed calendar age"
            ),
            "lookback_months": universe.lookback_months,
            "feature_names": list(universe.feature_names),
            "formation_eligibility_required": (
                universe.require_formation_eligibility
            ),
            "formation_eligibility_column": (
                universe.formation_eligibility_column
            ),
            "formation_eligibility_rule": (
                "only rows equal to True may enter membership construction "
                "or lookback-window intersections"
                if universe.require_formation_eligibility
                else "not enforced for this exploratory artifact"
            ),
        },
        "formation_input_sha256": universe.formation_input_sha256,
        "membership_content_sha256": universe.membership_content_sha256,
        "rows": len(serializable),
        "months_with_members": int(serializable["month"].nunique()),
        "source_months": [str(value) for value in universe.source_months],
        "monthly_membership_counts": {
            str(key): int(value)
            for key, value in serializable.groupby(
                "month", observed=True
            ).size().items()
        },
        "sources": {
            "panel": source_panel,
            "panel_receipt": source_receipt,
        },
        "output": {
            "path": str(target.resolve()),
            "bytes": target.stat().st_size,
            "sha256": artifact_sha256,
        },
        "implementation_sha256": _file_sha256(Path(__file__).resolve()),
    }
    temporary_receipt = receipt_path.with_suffix(
        f"{receipt_path.suffix}.tmp"
    )
    temporary_receipt.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_receipt.replace(receipt_path)
    return receipt_path


def load_frozen_formation_universe(
    output_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> FormationUniverse:
    """Load an artifact only after verifying all receipt/content bindings."""

    target = Path(output_path)
    receipt = (
        Path(receipt_path)
        if receipt_path is not None
        else target.with_suffix(".receipt.json")
    )
    if not target.is_file() or not receipt.is_file():
        raise ValueError("formation universe artifact or receipt is missing")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("formation universe receipt is not complete")
    if payload.get("schema_version") != UNIVERSE_SCHEMA_VERSION:
        raise ValueError("unsupported formation universe schema version")
    output = payload.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("formation universe receipt has invalid output metadata")
    declared_path = Path(str(output.get("path", ""))).resolve()
    if declared_path != target.resolve():
        raise ValueError("formation universe receipt binds a different path")
    artifact_sha256 = _file_sha256(target)
    if output.get("sha256") != artifact_sha256:
        raise ValueError("formation universe artifact hash mismatch")

    rule = payload.get("eligibility_rule", {})
    if not isinstance(rule, dict):
        raise ValueError("formation universe receipt has invalid rule metadata")
    membership = pd.read_parquet(target)
    universe = FormationUniverse(
        membership,
        feature_names=rule.get("feature_names", ()),
        lookback_months=int(rule.get("lookback_months", 0)),
        formation_input_sha256=str(payload.get("formation_input_sha256", "")),
        source_months=payload.get("source_months", ()),
        require_formation_eligibility=bool(
            rule.get("formation_eligibility_required", False)
        ),
        formation_eligibility_column=str(
            rule.get(
                "formation_eligibility_column",
                FORMATION_ELIGIBILITY_COLUMN,
            )
        ),
        universe_id=str(payload.get("universe_id", "")),
        artifact_sha256=artifact_sha256,
        receipt_sha256=_file_sha256(receipt),
    )
    if universe.membership_content_sha256 != payload.get(
        "membership_content_sha256"
    ):
        raise ValueError("formation universe membership content hash mismatch")
    if len(universe.membership) != int(payload.get("rows", -1)):
        raise ValueError("formation universe row count differs from receipt")
    return universe


def _formation_inputs(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    date_column: str,
    id_column: str,
    require_formation_eligibility: bool,
    formation_eligibility_column: str,
) -> pd.DataFrame:
    required = {date_column, id_column, *feature_names}
    if require_formation_eligibility:
        required.add(formation_eligibility_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"panel missing formation columns: {missing}")
    columns = [
        date_column,
        id_column,
        *(
            [formation_eligibility_column]
            if require_formation_eligibility
            else []
        ),
        *feature_names,
    ]
    data = frame.loc[:, columns].copy()
    data = data.rename(columns={date_column: "month", id_column: "permno"})
    if not isinstance(data["month"].dtype, pd.PeriodDtype):
        data["month"] = pd.PeriodIndex(
            data["month"].astype("string"),
            freq="M",
        )
    data["permno"] = pd.to_numeric(data["permno"], errors="raise").astype(
        np.int64
    )
    if data.duplicated(["month", "permno"]).any():
        raise ValueError("panel contains duplicate asset-month rows")
    if require_formation_eligibility:
        eligibility = data[formation_eligibility_column]
        if eligibility.isna().any():
            raise ValueError(
                f"{formation_eligibility_column} contains missing values"
            )
        if not eligibility.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(
                f"{formation_eligibility_column} must be strictly boolean"
            )
        data[formation_eligibility_column] = eligibility.astype(bool)
    return data.sort_values(["month", "permno"]).reset_index(drop=True)


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    descriptor = {
        "columns": list(frame.columns),
        "dtypes": [str(value) for value in frame.dtypes],
        "rows": len(frame),
    }
    digest.update(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=False,
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _membership_sha256(frame: pd.DataFrame) -> str:
    serializable = frame.copy()
    serializable["month"] = serializable["month"].astype(str)
    return _frame_sha256(serializable)


def _universe_id(
    *,
    feature_names: tuple[str, ...],
    lookback_months: int,
    formation_input_sha256: str,
    membership_content_sha256: str,
    require_formation_eligibility: bool = False,
    formation_eligibility_column: str = FORMATION_ELIGIBILITY_COLUMN,
) -> str:
    payload = {
        "schema_version": UNIVERSE_SCHEMA_VERSION,
        "feature_names": list(feature_names),
        "lookback_months": int(lookback_months),
        "information_cutoff": "formation_month_t",
        "formation_input_sha256": formation_input_sha256,
        "membership_content_sha256": membership_content_sha256,
    }
    if require_formation_eligibility:
        payload["formation_eligibility_required"] = True
        payload["formation_eligibility_column"] = (
            formation_eligibility_column
        )
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_reference(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"source file does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_month(value: str | pd.Period) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(value, freq="M")
