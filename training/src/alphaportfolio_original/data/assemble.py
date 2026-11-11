"""Assemble CRSP, CCM/Compustat, and WRDS ratio inputs into 51 features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import FEATURE_NAMES


WRDS_RATIO_FEATURES: tuple[str, ...] = (
    "beme",
    "ipm",
    "pm",
    "rna",
    "roa",
    "roe",
    "roic",
    "s2p",
)

MTHDELFLG_CATEGORY_BY_FLAG: dict[str, str] = {
    "N": "active",
    "A": "full_included",
    "P": "full_included",
    "V": "partial_included",
    "G": "excluded_or_censored",
    "M": "excluded_or_censored",
}
MTHRETFLG_CATEGORY_BY_FLAG: dict[str, str] = {
    "CR": "observed_complete_like",
    "DE": "observed_complete_like",
    "NS": "partial_or_special",
    "IP": "partial_or_special",
    "MP": "partial_or_special",
    "GP": "missing_period",
    "NT": "missing_period",
}
OUTCOME_ROW_MISSING_CATEGORY = "outcome_row_missing"

OUTCOME_QUALITY_BOOLEAN_COLUMNS: tuple[str, ...] = (
    "next_return_is_exit_or_delisting",
    "next_return_delisting_full_included",
    "next_return_delisting_partial_included",
    "next_return_delisting_excluded_or_censored",
    "next_return_is_active",
    "next_return_is_observed_complete_like",
    "next_return_is_partial_or_special",
    "next_return_is_missing_period",
)

OUTCOME_QUALITY_COLUMNS: tuple[str, ...] = (
    "next_return_mthdelflg",
    "next_return_mthdelflg_category",
    "next_return_mthretflg",
    "next_return_mthretflg_category",
    *OUTCOME_QUALITY_BOOLEAN_COLUMNS,
    "formation_month_end_eligible",
)

PANEL_AUDIT_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "primaryexch",
    *OUTCOME_QUALITY_COLUMNS,
    "fundamental_datadate",
    "fundamental_information_date",
    "fundamental_information_date_source",
    "fundamental_available_month",
    "fundamental_fyear",
    "fundamental_fyear_source",
    "free_cf_source_mode",
)


@dataclass(frozen=True)
class FeatureCoverage:
    feature: str
    nonmissing: int
    total: int
    coverage: float


def assemble_feature_panel(
    crsp_monthly: pd.DataFrame,
    linked_fundamentals: pd.DataFrame,
    *,
    ratio_monthly: pd.DataFrame | None = None,
    daily_risk_monthly: pd.DataFrame | None = None,
    allow_ratio_reconstruction: bool = False,
    strict: bool = True,
    minimum_feature_coverage: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the unstandardized 51-variable monthly panel.

    Formal replication defaults to ``strict=True`` and refuses an all-missing
    or nearly absent input instead of silently converting it to zero.
    """

    required_crsp = {
        "permno",
        "month",
        "ret",
        "ret_max",
        "me",
        "shrout",
        "turnover",
        "ldp",
        "delta_shrout",
        "spread",
        "std_turnover",
        "std_vol",
        "total_vol",
        "next_return",
    }
    missing = sorted(required_crsp - set(crsp_monthly))
    if missing:
        raise ValueError(f"CRSP monthly frame missing columns: {missing}")
    validate_outcome_quality_contract(
        crsp_monthly,
        required=strict,
        context="CRSP monthly input",
    )
    required_fundamental = {"permno", "available_month", "datadate"}
    if required_fundamental - set(linked_fundamentals):
        raise ValueError("linked fundamentals missing point-in-time keys")

    crsp = crsp_monthly.copy()
    crsp["month"] = _month(crsp["month"])
    fundamentals = linked_fundamentals.copy()
    fundamentals["available_month"] = _month(fundamentals["available_month"])
    merged = _grouped_asof(
        crsp.sort_values(["permno", "month"]),
        fundamentals.sort_values(["permno", "available_month"]),
        left_on="month",
        right_on="available_month",
        by="permno",
    )

    if ratio_monthly is not None:
        ratios = ratio_monthly.copy()
        ratios["month"] = _month(ratios["month"])
        missing_ratios = set(WRDS_RATIO_FEATURES) - set(ratios)
        if missing_ratios:
            raise ValueError(
                f"WRDS ratio frame missing: {sorted(missing_ratios)}"
            )
        merged = merged.merge(
            ratios[["permno", "month", *WRDS_RATIO_FEATURES]],
            on=["permno", "month"],
            how="left",
            suffixes=("", "_ratio"),
        )
    if daily_risk_monthly is not None:
        risk = daily_risk_monthly.copy()
        risk["month"] = _month(risk["month"])
        if {"permno", "month", "beta_daily", "idio_vol"} - set(risk):
            raise ValueError("daily risk frame must contain beta_daily and idio_vol")
        merged = merged.merge(
            risk[["permno", "month", "beta_daily", "idio_vol"]],
            on=["permno", "month"],
            how="left",
            suffixes=("", "_risk"),
        )

    me = merged["me"]
    at = merged.get("at", merged.get("at_feature"))
    be = merged.get("_be")
    merged["a2me"] = _divide(at * 1_000.0, me)
    merged["at"] = merged.get("at_feature", at)
    merged["debt2p"] = _divide(
        (merged.get("dltt") + merged.get("dlc")) * 1_000.0,
        me,
    )
    merged["e2p"] = _divide(merged.get("ib") * 1_000.0, me)
    # Compustat IB is reported in USD millions and CRSP SHROUT in thousands
    # of shares, so IB * 1000 / SHROUT is dollars per CRSP security share.
    # This follows Appendix B and deliberately overwrites any legacy
    # Compustat-CSHO proxy carried by an older intermediate file.
    merged["eps"] = _divide(
        merged.get("ib") * 1_000.0,
        merged.get("shrout"),
    )
    merged["nop"] = _divide(
        (
            merged.get("dvc")
            + merged.get("prstkc")
            - merged.get("sstk")
        )
        * 1_000.0,
        me,
    )
    merged["o2p"] = _divide(
        (
            merged.get("dvc")
            + merged.get("prstkc")
            - (merged.get("pstkrv") - merged.get("_lag_pstkrv"))
        )
        * 1_000.0,
        me,
    )
    deferred_tax = merged.get("txdb", merged.get("txditc"))
    merged["q"] = _divide(
        at + me / 1_000.0 - merged.get("ceq") - deferred_tax,
        at,
    )
    merged["roc"] = _divide(
        merged.get("dltt") + me / 1_000.0 - at,
        merged.get("che"),
    )

    if allow_ratio_reconstruction:
        reconstructed = {
            "beme": _divide(be * 1_000.0, me),
            "ipm": merged.get("ipm_reconstructed"),
            "pm": merged.get("pm_reconstructed"),
            "rna": merged.get("rna_reconstructed"),
            "roa": merged.get("roa_reconstructed"),
            "roe": merged.get("roe_reconstructed"),
            "roic": merged.get("roic_reconstructed"),
            "s2p": _divide(merged.get("sale") * 1_000.0, me),
        }
        for name, values in reconstructed.items():
            if name not in merged:
                merged[name] = values
            else:
                merged[name] = merged[name].combine_first(values)

    for feature in FEATURE_NAMES:
        if feature not in merged:
            merged[feature] = np.nan
    audit_source_columns = {
        "datadate": "fundamental_datadate",
        "information_date": "fundamental_information_date",
        "information_date_source": "fundamental_information_date_source",
        "available_month": "fundamental_available_month",
        "fyear": "fundamental_fyear",
        "fyear_source": "fundamental_fyear_source",
    }
    for source, target in audit_source_columns.items():
        merged[target] = merged[source] if source in merged else pd.NA
    if "free_cf_source_mode" not in merged:
        merged["free_cf_source_mode"] = pd.NA
    for column in ("gvkey", "primaryexch"):
        if column not in merged:
            merged[column] = pd.NA
    for column in OUTCOME_QUALITY_COLUMNS:
        if column not in merged:
            # Non-strict library callers may still assemble exploratory legacy
            # fixtures.  Formal callers are rejected by the contract above.
            merged[column] = pd.NA
    panel = merged[
        [
            "month",
            "permno",
            *PANEL_AUDIT_COLUMNS,
            *FEATURE_NAMES,
            "next_return",
        ]
    ].copy()
    coverage = feature_coverage(panel)
    if strict:
        failed = coverage[
            coverage["coverage"] < float(minimum_feature_coverage)
        ]
        if not failed.empty:
            labels = ", ".join(
                f"{row.feature}={row.coverage:.6%}"
                for row in failed.itertuples(index=False)
            )
            raise ValueError(
                "formal feature panel failed minimum coverage for: "
                f"{labels}. Supply WRDS ratio/risk data or an explicitly "
                "documented reconstruction."
            )
    return panel, coverage


def validate_outcome_quality_contract(
    frame: pd.DataFrame,
    *,
    required: bool,
    context: str = "panel",
) -> None:
    """Validate the CIZ outcome-quality metadata as one atomic contract.

    A legacy exploratory frame may omit the *entire* contract when
    ``required=False``.  Supplying only a subset is never accepted because it
    would make censoring and missing-period rows indistinguishable from clean
    realized outcomes.
    """

    present = set(OUTCOME_QUALITY_COLUMNS).intersection(frame.columns)
    if not present:
        if required:
            raise ValueError(
                f"{context} lacks required outcome-quality columns: "
                f"{list(OUTCOME_QUALITY_COLUMNS)}"
            )
        return
    missing = sorted(set(OUTCOME_QUALITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{context} has a partial outcome-quality contract; missing: "
            f"{missing}"
        )

    for column in (*OUTCOME_QUALITY_BOOLEAN_COLUMNS, "formation_month_end_eligible"):
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{context} has missing boolean values in {column}")
        if not values.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{context} column {column} must be boolean")

    delflag = frame["next_return_mthdelflg"].astype("string")
    delcategory = frame["next_return_mthdelflg_category"].astype("string")
    retflag = frame["next_return_mthretflg"].astype("string")
    retcategory = frame["next_return_mthretflg_category"].astype("string")
    allowed_del_categories = {
        *MTHDELFLG_CATEGORY_BY_FLAG.values(),
        OUTCOME_ROW_MISSING_CATEGORY,
    }
    allowed_ret_categories = {
        *MTHRETFLG_CATEGORY_BY_FLAG.values(),
        OUTCOME_ROW_MISSING_CATEGORY,
    }
    _require_nonmissing_legal_category(
        delcategory,
        allowed=allowed_del_categories,
        column="next_return_mthdelflg_category",
        context=context,
    )
    _require_nonmissing_legal_category(
        retcategory,
        allowed=allowed_ret_categories,
        column="next_return_mthretflg_category",
        context=context,
    )

    del_row_missing = delcategory.eq(OUTCOME_ROW_MISSING_CATEGORY)
    ret_row_missing = retcategory.eq(OUTCOME_ROW_MISSING_CATEGORY)
    if not np.array_equal(
        del_row_missing.to_numpy(dtype=bool),
        ret_row_missing.to_numpy(dtype=bool),
    ):
        raise ValueError(
            f"{context} outcome-row-missing categories disagree between "
            "mthdelflg and mthretflg"
        )
    if not np.array_equal(
        delflag.isna().to_numpy(dtype=bool),
        del_row_missing.to_numpy(dtype=bool),
    ):
        raise ValueError(
            f"{context} next_return_mthdelflg may be missing only when the "
            "next-month outcome row is missing"
        )
    if not np.array_equal(
        retflag.isna().to_numpy(dtype=bool),
        ret_row_missing.to_numpy(dtype=bool),
    ):
        raise ValueError(
            f"{context} next_return_mthretflg may be missing only when the "
            "next-month outcome row is missing"
        )

    observed = ~del_row_missing
    invalid_delflags = sorted(
        set(delflag.loc[observed].dropna().astype(str))
        - set(MTHDELFLG_CATEGORY_BY_FLAG)
    )
    if invalid_delflags:
        raise ValueError(
            f"{context} has invalid next_return_mthdelflg values: "
            f"{invalid_delflags}"
        )
    invalid_retflags = sorted(
        set(retflag.loc[observed].dropna().astype(str))
        - set(MTHRETFLG_CATEGORY_BY_FLAG)
    )
    if invalid_retflags:
        raise ValueError(
            f"{context} has invalid next_return_mthretflg values: "
            f"{invalid_retflags}"
        )

    expected_delcategory = delflag.map(MTHDELFLG_CATEGORY_BY_FLAG).astype(
        "string"
    )
    expected_delcategory.loc[del_row_missing] = OUTCOME_ROW_MISSING_CATEGORY
    expected_retcategory = retflag.map(MTHRETFLG_CATEGORY_BY_FLAG).astype(
        "string"
    )
    expected_retcategory.loc[ret_row_missing] = OUTCOME_ROW_MISSING_CATEGORY
    _require_equal_series(
        delcategory,
        expected_delcategory,
        context=context,
        description="mthdelflg category",
    )
    _require_equal_series(
        retcategory,
        expected_retcategory,
        context=context,
        description="mthretflg category",
    )

    expected_booleans = {
        "next_return_is_exit_or_delisting": observed & delflag.ne("N"),
        "next_return_delisting_full_included": delflag.isin(("A", "P")),
        "next_return_delisting_partial_included": delflag.eq("V"),
        "next_return_delisting_excluded_or_censored": delflag.isin(("G", "M")),
        "next_return_is_active": delflag.eq("N"),
        "next_return_is_observed_complete_like": retflag.isin(("CR", "DE")),
        "next_return_is_partial_or_special": retflag.isin(("NS", "IP", "MP")),
        "next_return_is_missing_period": retflag.isin(("GP", "NT")),
    }
    for column, expected in expected_booleans.items():
        actual_values = frame[column].to_numpy(dtype=bool)
        expected_values = expected.fillna(False).to_numpy(dtype=bool)
        if not np.array_equal(actual_values, expected_values):
            mismatches = int((actual_values != expected_values).sum())
            raise ValueError(
                f"{context} has {mismatches} rows inconsistent with {column}"
            )


def outcome_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return disclosure-ready counts after validating the formal contract."""

    validate_outcome_quality_contract(
        frame,
        required=True,
        context="outcome-quality summary input",
    )
    eligible = frame["formation_month_end_eligible"].astype(bool)
    eligible_frame = frame.loc[eligible]
    missing_or_censored = (
        eligible_frame["next_return"].isna()
        | eligible_frame["next_return_delisting_excluded_or_censored"].astype(bool)
        | eligible_frame["next_return_is_missing_period"].astype(bool)
    )

    def _counts(values: pd.Series, allowed: tuple[str, ...]) -> dict[str, int]:
        normalized = values.astype("string")
        return {
            value: int(normalized.eq(value).sum())
            for value in allowed
        }

    del_categories = (
        "active",
        "full_included",
        "partial_included",
        "excluded_or_censored",
        OUTCOME_ROW_MISSING_CATEGORY,
    )
    ret_categories = (
        "observed_complete_like",
        "partial_or_special",
        "missing_period",
        OUTCOME_ROW_MISSING_CATEGORY,
    )
    return {
        "rows": int(len(frame)),
        "formation_eligible_rows": int(eligible.sum()),
        "formation_ineligible_rows": int((~eligible).sum()),
        "next_return_missing_rows": int(frame["next_return"].isna().sum()),
        "mthdelflg_counts": _counts(
            frame["next_return_mthdelflg"],
            tuple(MTHDELFLG_CATEGORY_BY_FLAG),
        ),
        "mthdelflg_category_counts": _counts(
            frame["next_return_mthdelflg_category"],
            del_categories,
        ),
        "mthretflg_counts": _counts(
            frame["next_return_mthretflg"],
            tuple(MTHRETFLG_CATEGORY_BY_FLAG),
        ),
        "mthretflg_category_counts": _counts(
            frame["next_return_mthretflg_category"],
            ret_categories,
        ),
        "boolean_true_counts": {
            column: int(frame[column].sum())
            for column in OUTCOME_QUALITY_BOOLEAN_COLUMNS
        },
        "within_formation_eligible": {
            "rows": int(len(eligible_frame)),
            "next_return_missing_rows": int(
                eligible_frame["next_return"].isna().sum()
            ),
            "next_return_delisting_partial_included_rows": int(
                eligible_frame["next_return_delisting_partial_included"].sum()
            ),
            "next_return_delisting_excluded_or_censored_rows": int(
                eligible_frame[
                    "next_return_delisting_excluded_or_censored"
                ].sum()
            ),
            "next_return_partial_or_special_rows": int(
                eligible_frame["next_return_is_partial_or_special"].sum()
            ),
            "next_return_missing_period_rows": int(
                eligible_frame["next_return_is_missing_period"].sum()
            ),
            "next_month_outcome_row_missing_rows": int(
                eligible_frame["next_return_mthretflg_category"]
                .astype("string")
                .eq(OUTCOME_ROW_MISSING_CATEGORY)
                .sum()
            ),
            "missing_or_censored_union_rows": int(missing_or_censored.sum()),
            "mthdelflg_category_counts": _counts(
                eligible_frame["next_return_mthdelflg_category"],
                del_categories,
            ),
            "mthretflg_category_counts": _counts(
                eligible_frame["next_return_mthretflg_category"],
                ret_categories,
            ),
        },
    }


def _require_nonmissing_legal_category(
    values: pd.Series,
    *,
    allowed: set[str],
    column: str,
    context: str,
) -> None:
    invalid = sorted(set(values.dropna().astype(str)) - allowed)
    if values.isna().any() or invalid:
        raise ValueError(
            f"{context} has invalid {column}; "
            f"missing={int(values.isna().sum())}, invalid={invalid}"
        )


def _require_equal_series(
    actual: pd.Series,
    expected: pd.Series,
    *,
    context: str,
    description: str,
) -> None:
    left = actual.astype("string").reset_index(drop=True)
    right = expected.astype("string").reset_index(drop=True)
    if not left.equals(right):
        mismatches = int((left.fillna("<NA>") != right.fillna("<NA>")).sum())
        raise ValueError(
            f"{context} has {mismatches} inconsistent {description} rows"
        )


def feature_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(panel)
    for feature in FEATURE_NAMES:
        nonmissing = int(panel[feature].notna().sum()) if feature in panel else 0
        rows.append(
            {
                "feature": feature,
                "nonmissing": nonmissing,
                "total": total,
                "coverage": nonmissing / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _grouped_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str,
) -> pd.DataFrame:
    lhs = left.copy()
    rhs = right.copy()
    ordinal = "_point_in_time_month_ordinal"
    lhs[ordinal] = lhs[left_on].astype("int64")
    rhs[ordinal] = rhs[right_on].astype("int64")
    result = pd.merge_asof(
        lhs.sort_values([ordinal, by]),
        rhs.sort_values([ordinal, by]),
        on=ordinal,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_fundamental"),
    )
    return result.drop(columns=[ordinal]).sort_values(
        [by, left_on]
    ).reset_index(drop=True)


def _month(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.PeriodDtype):
        return values.astype("period[M]")
    return pd.to_datetime(values).dt.to_period("M")


def _divide(numerator: object, denominator: object) -> pd.Series:
    numerator_series = pd.Series(numerator, dtype=np.float64)
    denominator_series = pd.Series(denominator, dtype=np.float64)
    return numerator_series / denominator_series.where(
        denominator_series.abs() > 1.0e-12
    )
