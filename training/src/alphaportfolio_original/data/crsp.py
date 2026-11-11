"""CRSP CIZ daily-to-monthly feature construction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CIZ_COMMON_STOCK_FILTERS: dict[str, tuple[str, ...]] = {
    "sharetype": ("NS",),
    "securitytype": ("EQTY",),
    "securitysubtype": ("COM",),
    "usincflg": ("Y",),
    "issuertype": ("ACOR", "CORP"),
    "conditionaltype": ("RW",),
    "tradingstatusflg": ("A",),
    "primaryexch": ("N", "A", "Q"),
}


def filter_ciz_common_stocks(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply the formal universe filters stored with the downloaded CIZ data."""

    result = daily.copy()
    for column, allowed in CIZ_COMMON_STOCK_FILTERS.items():
        if column not in result.columns:
            raise ValueError(f"CRSP daily input is missing universe field {column}")
        result = result[result[column].isin(allowed)]
    return result


def construct_crsp_monthly(
    daily: pd.DataFrame,
    *,
    already_filtered: bool = False,
    finalize_cross_month_features: bool = True,
) -> pd.DataFrame:
    """Aggregate CIZ daily observations into paper monthly price features.

    CIZ total returns already integrate delisting outcomes.  ``next_return`` is
    the following calendar month's total return for the same PERMNO.
    """

    required = {
        "permno",
        "dlycaldt",
        "dlyret",
        "dlyretx",
        "dlyvol",
        "shrout",
        "dlycap",
        "dlybid",
        "dlyask",
        "dlyprc",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"CRSP daily input is missing columns: {missing}")
    data = daily.copy() if already_filtered else filter_ciz_common_stocks(daily)
    data["dlycaldt"] = pd.to_datetime(data["dlycaldt"], errors="raise")
    data["month"] = data["dlycaldt"].dt.to_period("M")
    numeric = [
        "dlyret",
        "dlyretx",
        "dlyvol",
        "shrout",
        "dlycap",
        "dlybid",
        "dlyask",
        "dlyprc",
    ]
    if "dlyclose" in data:
        numeric.append("dlyclose")
    for column in numeric:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).astype("float64")
    data = data.sort_values(["permno", "dlycaldt"])
    price = data.get("dlyclose", pd.Series(np.nan, index=data.index))
    data["_price"] = price.fillna(data["dlyprc"]).abs()
    midpoint = (data["dlyask"] + data["dlybid"]) / 2.0
    valid_spread = (
        (data["dlyask"] > 0)
        & (data["dlybid"] > 0)
        & (midpoint > 0)
    ).fillna(False)
    data["_spread"] = (
        (data["dlyask"] - data["dlybid"]) / midpoint
    ).where(valid_spread)
    shares = data["shrout"] * 1_000.0
    data["_daily_turnover"] = _safe_divide(data["dlyvol"], shares)

    identifiers = [
        column
        for column in (
            "permco",
            "ticker",
            "cusip",
            "hdrcusip",
            "siccd",
            "primaryexch",
        )
        if column in data.columns
    ]
    data["_gross_return"] = 1.0 + data["dlyret"]
    data["_gross_return_ex_distribution"] = 1.0 + data["dlyretx"]
    grouped = data.groupby(
        ["permno", "month"],
        sort=True,
        observed=True,
    )
    aggregation: dict[str, tuple[str, str]] = {
        "ret_max": ("dlyret", "max"),
        "me": ("dlycap", "last"),
        "price": ("_price", "last"),
        "shrout": ("shrout", "last"),
        "turnover": ("_daily_turnover", "sum"),
        "spread": ("_spread", "mean"),
        "std_turnover": ("_daily_turnover", "std"),
        "std_vol": ("dlyvol", "std"),
        "total_vol": ("dlyret", "std"),
        "trading_days": ("dlyret", "count"),
    }
    aggregation.update(
        {column: (column, "last") for column in identifiers}
    )
    monthly = grouped.agg(**aggregation).reset_index()
    compound = grouped["_gross_return"].prod(min_count=1) - 1.0
    compound_ex_distribution = (
        grouped["_gross_return_ex_distribution"].prod(min_count=1) - 1.0
    )
    monthly = monthly.merge(
        compound.rename("ret").reset_index(),
        on=["permno", "month"],
        how="left",
        validate="one_to_one",
    ).merge(
        compound_ex_distribution.rename("retx").reset_index(),
        on=["permno", "month"],
        how="left",
        validate="one_to_one",
    )
    monthly["permno"] = monthly["permno"].astype(np.int64)
    ordered = [
        "permno",
        "month",
        "ret",
        "retx",
        "ret_max",
        "me",
        "price",
        "shrout",
        "turnover",
        "spread",
        "std_turnover",
        "std_vol",
        "total_vol",
        "trading_days",
        *identifiers,
    ]
    monthly = monthly[ordered].sort_values(["permno", "month"])
    if not finalize_cross_month_features:
        return monthly.reset_index(drop=True)
    return finalize_crsp_monthly(monthly)


def finalize_crsp_monthly(monthly_statistics: pd.DataFrame) -> pd.DataFrame:
    """Add lagged/forward fields after month-level statistics are available."""

    required = {"permno", "month", "ret", "retx", "price", "shrout"}
    missing = sorted(required - set(monthly_statistics))
    if missing:
        raise ValueError(f"monthly statistics missing columns: {missing}")
    monthly = monthly_statistics.copy()
    if not isinstance(monthly["month"].dtype, pd.PeriodDtype):
        monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M")
    monthly = monthly.sort_values(["permno", "month"]).drop_duplicates(
        ["permno", "month"],
        keep="last",
    )
    previous = monthly[["permno", "month", "shrout"]].copy()
    previous["month"] = previous["month"] + 1
    previous = previous.rename(columns={"shrout": "_previous_shrout"})
    monthly = monthly.merge(previous, on=["permno", "month"], how="left")
    monthly["delta_shrout"] = _safe_divide(
        monthly["shrout"] - monthly["_previous_shrout"],
        monthly["_previous_shrout"],
    )
    monthly["_dividend_return"] = monthly["ret"] - monthly["retx"]
    rolling_dividends = (
        monthly.set_index("month")
        .groupby("permno", observed=True)["_dividend_return"]
        .rolling(12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    monthly["ldp"] = _safe_divide(
        rolling_dividends.to_numpy(),
        monthly["price"].to_numpy(),
    )
    future = monthly[["permno", "month", "ret"]].copy()
    future["month"] = future["month"] - 1
    future = future.rename(columns={"ret": "next_return"})
    monthly = monthly.merge(future, on=["permno", "month"], how="left")
    return monthly.drop(
        columns=["_dividend_return", "_previous_shrout"]
    ).reset_index(drop=True)


def compute_value_weighted_market_return(daily: pd.DataFrame) -> pd.DataFrame:
    """Construct a CRSP-like daily value-weighted market proxy.

    Official CRSP index data remain preferable.  This fallback uses prior-day
    market capitalization and is explicitly labeled as reconstructed.
    """

    required = {"dlycaldt", "dlyret", "dlyprevcap"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"market proxy input missing columns: {missing}")
    data = daily.loc[:, list(required)].copy()
    data["dlycaldt"] = pd.to_datetime(data["dlycaldt"])
    data["dlyret"] = pd.to_numeric(data["dlyret"], errors="coerce")
    data["dlyprevcap"] = pd.to_numeric(data["dlyprevcap"], errors="coerce")
    data = data[
        data["dlyret"].notna()
        & data["dlyprevcap"].notna()
        & (data["dlyprevcap"] > 0)
    ]
    data["_weighted"] = data["dlyret"] * data["dlyprevcap"]
    aggregate = data.groupby("dlycaldt", observed=True).agg(
        weighted_return=("_weighted", "sum"),
        total_weight=("dlyprevcap", "sum"),
        n_assets=("dlyret", "size"),
    )
    aggregate["market_return"] = (
        aggregate["weighted_return"] / aggregate["total_weight"]
    )
    return aggregate.reset_index()[
        ["dlycaldt", "market_return", "n_assets"]
    ]


def read_parquet_partitions(paths: Iterable[str]) -> pd.DataFrame:
    """Read explicitly supplied partitions; no implicit recursive data crawl."""

    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise ValueError("no parquet partitions supplied")
    return pd.concat(frames, ignore_index=True)


def _compound(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.prod(1.0 + clean.to_numpy(dtype=np.float64)) - 1.0)


def _safe_divide(numerator: object, denominator: object) -> object:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=np.isfinite(denominator_array) & (np.abs(denominator_array) > 1.0e-12),
    )
    if result.ndim == 0:
        return float(result)
    return result


def _last_finite(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else np.nan


def _last_nonmissing(values: pd.Series) -> object:
    clean = values.dropna()
    return clean.iloc[-1] if not clean.empty else pd.NA
