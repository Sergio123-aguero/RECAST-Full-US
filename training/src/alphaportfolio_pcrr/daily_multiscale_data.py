"""Daily stock features, cross-sectional normalization and monthly panel reads."""
from __future__ import annotations
from datetime import date
import json
from typing import Sequence
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from alphaportfolio_original.data.crsp import CIZ_COMMON_STOCK_FILTERS

DAILY_FEATURE_NAMES = (
    "return_1", "return_5", "return_21", "return_63", "return_126", "return_252",
    "volatility_21", "volatility_63", "volatility_252", "downside_21", "downside_63",
    "drawdown_21", "drawdown_63", "drawdown_252", "log_market_cap", "log_price",
    "log_dollar_volume", "log_mean_volume_21", "turnover_1", "turnover_21", "turnover_63",
    "relative_volume_21", "relative_volume_63", "spread_1", "spread_21",
    "amihud_21", "amihud_63", "zero_volume_fraction_21", "volatility_ratio_21_63",
)

NUMERIC_COLUMNS = ("dlyret", "dlyvol", "shrout", "dlycap", "dlybid", "dlyask", "dlyprc")


def read_months(path, columns, first, last, *, period_storage):
    """Local feature-only adapter for both valid Arrow monthly encodings.

    Registering pandas' Arrow extension changes the in-memory schema returned
    for the SAME Parquet bytes. It does not change the stored ordinal values.
    Do not relax or modify the sealed historical model's source reader.
    """
    if any(str(name).startswith("next_return") for name in columns):
        raise ValueError("multiscale monthly reader accepts feature columns only")
    source = pq.ParquetFile(path)
    field = source.schema_arrow.field("month")
    if period_storage:
        if isinstance(field.type, pa.ExtensionType):
            valid = (field.type.extension_name == "pandas.period"
                and pa.types.is_int64(field.type.storage_type)
                and json.loads(field.type.__arrow_ext_serialize__()) == {"freq": "M"})
        else:
            metadata = field.metadata or {}
            valid = (pa.types.is_int64(field.type)
                and metadata.get(b"ARROW:extension:name") == b"pandas.period"
                and json.loads(metadata.get(b"ARROW:extension:metadata", b"{}")) == {"freq": "M"})
        if not valid:
            raise ValueError("unexpected monthly period encoding; exact int64 pandas.period freq M required")
        lower, upper = pd.Period(first, freq="M").ordinal, pd.Period(last, freq="M").ordinal
    else:
        if not pa.types.is_string(field.type):
            raise ValueError("unexpected monthly membership string encoding")
        lower, upper = str(first), str(last)
    physical_index = source.schema_arrow.get_field_index("month")
    groups = []
    for i in range(source.num_row_groups):
        stats = source.metadata.row_group(i).column(physical_index).statistics
        if stats is None or not stats.has_min_max or not (stats.max < lower or stats.min > upper):
            groups.append(i)
    pieces = []
    for batch in source.iter_batches(batch_size=16384, row_groups=groups, columns=columns, use_threads=False):
        index = batch.schema.get_field_index("month")
        months = batch.column(index)
        if months.null_count:
            raise ValueError("missing monthly source date")
        ordinals = months.storage if isinstance(months.type, pa.ExtensionType) else months
        selected = batch.filter(pc.and_(pc.greater_equal(ordinals, lower), pc.less_equal(ordinals, upper)))
        if not selected.num_rows:
            continue
        selected_months = selected.column(index)
        if isinstance(selected_months.type, pa.ExtensionType):
            # Avoid pandas converting an extension to Period objects before
            # the explicit, identical ordinal reconstruction below.
            selected = selected.set_column(index, "month", selected_months.storage)
        frame = selected.to_pandas(ignore_metadata=True)
        frame["month"] = (pd.PeriodIndex.from_ordinals(frame.month.to_numpy(dtype=np.int64), freq="M")
            if period_storage else pd.PeriodIndex(frame.month, freq="M"))
        pieces.append(frame)
    if not pieces:
        raise ValueError("empty requested monthly feature slice")
    result = pd.concat(pieces, ignore_index=True)
    if (result.duplicated(["month", "permno"]).any()
            or tuple(sorted(result.month.unique())) != tuple(pd.period_range(first, last, freq="M"))):
        raise ValueError("duplicate asset-months or incomplete monthly feature calendar")
    return result


def available_memory_bytes() -> int:
    """Read available physical RAM without adding a dependency."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        import ctypes
        import os
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in
                      ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise RuntimeError("cannot establish available RAM")
            return int(status.available)
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def estimate_memory_bytes(sessions: int, assets: int, output_sessions: int, *, source_rows: int = 0) -> int:
    """Conservative peak estimate including float64 rolling work and a sample."""
    return int(source_rows * 100 + sessions * assets * 72
        + output_sessions * assets * len(DAILY_FEATURE_NAMES) * 4
        + assets * (252 * len(DAILY_FEATURE_NAMES) * 5 + 24 * 51 * 5) + 128 * 2**20)


def require_memory(estimated_bytes: int, *, reserve_bytes: int = 384 * 2**20) -> dict:
    available = available_memory_bytes()
    if estimated_bytes + reserve_bytes > available:
        raise MemoryError(f"full-universe build estimates {estimated_bytes / 2**30:.2f} GiB plus "
            f"{reserve_bytes / 2**30:.2f} GiB reserve; only {available / 2**30:.2f} GiB available; no axis reduction")
    return {"estimated_peak_bytes": estimated_bytes, "available_bytes": available, "reserve_bytes": reserve_bytes}


def _entry_eligible(rows: pd.DataFrame) -> np.ndarray:
    valid = np.isfinite(pd.to_numeric(rows.dlyprc, errors="coerce")) & (rows.dlyprc > 0)
    for name, allowed in CIZ_COMMON_STOCK_FILTERS.items():
        valid &= rows[name].isin(allowed)
    return np.asarray(valid, dtype=bool)


def _rolling(values, window, operation):
    roll = pd.DataFrame(values).rolling(window, min_periods=window)
    return getattr(roll, operation)().to_numpy(dtype=np.float32)


def _normalize_cross_section(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Each historical date uses only its own full eligible cross-section."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(len(values)):
        known = np.isfinite(values[i])
        population = values[i, known & eligible[i]]
        if not len(population):
            continue
        low, high = np.quantile(population, [.01, .99])
        clipped = np.clip(population.astype(np.float64), low, high)
        center, scale = clipped.mean(), clipped.std(ddof=0)
        out[i, known] = (np.clip(values[i, known], low, high) - center) / scale if scale > 1e-12 else 0.
    return out


def compute_daily_features(rows: pd.DataFrame, sessions: Sequence[date], *, output_start: date,
                            progress=None, enforce_memory=True) -> dict:
    """Vectorized calendar-grid rolling features; missing sessions stay missing."""
    dates = pd.DatetimeIndex(sessions)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("feature calendar must be unique and increasing")
    if rows.duplicated(["dlycaldt", "permno"]).any():
        raise ValueError("duplicate daily row")
    ids = np.sort(rows.permno.unique()).astype(np.int64)
    start = int(dates.get_indexer([pd.Timestamp(output_start)])[0])
    if start < 0:
        raise ValueError("output start is not an actual calendar session")
    estimated = estimate_memory_bytes(len(dates), len(ids), len(dates) - start, source_rows=len(rows))
    memory = require_memory(estimated) if enforce_memory else {"estimated_peak_bytes": estimated}
    time_index = dates.get_indexer(pd.to_datetime(rows.dlycaldt))
    if (time_index < 0).any():
        raise ValueError("daily source contains a non-calendar or out-of-range row")
    asset_index = np.searchsorted(ids, rows.permno.to_numpy())
    shape = (len(dates), len(ids))
    raw = {}
    for name in NUMERIC_COLUMNS:
        value = np.full(shape, np.nan, dtype=np.float32)
        value[time_index, asset_index] = rows[name].to_numpy(dtype=np.float32)
        raw[name] = value
    eligible = np.zeros(shape, dtype=bool)
    eligible[time_index, asset_index] = rows.entry_eligible.to_numpy(bool)
    cube = np.full((len(dates) - start, len(ids), len(DAILY_FEATURE_NAMES)), np.nan, dtype=np.float32)

    def save(name, values):
        cube[:, :, DAILY_FEATURE_NAMES.index(name)] = _normalize_cross_section(values[start:], eligible[start:])
        if progress:
            progress({"stage": "daily_feature", "feature": name, "shape": list(cube.shape)})

    ret, volume, shares, cap, bid, ask, price = (raw[x] for x in NUMERIC_COLUMNS)
    ret[~np.isfinite(ret) | (ret < -1)] = np.nan
    save("return_1", ret)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log1p(np.where(ret > -1, ret, np.nan)).astype(np.float32)
    for width in (5, 21, 63, 126, 252):
        # Exact -100% returns are valid; a known zero wealth factor dominates.
        safe_log = np.where(ret == -1, 0., logret)
        compound = np.expm1(_rolling(safe_log, width, "sum"))
        zero = _rolling(np.where(np.isfinite(ret), ret == -1, np.nan), width, "sum")
        compound[zero > 0] = -1.
        save(f"return_{width}", compound)
    vols = {}
    for width in (21, 63, 252):
        vols[width] = _rolling(ret, width, "std")
        save(f"volatility_{width}", vols[width])
    for width in (21, 63):
        save(f"downside_{width}", np.sqrt(_rolling(np.minimum(ret, 0.) ** 2, width, "mean")))
    cumulative = np.nancumsum(logret, axis=0, dtype=np.float64)
    for width in (21, 63, 252):
        maximum = _rolling(cumulative, width + 1, "max")
        drawdown = np.expm1(cumulative - maximum).astype(np.float32)
        drawdown[~np.isfinite(_rolling(logret, width, "sum"))] = np.nan
        save(f"drawdown_{width}", drawdown)
    del cumulative, drawdown, maximum, compound, zero, safe_log
    with np.errstate(divide="ignore", invalid="ignore"):
        dollar_volume = np.where((volume >= 0) & (price > 0), volume * price, np.nan)
        turnover = np.where((volume >= 0) & (shares > 0), volume / (shares * 1000.), np.nan)
        spread = np.where((bid > 0) & (ask >= bid), (ask - bid) / ((ask + bid) / 2.), np.nan)
        illiquidity = np.where(dollar_volume > 0, np.abs(ret) / dollar_volume, np.nan)
        save("log_market_cap", np.log(np.where(cap > 0, cap, np.nan)))
        save("log_price", np.log(np.where(price > 0, price, np.nan)))
        save("log_dollar_volume", np.log1p(dollar_volume))
        save("log_mean_volume_21", np.log1p(_rolling(np.where(volume >= 0, volume, np.nan), 21, "mean")))
        save("turnover_1", turnover)
        for width in (21, 63):
            save(f"turnover_{width}", _rolling(turnover, width, "mean"))
            average = _rolling(np.where(volume >= 0, volume, np.nan), width, "mean")
            save(f"relative_volume_{width}", np.where(average > 0, volume / average, np.nan))
            save(f"amihud_{width}", _rolling(illiquidity, width, "mean"))
        save("spread_1", spread)
        save("spread_21", _rolling(spread, 21, "mean"))
        save("zero_volume_fraction_21", _rolling(np.where(np.isfinite(volume) & (volume >= 0), volume == 0, np.nan), 21, "mean"))
        save("volatility_ratio_21_63", np.where(vols[63] > 0, vols[21] / vols[63], np.nan))
    return {"dates": tuple(d.date() for d in dates[start:]), "asset_ids": ids,
        "values": cube, "entry_eligible": eligible[start:], "memory": memory}
