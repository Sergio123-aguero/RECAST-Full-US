"""Point-in-time Compustat feature numerators and CCM linking."""

from __future__ import annotations

import numpy as np
import pandas as pd


STRICT_SUPPLEMENT_FIELDS: tuple[str, ...] = (
    "ajex",
    "txp",
    "xsga",
    "mib",
    "txdb",
    "wcapch",
)


def merge_annual_supplement(
    annual: pd.DataFrame,
    supplement: pd.DataFrame,
    *,
    required_fields: tuple[str, ...] = STRICT_SUPPLEMENT_FIELDS,
    optional_fields: tuple[str, ...] = ("scf",),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach a keyed Compustat supplement without changing base row identity."""

    keys = ["gvkey", "datadate"]
    missing_base = sorted(set(keys) - set(annual))
    missing_supplement = sorted(
        set(keys).union(required_fields) - set(supplement)
    )
    if missing_base:
        raise ValueError(f"annual input missing keys: {missing_base}")
    if missing_supplement:
        raise ValueError(
            f"annual supplement missing columns: {missing_supplement}"
        )
    selected_optional = [
        field for field in optional_fields if field in supplement.columns
    ]
    merge_fields = [*required_fields, *selected_optional]
    overlapping = sorted(set(merge_fields).intersection(annual.columns))
    if overlapping:
        raise ValueError(
            "base annual data already contains supplement fields: "
            + ", ".join(overlapping)
        )

    base = annual.copy()
    extra = supplement.loc[:, [*keys, *merge_fields]].copy()
    for frame in (base, extra):
        frame["gvkey"] = frame["gvkey"].astype(str).str.zfill(6)
        frame["datadate"] = pd.to_datetime(frame["datadate"], errors="raise")
    if base.duplicated(keys).any():
        raise ValueError("base annual data has duplicate gvkey-datadate keys")
    if extra.duplicated(keys).any():
        raise ValueError("annual supplement has duplicate gvkey-datadate keys")

    original_rows = len(base)
    merged = base.merge(
        extra,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_supplement_merge",
    )
    if len(merged) != original_rows:
        raise AssertionError("supplement merge changed the annual row count")
    matched = int((merged["_supplement_merge"] == "both").sum())
    merged = merged.drop(columns="_supplement_merge")
    audit = {
        "base_rows": original_rows,
        "supplement_rows": len(extra),
        "matched_rows": matched,
        "unmatched_base_rows": original_rows - matched,
        "key_match_rate": matched / original_rows if original_rows else 0.0,
        "field_nonmissing_rate": {
            field: float(merged[field].notna().mean())
            for field in merge_fields
        },
    }
    return merged, audit


def construct_annual_fundamental_features(
    annual: pd.DataFrame,
    *,
    fallback_publication_lag_months: int = 6,
    specification: str = "strict_near_replication",
) -> pd.DataFrame:
    """Construct Appendix-B accounting features before merging market equity.

    ``available_month`` is the first month in which the observation may enter a
    policy state.  When a public date is unavailable, the conservative fallback
    is fiscal period end plus six months, then the paper's extra one-month lag.
    """

    allowed_specifications = {
        "strict_near_replication",
        "modern_cashflow_harmonized",
        "sensitivity_reconstructed",
    }
    if specification not in allowed_specifications:
        raise ValueError(
            f"unsupported fundamental specification: {specification}"
        )
    required = {"gvkey", "datadate", "at", "sale", "ceq", "ib"}
    if specification in {
        "strict_near_replication",
        "modern_cashflow_harmonized",
    }:
        required.update({"fyear", "ni", "dp", "capx"})
        required.update(STRICT_SUPPLEMENT_FIELDS)
    if specification == "modern_cashflow_harmonized":
        required.add("oancf")
    missing = sorted(required - set(annual.columns))
    if missing:
        raise ValueError(f"Compustat annual input missing columns: {missing}")
    data = annual.copy()
    data["datadate"] = pd.to_datetime(data["datadate"], errors="raise")
    data = _filter_standard_compustat(data)
    data = data.sort_values(["gvkey", "datadate"]).drop_duplicates(
        ["gvkey", "datadate"],
        keep="last",
    )
    for column in data.columns:
        if column not in {
            "gvkey",
            "indfmt",
            "consol",
            "popsrc",
            "datafmt",
            "curcd",
            "fic",
            "costat",
        } and column not in {
            "datadate",
            "pdate",
            "fdate",
            "apdedate",
        }:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "fyear" not in data:
        data["fyear"] = data["datadate"].dt.year
        data["fyear_source"] = "datadate_year_fallback"
    else:
        data["fyear_source"] = "compustat_fyear"
    data["fyear"] = pd.to_numeric(data["fyear"], errors="coerce")

    group = data.groupby("gvkey", sort=False, observed=True)
    previous_fyear = group["fyear"].shift(1)
    adjacent_fiscal_year = (data["fyear"] - previous_fyear).eq(1.0)

    def lag(column: str) -> pd.Series:
        if column not in data:
            return _nan(data)
        return group[column].shift(1).where(adjacent_fiscal_year)

    value = lambda column: data[column] if column in data else _nan(data)
    zero = lambda column: value(column).fillna(0.0)

    preferred = value("pstkrv").combine_first(value("pstkl")).combine_first(
        value("pstk")
    ).fillna(0.0)
    seq = value("seq").combine_first(value("ceq") + zero("pstk"))
    book_equity = seq + zero("txditc") - preferred
    data["_be"] = book_equity
    data["_lag_be"] = group["_be"].shift(1).where(adjacent_fiscal_year)
    data["_lag_at"] = lag("at")
    data["_lag_sale"] = lag("sale")
    data["_lag_cogs"] = lag("cogs")
    data["_lag_ceq"] = lag("ceq")
    data["_lag_invt"] = lag("invt")
    data["_lag_ppent"] = lag("ppent")
    data["_lag_pstkrv"] = lag("pstkrv")

    working_capital = (
        value("act")
        + value("che")
        - value("lct")
        - value("dlc")
        - value("txp")
    )
    data["_working_capital"] = working_capital
    data["_lag_working_capital"] = group["_working_capital"].shift(1).where(
        adjacent_fiscal_year
    )
    noa_dollars = (
        (value("at") - zero("che") - zero("ivao"))
        - (
            value("at")
            - zero("dlc")
            - zero("dltt")
            - zero("mib")
            - zero("pstk")
            - zero("ceq")
        )
    )
    data["_noa_dollars"] = noa_dollars
    data["_lag_noa_dollars"] = group["_noa_dollars"].shift(1).where(
        adjacent_fiscal_year
    )

    data["oa"] = _divide(
        working_capital - data["_lag_working_capital"] - value("dp"),
        data["_lag_at"],
    )
    data["aoa"] = data["oa"].abs()
    data["at_feature"] = value("at")
    data["c"] = _divide(value("che"), value("at"))
    data["c2d"] = _divide(value("ib") + value("dp"), value("lt"))
    data["cto"] = _divide(value("sale"), data["_lag_at"])
    data["delta_ceq"] = _divide(
        value("ceq") - data["_lag_ceq"],
        data["_lag_ceq"],
    )
    data["delta_gm_delta_sales"] = _divide(
        value("sale") - value("cogs"),
        data["_lag_sale"] - data["_lag_cogs"],
    ) - _divide(value("sale"), data["_lag_sale"])
    adjusted_shares = value("csho") * value("ajex")
    lag_adjusted_shares = adjusted_shares.groupby(
        data["gvkey"],
        sort=False,
        observed=True,
    ).shift(1).where(adjacent_fiscal_year)
    data["delta_so"] = np.log(adjusted_shares.where(adjusted_shares > 0)) - np.log(
        lag_adjusted_shares.where(lag_adjusted_shares > 0)
    )
    data["delta_pi2a"] = _divide(
        (value("ppent") + value("invt"))
        - (data["_lag_ppent"] + data["_lag_invt"]),
        data["_lag_at"],
    )
    # Appendix B defines EPS with CRSP monthly SHROUT, not Compustat CSHO.
    # The exact feature is therefore constructed only after the PIT annual
    # record is joined to the formation-month CRSP row in assemble.py.
    if specification == "sensitivity_reconstructed":
        data["eps_compustat_proxy"] = _divide(value("ib"), value("csho"))
    paper_direct_free_cf = _divide(
        value("ni") + value("dp") + value("wcapch") + value("capx"),
        book_equity,
    )
    if specification == "strict_near_replication":
        data["free_cf"] = paper_direct_free_cf
        data["free_cf_source_mode"] = np.where(
            paper_direct_free_cf.notna(),
            "paper_direct_wcapch",
            "missing",
        )
    elif specification == "modern_cashflow_harmonized":
        # WCAPCH is largely discontinued in modern Compustat. OANCF replaces
        # NI + DP + WCAPCH as an explicitly disclosed cash-flow statement
        # total. Retaining the paper's printed +CAPX sign makes this a
        # transparent, sign-preserving reconstruction, not a direct formula.
        harmonized_free_cf = _divide(
            value("oancf") + value("capx"),
            book_equity,
        )
        data["free_cf"] = paper_direct_free_cf.combine_first(
            harmonized_free_cf
        )
        data["free_cf_source_mode"] = np.select(
            [
                paper_direct_free_cf.notna(),
                paper_direct_free_cf.isna() & harmonized_free_cf.notna(),
            ],
            [
                "paper_direct_wcapch",
                "reconstructed_oancf_plus_capx",
            ],
            default="missing",
        )
    else:
        sensitivity_working_capital_change = value("wcapch").combine_first(
            value("wcapc")
        )
        data["free_cf"] = _divide(
            value("ni")
            + value("dp")
            + sensitivity_working_capital_change
            + value("capx"),
            book_equity,
        )
        data["free_cf_source_mode"] = np.select(
            [
                value("wcapch").notna(),
                value("wcapch").isna() & value("wcapc").notna(),
            ],
            ["paper_direct_wcapch", "sensitivity_wcapc_substitution"],
            default="missing",
        )
    data["investment"] = _divide(
        value("at") - data["_lag_at"],
        data["_lag_at"],
    )
    data["ivc"] = _divide(
        2.0 * (value("invt") - data["_lag_invt"]),
        value("at") + data["_lag_at"],
    )
    data["lev"] = _divide(value("dltt") + value("dlc"), value("seq"))
    data["noa"] = _divide(noa_dollars, data["_lag_at"])
    data["ol"] = _divide(value("cogs") + value("xsga"), value("at"))
    data["pcm"] = _divide(value("sale") - value("cogs"), value("sale"))
    data["prof"] = _divide(value("gp"), book_equity)
    data["s2c"] = _divide(value("sale"), value("che"))
    data["sale_g"] = _divide(value("sale"), data["_lag_sale"]) - 1.0
    data["sat"] = _divide(value("sale"), value("at"))
    data["sga2s"] = _divide(value("xsga"), value("sale"))
    data["tan"] = _divide(
        0.715 * value("rect")
        + 0.547 * value("invt")
        + 0.535 * value("ppent")
        + value("che"),
        value("at"),
    )

    # Reconstructions for variables the paper obtains from WRDS Financial Ratios.
    data["ipm_reconstructed"] = _divide(value("ib") + value("txt"), value("sale"))
    data["pm_reconstructed"] = _divide(value("oiadp"), value("sale"))
    data["rna_reconstructed"] = _divide(value("oiadp"), data["_lag_noa_dollars"])
    data["roa_reconstructed"] = _divide(value("ib"), data["_lag_at"])
    data["roe_reconstructed"] = _divide(value("ib"), data["_lag_be"])
    data["roic_reconstructed"] = _divide(
        value("oiadp"),
        value("icapt").combine_first(
            value("ceq") + zero("dltt") + zero("dlc") - zero("che")
        ),
    )

    public_date, public_date_source = _public_information_date(data)
    fallback = data["datadate"] + pd.offsets.MonthEnd(
        fallback_publication_lag_months
    )
    data["information_date"] = public_date.fillna(fallback)
    data["information_date_source"] = public_date_source.fillna(
        f"datadate_plus_{fallback_publication_lag_months}m"
    )
    data["available_month"] = (
        data["information_date"].dt.to_period("M") + 1
    )
    return data


def apply_primary_ccm_link(
    fundamentals: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    """Map GVKEY records to PERMNO using valid primary LC/LU intervals."""

    required_f = {"gvkey", "datadate"}
    required_l = {"gvkey", "lpermno", "linkdt", "linkenddt"}
    if required_f - set(fundamentals):
        raise ValueError("fundamentals missing gvkey/datadate")
    if required_l - set(links):
        raise ValueError("CCM links missing required interval columns")
    link = links.copy()
    if "linktype" in link:
        link = link[link["linktype"].isin(["LC", "LU"])]
    if "linkprim" in link:
        link = link[link["linkprim"].isin(["P", "C"])]
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"]).fillna(
        pd.Timestamp("2262-04-11")
    )
    merged = fundamentals.merge(link, on="gvkey", how="left", suffixes=("", "_link"))
    valid = (
        merged["lpermno"].notna()
        & (merged["datadate"] >= merged["linkdt"])
        & (merged["datadate"] <= merged["linkenddt"])
    )
    merged = merged[valid].copy()
    if "linktype_rank" in merged:
        sort_columns = ["gvkey", "datadate", "linktype_rank"]
        if "linkprim_rank" in merged:
            sort_columns.append("linkprim_rank")
        merged = merged.sort_values(sort_columns)
    merged = merged.drop_duplicates(["gvkey", "datadate"], keep="first")
    merged["permno"] = merged["lpermno"].astype(np.int64)
    return merged


def _filter_standard_compustat(data: pd.DataFrame) -> pd.DataFrame:
    filters = {
        "indfmt": "INDL",
        "consol": "C",
        "popsrc": "D",
        "datafmt": "STD",
    }
    result = data
    for column, value in filters.items():
        if column in result:
            result = result[result[column] == value]
    return result


def _public_information_date(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return a public date without treating period-end fields as disclosure."""

    result = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    source = pd.Series(pd.NA, index=data.index, dtype="string")
    for column in ("pdate", "fdate"):
        if column not in data:
            continue
        candidate = pd.to_datetime(data[column], errors="coerce")
        candidate = candidate.where(candidate >= data["datadate"])
        selected = result.isna() & candidate.notna()
        result.loc[selected] = candidate.loc[selected]
        source.loc[selected] = column
    return result, source


def _nan(data: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=data.index, dtype=np.float64)


def _divide(numerator: object, denominator: object) -> pd.Series:
    numerator_series = pd.Series(numerator, dtype=np.float64)
    denominator_series = pd.Series(denominator, dtype=np.float64)
    denominator_series = denominator_series.where(
        denominator_series.abs() > 1.0e-12
    )
    return numerator_series / denominator_series
