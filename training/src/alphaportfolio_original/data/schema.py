"""The 51 paper inputs from Appendix B (PDF pages 82-85)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    paper_label: str
    category: str
    primary_source: str
    description: str


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("a2me", "A2ME", "value", "CRSP+Compustat", "Total assets / market equity."),
    FeatureDefinition("oa", "OA", "intangibles", "Compustat", "Operating accruals / lagged assets."),
    FeatureDefinition("aoa", "AOA", "intangibles", "Compustat", "Absolute operating accruals."),
    FeatureDefinition("at", "AT", "investment", "Compustat", "Total assets."),
    FeatureDefinition("beme", "BEME", "value", "CRSP+Compustat", "Book equity / market equity."),
    FeatureDefinition("beta_daily", "Beta daily", "price", "CRSP daily", "Dimson-style daily market beta."),
    FeatureDefinition("c", "C", "profitability", "Compustat", "Cash and short-term investments / assets."),
    FeatureDefinition("c2d", "C2D", "profitability", "Compustat", "(Income + depreciation) / liabilities."),
    FeatureDefinition("cto", "CTO", "profitability", "Compustat", "Sales / lagged assets."),
    FeatureDefinition("debt2p", "Debt2P", "value", "CRSP+Compustat", "Debt / market equity."),
    FeatureDefinition("delta_ceq", "Delta ceq", "investment", "Compustat", "Percentage change in common equity."),
    FeatureDefinition(
        "delta_gm_delta_sales",
        "Delta(DeltaGm-DeltaSales)",
        "profitability",
        "Compustat",
        "Change in gross margin minus change in sales.",
    ),
    FeatureDefinition("delta_so", "Delta So", "investment", "Compustat", "Log change in split-adjusted shares."),
    FeatureDefinition("delta_shrout", "Delta shrout", "investment", "CRSP", "Percentage change in shares outstanding."),
    FeatureDefinition("delta_pi2a", "Delta PI2A", "investment", "Compustat", "Change in PP&E plus inventory / lagged assets."),
    FeatureDefinition("e2p", "E2P", "value", "CRSP+Compustat", "Income before extraordinary items / market equity."),
    FeatureDefinition("eps", "EPS", "profitability", "Compustat", "Income before extraordinary items / shares."),
    FeatureDefinition("free_cf", "Free CF", "profitability", "Compustat", "Free cash flow / book equity."),
    FeatureDefinition("idio_vol", "Idol vol", "price", "CRSP daily+FF3", "Residual volatility from FF3 regression."),
    FeatureDefinition("investment", "Investment", "investment", "Compustat", "Annual asset growth."),
    FeatureDefinition("ipm", "IPM", "profitability", "Compustat", "Pretax profit margin."),
    FeatureDefinition("ivc", "IVC", "investment", "Compustat", "Inventory change / average assets."),
    FeatureDefinition("lev", "Lev", "value", "Compustat", "Debt / stockholders' equity."),
    FeatureDefinition("ldp", "LDP", "value", "CRSP", "Annual dividends / price."),
    FeatureDefinition("me", "ME", "value", "CRSP", "Market equity."),
    FeatureDefinition("turnover", "Turnover", "friction", "CRSP daily", "Volume / shares outstanding."),
    FeatureDefinition("noa", "NOA", "intangibles", "Compustat", "Net operating assets / lagged assets."),
    FeatureDefinition("nop", "NOP", "value", "CRSP+Compustat", "Net payout / market equity."),
    FeatureDefinition("o2p", "O2P", "value", "CRSP+Compustat", "Payout / market equity."),
    FeatureDefinition("ol", "OL", "profitability", "Compustat", "Operating costs / assets."),
    FeatureDefinition("pcm", "PCM", "profitability", "Compustat", "Price-cost margin."),
    FeatureDefinition("pm", "PM", "profitability", "Compustat", "Operating income / sales."),
    FeatureDefinition("prof", "Prof", "profitability", "Compustat", "Gross profitability / book equity."),
    FeatureDefinition("q", "Q", "value", "CRSP+Compustat", "Tobin's Q."),
    FeatureDefinition("ret", "Ret", "price", "CRSP", "Monthly total return."),
    FeatureDefinition("ret_max", "Ret max", "price", "CRSP daily", "Maximum daily return in month."),
    FeatureDefinition("rna", "RNA", "profitability", "Compustat", "Return on net operating assets."),
    FeatureDefinition("roa", "ROA", "profitability", "Compustat", "Return on assets."),
    FeatureDefinition("roc", "ROC", "profitability", "CRSP+Compustat", "Return-on-capital measure in Appendix B."),
    FeatureDefinition("roe", "ROE", "profitability", "Compustat", "Return on equity."),
    FeatureDefinition("roic", "ROIC", "profitability", "Compustat", "Return on invested capital."),
    FeatureDefinition("s2c", "S2C", "profitability", "Compustat", "Sales / cash."),
    FeatureDefinition("sale_g", "Sale g", "investment", "Compustat", "Annual sales growth."),
    FeatureDefinition("sat", "SAT", "profitability", "Compustat", "Sales / assets."),
    FeatureDefinition("s2p", "S2P", "value", "CRSP+Compustat", "Sales / market equity."),
    FeatureDefinition("sga2s", "SGA2S", "profitability", "Compustat", "SG&A / sales."),
    FeatureDefinition("spread", "Spread", "friction", "CRSP daily", "Average daily bid-ask spread."),
    FeatureDefinition("std_turnover", "Std turnover", "friction", "CRSP daily", "Monthly SD of daily turnover."),
    FeatureDefinition("std_vol", "Std vol", "friction", "CRSP daily", "Monthly SD of daily volume."),
    FeatureDefinition("tan", "Tan", "intangibles", "Compustat", "Asset tangibility."),
    FeatureDefinition("total_vol", "Total vol", "price", "CRSP daily", "Monthly SD of daily returns."),
)

FEATURE_NAMES: tuple[str, ...] = tuple(item.name for item in FEATURE_DEFINITIONS)

if len(FEATURE_NAMES) != 51 or len(set(FEATURE_NAMES)) != 51:
    raise RuntimeError("Appendix B feature schema must contain 51 unique variables")

