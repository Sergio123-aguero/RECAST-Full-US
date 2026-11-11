# Obtaining the Full US data

The historical experiments use licensed US equity data. Sign in to Wharton Research Data Services (WRDS) with your own institutional account and open **Get Data**. Your institutional WRDS representative can confirm access to each product below, including the separately subscribed CRSP/Compustat link history. Keep vendor exports and the resulting feature and execution files in your licensed workspace. The CPU quickstart uses generated inputs and can be run immediately. [WRDS account registration](https://wrds-www.wharton.upenn.edu/register/) · [CPU quickstart](../README.md#quickstart)

## What to request

| Source in WRDS | Records to export | Use in RECAST |
| --- | --- | --- |
| CRSP US Stock, Stock Files Version 2 (CIZ) | Daily stock file `dsf_v2` by `permno, dlycaldt` and monthly stock file `msf_v2` by `permno, mthcaldt`. Include prices, returns and their flags, volume, shares, quotes, and dated security and exchange classifications. | Daily price and volume features, monthly market characteristics and outcomes, and inputs for execution and valuation. [CRSP](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/) |
| CRSP corporate actions | Stock distributions (`stkdistributions`, `disexdt`) and delistings (`stkdelists`, `delistingdt`). Retain the security identifier and event, payment, and terminal-value fields where available. | Account for distributions and settle existing positions after a security exits. |
| Compustat North America and CRSP/Compustat Merged (CCM) | Fundamentals Annual (`comp.funda`) by `gvkey, datadate`, with publication dates where available. From CCM Link History obtain `gvkey, lpermno, linkdt, linkenddt, linktype, linkprim`. | Join company accounts to securities over valid link intervals and apply the financial reporting lag. [Compustat](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/sp-global-market-intelligence/) · [CCM query forms](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-compustat-with-crsp/) |
| WRDS Financial Ratios and daily US Fama/French research factors | Firm-level ratios with their identifiers and dates. For risk features, obtain daily market excess return, SMB, HML and the risk-free rate. | Monthly valuation, profitability, beta and idiosyncratic-volatility features. [Firm-level ratios](https://wrds-www.wharton.upenn.edu/pages/grid-items/financial-ratios-firm-level/) · [Daily factor series](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/data_library.html) |

The daily experiment trains over **2016–2017**, selects the checkpoint on **2018–2020**, and tests over **2021–2025**. Download earlier history too. Each decision uses 252 trading days of daily features and the latest 24 completed months of monthly features. Rolling characteristics and lagged annual accounts need observations before the first input window.

Export the historical cross-section, including securities that subsequently delisted. The reference Full US cohort contains 3,454 common stocks anchored to year-end 2015 exchange membership and screened using pre-2021 records. Use the CRSP permanent security identifier (`PERMNO`) across ticker changes. Apply eligibility and execution rules on each date and retain the information needed to value existing positions.

## Download one partition with Python

Install the WRDS client with `python -m pip install wrds` and run the example from the repository root. It uses your WRDS login and saves one month of daily observations on your machine. For a larger export, repeat with adjacent date ranges and select the full set of fields for each input above. The WRDS web query forms can also export by date and variable. Record the library, fields, dates and extraction date alongside each local export. [Official WRDS Python client](https://github.com/wharton/wrds)

```python
from pathlib import Path
import wrds

out = Path("wrds-downloads")
out.mkdir(exist_ok=True)
db = wrds.Connection()
try:
    daily = db.raw_sql(
        """
        SELECT permno, dlycaldt, dlyprc, dlyret, dlyvol, shrout
        FROM crsp.dsf_v2
        WHERE dlycaldt >= %(start)s AND dlycaldt < %(stop)s
        ORDER BY permno, dlycaldt
        """,
        params={"start": "2016-01-01", "stop": "2016-02-01"},
        date_cols=["dlycaldt"],
    )
    daily.to_csv(out / "crsp-daily-2016-01.csv.gz", index=False)
finally:
    db.close()
```

## From exports to model inputs

The model reads 71 ordered daily features and 51 ordered monthly features, along with observation masks. Execution and valuation use separate, unnormalized market inputs. For preparing your own panels, the repository provides the [feature order and normalization rules](../training/src/alphaportfolio_pcrr/dq_v1/feature_contract.py), [daily feature calculations](../training/src/alphaportfolio_pcrr/dq_v1/feature_expansion.py), [accounting and CCM joins](../training/src/alphaportfolio_original/data/fundamentals.py), and [monthly panel assembly](../training/src/alphaportfolio_original/data/assemble.py).

The local data verifier checks filenames, sizes and SHA-256 digests against the manifests that accompany your prepared inputs. The materializer accepts a training/validation `shared/` directory with `MANIFEST.json` and a test-transfer directory containing `data/dq-test/MANIFEST.json` and `SOURCE_MANIFEST.json`. WRDS delivers source tables. Preparing the model-ready panels, masks and valuation inputs from a fresh export requires local preprocessing. [Materialization and verification commands](../README.md#historical-data-and-evaluation-periods)
