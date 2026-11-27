"""Prepare the FRED-MD series for the factor-augmented VAR tutorial.

Produces two files, consumed by ``docs/tutorials/factor-augmented-var.py``:

- ``docs/tutorials/data/fred_md_1984_2007.csv``: from 1984-01 to 2007-12, the
  percent changes from the previous month of the four components of the
  composite index of coincident indicators (The Conference Board, Business
  Cycle Indicators Handbook, 2001: payroll employment, industrial production,
  real personal income less transfers, real manufacturing and trade sales) and
  of the consumer price index, and the effective federal funds rate in
  percent.
- ``docs/tutorials/data/fred_md_1984_2007_series.csv``: one row per series,
  with its FRED-MD mnemonic, description, source, FRED-MD's transformation
  code, the transformation code used, and the unit after the transformation.

Data source: FRED-MD (McCracken and Ng 2016), the monthly database for
macroeconomic research that the Federal Reserve Bank of St. Louis publishes at
<https://www.stlouisfed.org/research/economists/mccracken/fred-databases>. The
script reads the fixed vintage named in ``VINTAGE``, so it reproduces the
committed CSVs. Every series is published by the Bureau of Labor Statistics,
the Bureau of Economic Analysis, the Federal Reserve Board or the Federal
Reserve Bank of St. Louis, which splices real manufacturing and trade sales
from two Bureau of Economic Analysis series; FRED-MD's S&P, Conference Board,
ISM and University of Michigan series are copyrighted and not used.

Transformation codes; the logarithms are multiplied by 100. The change in the
logarithm from the previous month, times 100, approximates the percent change
from the previous month, and the CSVs call it that:

  1  x_t                                           (no transformation)
  5  100 (log x_t - log x_{t-1})                   (percent change from the previous month)
  6  100 (log x_t - 2 log x_{t-1} + log x_{t-2})   (change in that percent change)

Each series gets the code in the vintage's ``Transform:`` row, except that the
consumer price index, which FRED-MD codes 6, gets code 5: its percent change
from the previous month, consumer price inflation, is stationary over
1984-2007, and the change in consumer price inflation is mostly noise.
The federal funds rate is kept in percent, untransformed.

Re-run:  uv run python scripts/fetch_fred_md.py
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

VINTAGE = "2026-10"
URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/" + f"{VINTAGE}-md.csv"
START, END = "1984-01-01", "2007-12-01"
DATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "data"
PANEL_OUTPUT = DATA_DIR / "fred_md_1984_2007.csv"
SERIES_OUTPUT = DATA_DIR / "fred_md_1984_2007_series.csv"


class Series(NamedTuple):
    """One FRED-MD series: its mnemonic, a description and the institution that publishes it."""

    mnemonic: str
    description: str
    source: str


class Transformation(NamedTuple):
    """One FRED-MD transformation code: the map from the raw series and the unit of the result."""

    apply: Callable[[pd.Series], pd.Series]
    unit: str


# Column name in the committed CSV -> the FRED-MD series it holds.
SERIES = {
    "industrial_production": Series("INDPRO", "Industrial production: total index", "Federal Reserve Board"),
    "payroll_employment": Series(
        "PAYEMS", "Payroll employment: all employees, total nonfarm", "Bureau of Labor Statistics"
    ),
    "real_personal_income_less_transfers": Series(
        "W875RX1", "Real personal income less transfers", "Bureau of Economic Analysis"
    ),
    "real_manufacturing_trade_sales": Series(
        "CMRMTSPLx",
        "Real manufacturing and trade sales",
        "Federal Reserve Bank of St. Louis, spliced from two Bureau of Economic Analysis series",
    ),
    "consumer_price_inflation": Series("CPIAUCSL", "Consumer price index: all items", "Bureau of Labor Statistics"),
    "federal_funds_rate": Series("FEDFUNDS", "Effective federal funds rate", "Federal Reserve Board"),
}

# The code used for each series, where it differs from FRED-MD's.
CODE_USED = {"CPIAUCSL": 5, "FEDFUNDS": 1}

# The codes the selected series use. Code 1 leaves a series in its raw unit; the only series
# left untransformed is the federal funds rate, whose raw unit is percent.
TRANSFORMATIONS = {
    1: Transformation(apply=lambda series: series, unit="percent"),
    5: Transformation(apply=lambda series: 100 * np.log(series).diff(), unit="percent change from the previous month"),
}


def main() -> None:
    """Download the FRED-MD vintage, transform and trim the selected series, and write both CSVs."""
    with urllib.request.urlopen(URL, timeout=120) as response:  # noqa: S310
        text = response.read().decode("utf-8")

    raw = pd.read_csv(io.StringIO(text))
    if str(raw.iloc[0]["sasdate"]).strip() != "Transform:":
        raise ValueError("the first row after the header is not FRED-MD's transformation-code row")
    codes = raw.iloc[0].drop("sasdate").astype(float).astype(int)
    raw = raw.iloc[1:].dropna(subset=["sasdate"])
    raw.index = pd.to_datetime(raw.pop("sasdate"), format="%m/%d/%Y")
    raw = raw.astype(float)

    columns = {}
    rows = []
    for name, series in SERIES.items():
        fred_md_code = int(codes[series.mnemonic])
        code = CODE_USED.get(series.mnemonic, fred_md_code)
        if code not in TRANSFORMATIONS:
            raise ValueError(f"transformation code {code} of {series.mnemonic} is not in TRANSFORMATIONS")
        transformation = TRANSFORMATIONS[code]
        columns[name] = transformation.apply(raw[series.mnemonic])
        rows.append({
            "name": name,
            "fred_md_mnemonic": series.mnemonic,
            "description": series.description,
            "source": series.source,
            "fred_md_transformation_code": fred_md_code,
            "transformation_code": code,
            "unit": transformation.unit,
        })

    panel = pd.DataFrame(columns).loc[START:END]
    panel.index.name = "date"
    if panel.isna().any().any():
        missing = panel.columns[panel.isna().any()].tolist()
        raise ValueError(f"missing values between {START} and {END} in {missing}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.round(6).to_csv(PANEL_OUTPUT)
    pd.DataFrame(rows).to_csv(SERIES_OUTPUT, index=False)
    print(f"wrote {len(panel)} months and {panel.shape[1]} columns from FRED-MD vintage {VINTAGE} -> {PANEL_OUTPUT}")


if __name__ == "__main__":
    main()
