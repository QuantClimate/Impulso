"""Fetch the El Niño and food commodity price dataset for the El Niño tutorial.

Produces `docs/tutorials/data/el_nino_commodity_prices.csv`, consumed by
`docs/tutorials/el-nino-commodity-prices.py`: one row per quarter from
1950Q1 to 2024Q4, indexed by the first day of the quarter. Each value is the
mean of the quarter's three monthly values; a quarter with a month missing is
left empty.

Columns:

  nino12                                 Niño 1+2: sea-surface-temperature anomaly over 0-10S, 90W-80W (deg C)
  nino3                                  Niño 3: sea-surface-temperature anomaly over 5N-5S, 150W-90W (deg C)
  nino4                                  Niño 4: sea-surface-temperature anomaly over 5N-5S, 160E-150W (deg C)
  southern_oscillation_index             Southern Oscillation Index: standardised sea-level pressure at Tahiti
                                         minus standardised sea-level pressure at Darwin, divided by the
                                         standard deviation of that difference (standard deviations)
  equatorial_southern_oscillation_index  equatorial Southern Oscillation Index: standardised sea-level-pressure
                                         anomaly over the eastern equatorial Pacific (5N-5S, 130W-80W) minus
                                         that over Indonesia (5N-5S, 90E-140E), normalised (standard deviations)
  food_price_index                       food price index: the World Bank's Food price index, nominal US
                                         dollars, 2010 = 100
  consumer_price_index                   consumer price index: US consumer price index for all urban
                                         consumers, all items in U.S. city average, seasonally adjusted,
                                         1982-84 = 100

Sources and licences:

- The five climate indices are from the NOAA Climate Prediction Center
  <https://www.cpc.ncep.noaa.gov/data/indices/>. Niño 1+2, Niño 3 and Niño 4 are
  anomalies from the 1991-2020 monthly means of the Extended Reconstructed Sea
  Surface Temperature, version 5 (Huang et al. 2017), in the file
  `ersst5.nino.mth.91-20.ascii`. The Climate Prediction Center's index page
  listed this file as "Monthly ERSSTv5 (1991-2020 base period)" Niño-region
  data (Internet Archive copy of 15 January 2026,
  <https://web.archive.org/web/20260115121659/https://www.cpc.ncep.noaa.gov/data/indices/>).
  The current page lists ERSSTv6 products and no longer links the file, but
  the file is still served at the address in `NINO_REGIONS_URL` (last
  modified 5 August 2026, when it was checked on 27 November 2026). The
  index page lists `soi` as "(Stand Tahiti - Stand Darwin) Sea Level
  Pressure", whose `STANDARDIZED DATA` table is read here, and
  `reqsoi.for` as the equatorial Southern Oscillation Index, defined in the
  Climate Diagnostics Bulletin, Figure T2. National Weather Service
  information "is in the public domain, unless specifically noted otherwise"
  <https://www.weather.gov/disclaimer>.
- The food price index is from the World Bank Commodity Price Data (the Pink
  Sheet), file `CMO-Historical-Data-Monthly.xlsx`, sheet "Monthly Indices",
  column "Food" <https://www.worldbank.org/en/research/commodity-markets>.
  Licensed under Creative Commons Attribution 4.0
  <https://creativecommons.org/licenses/by/4.0/> (World Bank Data Catalog,
  dataset 0038238). The file's address changes with each monthly release; the
  address in `WORLD_BANK_URL` is the release of 3 November 2026. This script
  changes the series: it averages the monthly values to quarters, and the
  tutorial then deflates them by the consumer price index and takes 100 times
  the quarterly change in the logarithm.
- The consumer price index is series CUSR0000SA0 from the US Bureau of Labor
  Statistics public API <https://www.bls.gov/developers/>; the Bureau's series
  page <https://data.bls.gov/timeseries/CUSR0000SA0> gives its title, "All
  items in U.S. city average, all urban consumers, seasonally adjusted". The Bureau states
  that everything it publishes is in the public domain and asks users to cite
  it as the source <https://www.bls.gov/opub/copyright-information.htm>. The
  unregistered API returns at most 10 years per request and allows 25
  requests a day; this script makes 8.

Re-run (openpyxl reads the World Bank file and is not a project dependency):

    uv run --with openpyxl python scripts/fetch_el_nino_commodity_prices.py
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

NINO_REGIONS_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
SOUTHERN_OSCILLATION_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
EQUATORIAL_SOUTHERN_OSCILLATION_URL = "https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"
WORLD_BANK_URL = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)
BUREAU_OF_LABOR_STATISTICS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
CONSUMER_PRICE_INDEX_SERIES = "CUSR0000SA0"
USER_AGENT = "Mozilla/5.0 (compatible; impulso-docs-data)"
FIRST_YEAR, LAST_YEAR = 1950, 2024
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "data" / "el_nino_commodity_prices.csv"


def _get(url: str) -> bytes:
    """Download one file. The World Bank server refuses Python's default user agent, so send a generic one."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        return response.read()


def _climate_prediction_center_table(text: str, after: str | None = None) -> pd.Series:
    """Read a Climate Prediction Center table with one row per year and twelve monthly columns.

    Args:
        text: The file's text.
        after: If given, read only the table that follows the first line containing this text.

    Returns:
        Monthly values on a monthly `PeriodIndex`. The files' missing-value codes, -999.9 and 999.9,
        are dropped.
    """
    lines = text.splitlines()
    if after is not None:
        lines = lines[next(i for i, line in enumerate(lines) if after in line) + 1 :]
    values = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit():
            for month, value in enumerate(parts[1:], start=1):
                if abs(float(value)) < 999:
                    values[pd.Period(year=int(parts[0]), month=month, freq="M")] = float(value)
    return pd.Series(values).sort_index()


def _nino_anomalies() -> pd.DataFrame:
    """Niño 1+2, Niño 3 and Niño 4, monthly (deg C), from the Climate Prediction Center's Niño-region file."""
    table = pd.read_csv(io.BytesIO(_get(NINO_REGIONS_URL)), sep=r"\s+")
    # Header: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM: each region's temperature, then its anomaly.
    regions = ["nino12", "nino3", "nino4", "nino34"]
    table.columns = ["year", "month"] + [name for r in regions for name in (f"{r}_temperature", r)]
    table.index = pd.PeriodIndex.from_fields(year=table["year"], month=table["month"], freq="M")
    return table[["nino12", "nino3", "nino4"]]


def _post_json(url: str, body: dict) -> dict:
    """Send a JSON body by POST and return the decoded JSON reply."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        return json.load(response)


def _consumer_price_index() -> pd.Series:
    """The consumer price index, monthly, from the Bureau of Labor Statistics public API in 10-year requests."""
    values = {}
    for start in range(FIRST_YEAR, LAST_YEAR + 1, 10):
        body = {
            "seriesid": [CONSUMER_PRICE_INDEX_SERIES],
            "startyear": str(start),
            "endyear": str(min(start + 9, LAST_YEAR)),
        }
        payload = _post_json(BUREAU_OF_LABOR_STATISTICS_URL, body)
        if payload["status"] != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"Bureau of Labor Statistics API request for {start} failed: {payload['message']}")
        for row in payload["Results"]["series"][0]["data"]:
            if row["period"].startswith("M") and row["period"] != "M13" and row["value"] not in ("-", ""):
                values[pd.Period(year=int(row["year"]), month=int(row["period"][1:]), freq="M")] = float(row["value"])
    return pd.Series(values).sort_index()


def _food_price_index() -> pd.Series:
    """The food price index, monthly, from the Pink Sheet's "Monthly Indices" sheet."""
    sheet = pd.read_excel(io.BytesIO(_get(WORLD_BANK_URL)), sheet_name="Monthly Indices", header=None)
    food_columns = [c for c in sheet.columns if sheet.loc[5:8, c].astype(str).str.strip().eq("Food **").any()]
    if len(food_columns) != 1:
        raise ValueError(f"expected one 'Food **' column in the sheet header, found {food_columns}")
    rows = sheet[sheet[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    index = pd.PeriodIndex([pd.Period(year=int(s[:4]), month=int(s[5:]), freq="M") for s in rows[0]])
    return pd.Series(pd.to_numeric(rows[food_columns[0]].to_numpy(), errors="coerce"), index=index)


def main() -> None:
    """Download every source, average to quarters, and write the CSV."""
    monthly = _nino_anomalies()
    monthly["southern_oscillation_index"] = _climate_prediction_center_table(
        _get(SOUTHERN_OSCILLATION_URL).decode(), after="STANDARDIZED"
    )
    monthly["equatorial_southern_oscillation_index"] = _climate_prediction_center_table(
        _get(EQUATORIAL_SOUTHERN_OSCILLATION_URL).decode()
    )
    monthly = monthly.join(_food_price_index().rename("food_price_index"), how="outer")
    monthly = monthly.join(_consumer_price_index().rename("consumer_price_index"), how="outer")

    quarter = monthly.index.asfreq("Q")
    quarterly = monthly.groupby(quarter).mean().where(monthly.groupby(quarter).count() == 3)
    quarterly = quarterly.loc[f"{FIRST_YEAR}Q1" : f"{LAST_YEAR}Q4"]
    quarterly.index = quarterly.index.to_timestamp()
    quarterly.index.name = "quarter_start"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    quarterly.round(6).to_csv(OUTPUT)
    print(f"wrote {len(quarterly)} quarterly rows ({FIRST_YEAR}Q1..{LAST_YEAR}Q4) -> {OUTPUT}")


if __name__ == "__main__":
    main()
