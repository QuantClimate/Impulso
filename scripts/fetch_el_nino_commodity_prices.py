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
  palm_oil_price, rice_price,            crop prices: the World Bank's prices of palm oil, rice (Thai 5%),
  soybean_price, maize_price,            soybeans, maize and wheat (US hard red winter), nominal US dollars
  wheat_price                            per tonne
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
- The food price index and the five crop prices are from the World Bank
  Commodity Price Data (the Pink Sheet), file
  `CMO-Historical-Data-Monthly.xlsx`: sheet "Monthly Indices", column "Food",
  and sheet "Monthly Prices", the columns named in `CROP_PRICES`
  <https://www.worldbank.org/en/research/commodity-markets>.
  Licensed under Creative Commons Attribution 4.0
  <https://creativecommons.org/licenses/by/4.0/> (World Bank Data Catalog,
  dataset 0038238). The file's address changes with each monthly release; the
  address in `WORLD_BANK_URL` is the release of 3 November 2026. This script
  changes the series: it averages the monthly values to quarters. The tutorial
  then deflates the food price index by the consumer price index, forms a
  relative price from the crop prices, and takes 100 times the quarterly
  change in each logarithm.
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
# Column label in the Pink Sheet's "Monthly Prices" sheet: column name in the output.
CROP_PRICES = {
    "Palm oil": "palm_oil_price",
    "Rice, Thai 5%": "rice_price",
    "Soybeans": "soybean_price",
    "Maize": "maize_price",
    "Wheat, US HRW": "wheat_price",
}
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
    return table.loc[:, ["nino12", "nino3", "nino4"]]


def _post_json(url: str, body: dict) -> dict:
    """Send a JSON body by POST and return the decoded JSON reply."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        reply = response.read()
    try:
        return json.loads(reply)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{url} did not return JSON: {reply[:200]!r}") from error


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


def _pink_sheet_column(sheet: pd.DataFrame, label: str) -> pd.Series:
    """One monthly column of a Pink Sheet sheet, found by its label in the sheet's header rows."""
    columns = [c for c in sheet.columns if sheet.loc[3:8, c].astype(str).str.strip().eq(label).any()]
    if len(columns) != 1:
        raise ValueError(f"expected one {label!r} column in the sheet header, found {columns}")
    rows = sheet[sheet[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    index = pd.PeriodIndex([pd.Period(year=int(s[:4]), month=int(s[5:]), freq="M") for s in rows[0]])
    values = [float("nan") if isinstance(v, str) else float(v) for v in rows.loc[:, columns[0]]]  # text marks a gap
    return pd.Series(values, index=index)


def _world_bank_prices() -> pd.DataFrame:
    """The food price index and the five crop prices, monthly, from the Pink Sheet."""
    workbook = _get(WORLD_BANK_URL)
    indices = pd.read_excel(io.BytesIO(workbook), sheet_name="Monthly Indices", header=None)
    monthly_prices = pd.read_excel(io.BytesIO(workbook), sheet_name="Monthly Prices", header=None)
    prices = {"food_price_index": _pink_sheet_column(indices, "Food **")}
    prices |= {name: _pink_sheet_column(monthly_prices, label) for label, name in CROP_PRICES.items()}
    return pd.DataFrame(prices)


def main() -> None:
    """Download every source, average to quarters, and write the CSV."""
    monthly = _nino_anomalies()
    monthly["southern_oscillation_index"] = _climate_prediction_center_table(
        _get(SOUTHERN_OSCILLATION_URL).decode(), after="STANDARDIZED"
    )
    monthly["equatorial_southern_oscillation_index"] = _climate_prediction_center_table(
        _get(EQUATORIAL_SOUTHERN_OSCILLATION_URL).decode()
    )
    monthly = monthly.join(_world_bank_prices(), how="outer")
    monthly = monthly.join(_consumer_price_index().rename("consumer_price_index"), how="outer")

    quarter = pd.PeriodIndex(monthly.index).asfreq("Q")
    quarterly = pd.DataFrame(monthly.groupby(quarter).mean())
    quarterly = quarterly[monthly.groupby(quarter).count() == 3]  # a quarter with a month missing is left empty
    quarterly = quarterly.loc[f"{FIRST_YEAR}Q1" : f"{LAST_YEAR}Q4"]
    quarterly.index = quarterly.index.to_timestamp()
    quarterly.index.name = "quarter_start"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    quarterly.round(6).to_csv(OUTPUT)
    print(f"wrote {len(quarterly)} quarterly rows ({FIRST_YEAR}Q1..{LAST_YEAR}Q4) -> {OUTPUT}")


if __name__ == "__main__":
    main()
