"""
weather_data.py

Notebook-friendly NOAA GHCND weather utilities for the gas-market simulator.

Purpose
-------
Download daily NOAA GHCND station data, compute HDD/CDD, aggregate stations into
regional daily/monthly weather series, and merge HDD/CDD into the EIA monthly
calibration panel used by the GasMarket simulator.

This file intentionally contains no command-line argument parsing and no main()
entrypoint. It is designed to be imported from a notebook.

Typical notebook usage
----------------------
from weather_data import build_region_weather_monthly, merge_weather_into_monthly_panel

weather_monthly = build_region_weather_monthly(
    stations_csv="scripts/noaa/major_airports_by_state_ghcnd.csv",
    region="all",
    start="2018-01-01",
    end=None,
    cache_dir="data/raw/noaa/stations",
)

monthly_with_weather = merge_weather_into_monthly_panel(
    monthly_df,
    weather_monthly,
    region="lower_48",
)

Expected station CSV columns
----------------------------
Required:
    ghcnd_station_id
Optional:
    region          # east, midwest, south, west, lower_48, etc.
    pipeline        # legacy alternative to region
    station_name
    state           # used to map stations into weather regions when region='all'
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

NOAA_GHCND_ACCESS_BASE = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"
)

STATE_TO_WEATHER_REGION = {
    "CT": "east", "ME": "east", "MA": "east", "NH": "east", "RI": "east",
    "VT": "east", "NJ": "east", "NY": "east", "PA": "east",
    "IL": "midwest", "IN": "midwest", "MI": "midwest", "OH": "midwest",
    "WI": "midwest", "IA": "midwest", "KS": "midwest", "MN": "midwest",
    "MO": "midwest", "NE": "midwest", "ND": "midwest", "SD": "midwest",
    "DE": "south", "FL": "south", "GA": "south", "MD": "south", "NC": "south",
    "SC": "south", "VA": "south", "DC": "south", "WV": "south", "AL": "south",
    "KY": "south", "MS": "south", "TN": "south", "AR": "south", "LA": "south",
    "OK": "south", "TX": "south",
    "AZ": "west", "CO": "west", "ID": "west", "MT": "west", "NV": "west",
    "NM": "west", "UT": "west", "WY": "west", "AK": "west", "CA": "west",
    "HI": "west", "OR": "west", "WA": "west",
}


@dataclass(frozen=True)
class StationMetaItem:
    region: str
    ghcnd_station_id: str
    station_name: Optional[str] = None
    state: Optional[str] = None


def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


def compute_hdd_from_tavg_c(tavg_c: float, base_f: float = 65.0) -> float:
    """Heating degree days from average daily temperature in Celsius."""
    return max(0.0, base_f - c_to_f(tavg_c))


def compute_cdd_from_tavg_c(tavg_c: float, base_f: float = 65.0) -> float:
    """Cooling degree days from average daily temperature in Celsius."""
    return max(0.0, c_to_f(tavg_c) - base_f)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_start_date(start: Optional[str] = None, days_ago: Optional[int] = None) -> Optional[str]:
    """Resolve either explicit start date or relative days_ago into YYYY-MM-DD."""
    if start:
        return start
    if days_ago is not None and days_ago > 0:
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return None


def load_station_meta(stations_csv: str, region: str = "all") -> List[StationMetaItem]:
    """
    Load station metadata from CSV and optionally filter by region.

    If the CSV lacks a region column and region is 'all' or 'lower_48', all stations
    are treated as lower_48.
    """
    df = pd.read_csv(stations_csv)

    if "ghcnd_station_id" not in df.columns:
        raise ValueError("stations CSV missing required column: 'ghcnd_station_id'")

    if "region" not in df.columns:
        if "pipeline" in df.columns:
            df = df.rename(columns={"pipeline": "region"})
        else:
            if region.strip().lower() not in {"all", "lower_48"}:
                raise ValueError(
                    "stations CSV missing 'region' column. Use region='all'/'lower_48' "
                    "or provide a region column."
                )
            df = df.copy()
            df["region"] = "lower_48"

    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df["ghcnd_station_id"] = df["ghcnd_station_id"].astype(str).str.strip()

    region_key = region.strip().lower()
    if region_key != "all":
        df = df.loc[df["region"] == region_key].copy()

    if df.empty:
        raise ValueError(f"No stations found for region={region!r}")

    items: List[StationMetaItem] = []
    for _, r in df.iterrows():
        items.append(
            StationMetaItem(
                region=str(r["region"]),
                ghcnd_station_id=str(r["ghcnd_station_id"]),
                station_name=str(r["station_name"]) if "station_name" in df.columns and pd.notna(r.get("station_name")) else None,
                state=str(r["state"]).strip().upper() if "state" in df.columns and pd.notna(r.get("state")) else None,
            )
        )

    # De-duplicate by station id.
    seen = set()
    unique: List[StationMetaItem] = []
    for item in items:
        if item.ghcnd_station_id not in seen:
            unique.append(item)
            seen.add(item.ghcnd_station_id)
    return unique


def download_station_csv(
    station_id: str,
    cache_dir: str = "data/raw/noaa/stations",
    timeout: int = 60,
    force: bool = False,
) -> str:
    """Download one NOAA GHCND station CSV with simple local caching."""
    safe_mkdir(cache_dir)
    out_path = os.path.join(cache_dir, f"{station_id}.csv")

    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    url = f"{NOAA_GHCND_ACCESS_BASE}{station_id}.csv"
    headers = {"User-Agent": "EnergyAtlasGasSimulator/0.1 (notebook weather calibration)"}
    response = requests.get(url, timeout=timeout, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Failed download {station_id}: HTTP {response.status_code} url={url}")

    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def read_and_normalize_station_file(
    station_id: str,
    filepath: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    base_f: float = 65.0,
) -> pd.DataFrame:
    """
    Read one GHCND station CSV and return daily temperature/HDD/CDD data.

    GHCND daily temperature fields are stored in tenths of degrees Celsius.
    TAVG is preferred; if missing, average of TMIN/TMAX is used.
    """
    df = pd.read_csv(
        filepath,
        usecols=lambda c: c in {"DATE", "TAVG", "TMIN", "TMAX"},
        low_memory=False,
    )

    if "DATE" not in df.columns:
        raise ValueError(f"{station_id}: missing DATE column in {filepath}")

    keep_cols = ["DATE"] + [c for c in ("TAVG", "TMIN", "TMAX") if c in df.columns]
    df = df[keep_cols].copy().rename(columns={"DATE": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if start:
        df = df.loc[df["date"] >= pd.to_datetime(start)].copy()
    if end:
        df = df.loc[df["date"] <= pd.to_datetime(end)].copy()

    for col in ("TAVG", "TMIN", "TMAX"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0

    df["tavg_c"] = df["TAVG"] if "TAVG" in df.columns else pd.NA
    df["tmin_c"] = df["TMIN"] if "TMIN" in df.columns else pd.NA
    df["tmax_c"] = df["TMAX"] if "TMAX" in df.columns else pd.NA

    missing_tavg = df["tavg_c"].isna() & df["tmin_c"].notna() & df["tmax_c"].notna()
    df.loc[missing_tavg, "tavg_c"] = (df.loc[missing_tavg, "tmin_c"] + df.loc[missing_tavg, "tmax_c"]) / 2.0

    df["hdd"] = df["tavg_c"].apply(lambda x: compute_hdd_from_tavg_c(float(x), base_f) if pd.notna(x) else pd.NA)
    df["cdd"] = df["tavg_c"].apply(lambda x: compute_cdd_from_tavg_c(float(x), base_f) if pd.notna(x) else pd.NA)
    df["tavg_f"] = df["tavg_c"].apply(lambda x: c_to_f(float(x)) if pd.notna(x) else pd.NA)
    df["ghcnd_station_id"] = station_id

    return df[["ghcnd_station_id", "date", "tavg_c", "tavg_f", "tmin_c", "tmax_c", "hdd", "cdd"]].copy()


def download_and_normalize_stations(
    stations: Iterable[StationMetaItem],
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = "data/raw/noaa/stations",
    timeout: int = 60,
    force: bool = False,
    base_f: float = 65.0,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Download and normalize a collection of station files into one daily dataframe."""
    frames: List[pd.DataFrame] = []
    errors: List[str] = []

    for station in stations:
        try:
            filepath = download_station_csv(
                station.ghcnd_station_id,
                cache_dir=cache_dir,
                timeout=timeout,
                force=force,
            )
            df_station = read_and_normalize_station_file(
                station_id=station.ghcnd_station_id,
                filepath=filepath,
                start=start,
                end=end,
                base_f=base_f,
            )
            df_station["region"] = station.region
            df_station["station_name"] = station.station_name
            df_station["state"] = station.state
            frames.append(df_station)
        except Exception as exc:  # noqa: BLE001 - notebook utility should be tolerant by default
            msg = f"{station.ghcnd_station_id}: {exc}"
            errors.append(msg)
            if not continue_on_error:
                raise

    if not frames:
        raise RuntimeError(f"No station data loaded. Errors: {errors[:5]}")

    out = pd.concat(frames, ignore_index=True)
    out.attrs["errors"] = errors
    return out


def aggregate_region_daily(df_all: pd.DataFrame, region_id: str = "lower_48") -> pd.DataFrame:
    """Aggregate normalized station data to one regional daily HDD/CDD series."""
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("tavg_c", "tavg_f", "hdd", "cdd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df.dropna(subset=["tavg_c"]).copy()
    if valid.empty:
        raise ValueError(f"No valid temperature observations for region_id={region_id!r}")

    agg = (
        valid.groupby("date", as_index=False)
        .agg(
            n_stations_used=("ghcnd_station_id", "nunique"),
            tavg_c_median=("tavg_c", "median"),
            tavg_f_median=("tavg_f", "median"),
            hdd_median=("hdd", "median"),
            cdd_median=("cdd", "median"),
            tavg_c_mean=("tavg_c", "mean"),
            tavg_f_mean=("tavg_f", "mean"),
            hdd_mean=("hdd", "mean"),
            cdd_mean=("cdd", "mean"),
        )
        .sort_values("date")
    )
    agg.insert(0, "region_id", region_id)
    return agg


def aggregate_all_weather_regions_daily(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate station data into east/midwest/south/west plus lower_48.

    Requires a state column in station metadata to map stations to broad weather regions.
    """
    df = df_all.copy()
    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["weather_region"] = df["state"].map(STATE_TO_WEATHER_REGION)

    frames: List[pd.DataFrame] = []
    for region_id in ("east", "midwest", "south", "west"):
        subset = df.loc[df["weather_region"] == region_id].copy()
        if not subset.empty:
            frames.append(aggregate_region_daily(subset, region_id=region_id))

    frames.append(aggregate_region_daily(df, region_id="lower_48"))
    return pd.concat(frames, ignore_index=True)


def daily_to_monthly_weather(
    weather_daily: pd.DataFrame,
    value_method: str = "median",
) -> pd.DataFrame:
    """
    Convert daily regional weather to monthly HDD/CDD.

    HDD/CDD are summed across days. Temperature is averaged across days.
    value_method chooses median or mean station aggregation columns.
    """
    if value_method not in {"median", "mean"}:
        raise ValueError("value_method must be 'median' or 'mean'")

    df = weather_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    hdd_col = f"hdd_{value_method}"
    cdd_col = f"cdd_{value_method}"
    tavg_f_col = f"tavg_f_{value_method}"
    tavg_c_col = f"tavg_c_{value_method}"

    required = ["region_id", "month", hdd_col, cdd_col, tavg_f_col, tavg_c_col, "n_stations_used"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"weather_daily missing columns: {missing}")

    monthly = (
        df.groupby(["region_id", "month"], as_index=False)
        .agg(
            hdd=(hdd_col, "sum"),
            cdd=(cdd_col, "sum"),
            tavg_f=(tavg_f_col, "mean"),
            tavg_c=(tavg_c_col, "mean"),
            n_weather_days=("date", "nunique"),
            avg_stations_used=("n_stations_used", "mean"),
        )
        .rename(columns={"month": "date"})
        .sort_values(["region_id", "date"])
    )
    return monthly


def build_region_weather_daily(
    stations_csv: str,
    region: str = "all",
    start: Optional[str] = None,
    end: Optional[str] = None,
    days_ago: Optional[int] = None,
    cache_dir: str = "data/raw/noaa/stations",
    timeout: int = 60,
    force: bool = False,
    base_f: float = 65.0,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """
    High-level notebook function: station CSV -> daily regional HDD/CDD dataframe.

    region='all' returns east/midwest/south/west/lower_48 when states are available.
    region='lower_48' or any explicit region returns one aggregate region.
    """
    start_date = resolve_start_date(start=start, days_ago=days_ago)
    stations = load_station_meta(stations_csv, region=region)
    station_data = download_and_normalize_stations(
        stations,
        start=start_date,
        end=end,
        cache_dir=cache_dir,
        timeout=timeout,
        force=force,
        base_f=base_f,
        continue_on_error=continue_on_error,
    )

    if region.strip().lower() == "all":
        return aggregate_all_weather_regions_daily(station_data)
    return aggregate_region_daily(station_data, region_id=region.strip().lower())


def build_region_weather_monthly(
    stations_csv: str,
    region: str = "all",
    start: Optional[str] = None,
    end: Optional[str] = None,
    days_ago: Optional[int] = None,
    cache_dir: str = "data/raw/noaa/stations",
    timeout: int = 60,
    force: bool = False,
    base_f: float = 65.0,
    value_method: str = "median",
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """High-level notebook function: station CSV -> monthly regional HDD/CDD dataframe."""
    daily = build_region_weather_daily(
        stations_csv=stations_csv,
        region=region,
        start=start,
        end=end,
        days_ago=days_ago,
        cache_dir=cache_dir,
        timeout=timeout,
        force=force,
        base_f=base_f,
        continue_on_error=continue_on_error,
    )
    return daily_to_monthly_weather(daily, value_method=value_method)


def merge_weather_into_monthly_panel(
    monthly_panel: pd.DataFrame,
    weather_monthly: pd.DataFrame,
    region: str = "lower_48",
    hdd_col_name: str = "hdd",
    cdd_col_name: str = "cdd",
) -> pd.DataFrame:
    """
    Merge monthly HDD/CDD into the simulator's EIA monthly calibration panel.

    Returns a copy with columns:
        hdd, cdd, tavg_f, tavg_c, n_weather_days, avg_stations_used
    indexed by month start date.
    """
    panel = monthly_panel.copy()
    panel.index = pd.to_datetime(panel.index).to_period("M").to_timestamp()
    panel.index.name = "date"

    wx = weather_monthly.copy()
    wx["date"] = pd.to_datetime(wx["date"]).dt.to_period("M").dt.to_timestamp()
    wx = wx.loc[wx["region_id"] == region].copy()
    if wx.empty:
        raise ValueError(f"No weather rows found for region={region!r}")

    keep = ["date", "hdd", "cdd", "tavg_f", "tavg_c", "n_weather_days", "avg_stations_used"]
    wx = wx[keep].drop_duplicates(subset=["date"]).set_index("date").sort_index()
    wx = wx.rename(columns={"hdd": hdd_col_name, "cdd": cdd_col_name})

    return panel.join(wx, how="left")


def estimate_weather_demand_sensitivity(
    monthly_panel_with_weather: pd.DataFrame,
    demand_col: str = "demand_bcf",
    hdd_col: str = "hdd",
    cdd_col: str = "cdd",
) -> Dict[str, float]:
    """
    Estimate simple demand sensitivities to HDD/CDD using linear regression.

    Returns coefficients in Bcf/month per monthly HDD or CDD unit. This is a simple
    first-pass model, useful for feeding scenario analysis.
    """
    from sklearn.linear_model import LinearRegression

    df = monthly_panel_with_weather[[demand_col, hdd_col, cdd_col]].dropna().copy()
    if len(df) < 12:
        raise ValueError("Need at least 12 monthly observations with demand/HDD/CDD.")

    X = df[[hdd_col, cdd_col]].to_numpy()
    y = df[demand_col].to_numpy()
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    residual = y - y_hat

    return {
        "demand_intercept_bcf": float(model.intercept_),
        "hdd_sensitivity_bcf_per_hdd": float(model.coef_[0]),
        "cdd_sensitivity_bcf_per_cdd": float(model.coef_[1]),
        "weather_model_residual_std_bcf": float(pd.Series(residual).std(ddof=1)),
        "weather_model_r2": float(model.score(X, y)),
    }
