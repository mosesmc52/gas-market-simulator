"""
calibration.py

Calibrate a simple natural gas market simulator scenario from EIA data using
mosesmc52/eia-ng / eia-ng-client.

Install:
    pip install eia-ng-client pandas numpy scikit-learn

Environment:
    export EIA_API_KEY="your_key"

Usage:
    from calibration import calibrate_reference_scenario

    scenario, monthly = calibrate_reference_scenario(start="2018-01")
    print(scenario)

Then pass `scenario` into your simulator's `run_scenario()` function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from eia_ng import EIAClient
from sklearn.linear_model import LinearRegression

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class CalibrationConfig:
    """Controls how EIA history is converted into simulator parameters."""

    start: str = "2018-01"
    end: Optional[str] = None
    storage_region: str = "lower48"
    lng_exports_key: str = "united_states_lng_total"
    production_state: Optional[str] = "united_states_total"
    demand_state: Optional[str] = "united_states_total"
    price_floor: float = 0.50
    fallback_supply_elasticity: float = 120.0
    fallback_demand_elasticity: float = -90.0
    pipeline_capacity_quantile: float = 0.95
    recent_months_for_initial_storage: int = 1
    api_key = None


# -----------------------------
# Generic cleanup helpers
# -----------------------------

def _to_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df

    # Normalize likely EIA column names.
    rename_candidates = {
        "period": "date",
        "value": "value",
        "duoarea": "area",
        "area-name": "area_name",
        "series-description": "series_description",
        "units": "units",
        "value-units": "units",
    }
    df = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns})

    if "date" not in df.columns:
        raise ValueError(f"Could not find date/period column in EIA result. Columns={list(df.columns)}")

    if "value" not in df.columns:
        # Some EIA responses use a named data column. Pick a numeric-looking fallback.
        numeric_candidates = [c for c in df.columns if c.lower() in {"price", "volume", "quantity", "generation"}]
        if numeric_candidates:
            df = df.rename(columns={numeric_candidates[0]: "value"})
        else:
            raise ValueError(f"Could not find value column in EIA result. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df


def _monthly_series(df: pd.DataFrame, name: str, how: str = "sum") -> pd.Series:
    """
    Convert an EIA dataframe with date/value into a monthly time series.

    how='sum' is useful for monthly volumes if source is daily/weekly.
    how='mean' is useful for prices and storage levels.
    how='last' is useful for ending storage inventory.
    """
    if df.empty:
        return pd.Series(dtype=float, name=name)

    s = df.set_index("date")["value"].sort_index()

    if how == "sum":
        out = s.resample("MS").sum(min_count=1)
    elif how == "mean":
        out = s.resample("MS").mean()
    elif how == "last":
        out = s.resample("MS").last()
    else:
        raise ValueError("how must be one of: sum, mean, last")

    out.name = name
    return out.dropna()


def _maybe_mmcf_to_bcf(s: pd.Series, raw_df: pd.DataFrame) -> pd.Series:
    """
    Convert MMcf to Bcf when EIA unit metadata suggests MMcf.
    If units are absent, assume the eia-ng wrapper already returns the correct values.
    """
    units = " ".join(str(x).lower() for x in raw_df.get("units", pd.Series(dtype=str)).dropna().unique())
    if "mmcf" in units or "million cubic feet" in units:
        return s / 1000.0
    return s


def _month_dummies(index) -> pd.DataFrame:
    index = pd.to_datetime(index)
    return pd.get_dummies(index.month, prefix="m", drop_first=True).set_index(index)


# -----------------------------
# EIA loading through eia-ng
# -----------------------------

def load_eia_monthly_panel(config: CalibrationConfig) -> pd.DataFrame:
    """
    Load and align the EIA series needed to calibrate the simulator.

    This uses public eia-ng methods shown in the repo README:
      client.natural_gas.production(...)
      client.natural_gas.consumption(...)
      client.natural_gas.storage(...)
      client.natural_gas.exports(...)
      client.natural_gas.spot_prices(...)

    If your local method signatures differ, adjust only this function.
    """


    client = EIAClient(api_key=config.api_key or os.getenv("EIA_API_KEY"))
    ng = client.natural_gas

    # Production: U.S. total unless a state is supplied.
    production_rows = ng.production(start=config.start, state=config.production_state)
    production_raw = _to_frame(production_rows)
    production = _monthly_series(production_raw, "production_bcf", how="sum")
    production = _maybe_mmcf_to_bcf(production, production_raw)

    # Consumption: U.S. total unless a state is supplied.
    consumption_rows = ng.consumption(start=config.start, state=config.demand_state)
    consumption_raw = _to_frame(consumption_rows)
    demand = _monthly_series(consumption_raw, "demand_bcf", how="sum")
    demand = _maybe_mmcf_to_bcf(demand, consumption_raw)

    # Industrial consumption is optional in some wrappers. Try common argument names.
    industrial = pd.Series(dtype=float, name="industrial_demand_bcf")
    for kwargs in (
        {"sector": "industrial"},
        {"consumer": "industrial"},
        {"end_use": "industrial"},
    ):
        try:
            industrial_rows = ng.consumption(start=config.start, state=config.demand_state, **kwargs)
            industrial_raw = _to_frame(industrial_rows)
            industrial = _monthly_series(industrial_raw, "industrial_demand_bcf", how="sum")
            industrial = _maybe_mmcf_to_bcf(industrial, industrial_raw)
            break
        except TypeError:
            continue
        except Exception:
            continue

    # Storage: weekly Lower 48 working gas. Use monthly ending inventory.
    storage_rows = ng.storage(start=config.start, region=config.storage_region)
    storage_raw = _to_frame(storage_rows)
    storage = _monthly_series(storage_raw, "storage_bcf", how="last")
    storage = _maybe_mmcf_to_bcf(storage, storage_raw)

    # LNG exports. eia-ng README shows exports(country=...). Use LNG key as default.
    exports_rows = ng.exports(start=config.start, country=config.lng_exports_key)
    exports_raw = _to_frame(exports_rows)
    lng_exports = _monthly_series(exports_raw, "lng_exports_bcf", how="sum")
    lng_exports = _maybe_mmcf_to_bcf(lng_exports, exports_raw)

    # Henry Hub daily spot prices. Monthly average.
    price_rows = ng.spot_prices(start=config.start)
    price_raw = _to_frame(price_rows)
    price = _monthly_series(price_raw, "henry_hub_price", how="mean")

    panel = pd.concat([production, demand, industrial, storage, lng_exports, price], axis=1).sort_index()

    if config.end:
        panel = panel.loc[: pd.to_datetime(config.end)]

    # Keep rows with the core calibration fields.
    core = ["production_bcf", "demand_bcf", "storage_bcf", "lng_exports_bcf", "henry_hub_price"]
    panel = panel.dropna(subset=[c for c in core if c in panel.columns])

    if "industrial_demand_bcf" not in panel.columns or panel["industrial_demand_bcf"].dropna().empty:
        # Fallback: industrial demand is often roughly a subcomponent. Use 25% of total demand
        # as a placeholder until sector-specific consumption is wired correctly.
        panel["industrial_demand_bcf"] = 0.25 * panel["demand_bcf"]

    panel["storage_change_bcf"] = panel["storage_bcf"].diff()
    return panel


# -----------------------------
# Calibration logic
# -----------------------------

def estimate_seasonal_factors(monthly: pd.DataFrame, demand_col: str = "demand_bcf") -> list[float]:
    """Return 12 demand multipliers indexed Jan..Dec."""
    df = monthly.copy()
    df.index = pd.to_datetime(df.index)

    monthly_avg = df.groupby(df.index.month)[demand_col].mean()
    annual_avg = df[demand_col].mean()

    factors = (monthly_avg / annual_avg).reindex(range(1, 13)).fillna(1.0)
    return factors.tolist()


def estimate_weather_residuals(
    monthly: pd.DataFrame,
    seasonal_factors: list[float],
    demand_col: str = "demand_bcf",
) -> pd.Series:
    """Estimate demand residuals after removing monthly seasonality."""
    df = monthly.copy()
    df.index = pd.to_datetime(df.index)

    annual_avg = df[demand_col].mean()

    expected = [
        annual_avg * seasonal_factors[month - 1]
        for month in df.index.month
    ]

    return df[demand_col] - expected


def estimate_supply_elasticity(monthly: pd.DataFrame, fallback: float = 120.0) -> float:
    """
    Estimate Bcf/month supply response to $/MMBtu using lagged Henry Hub price.
    """
    df = monthly[["production_bcf", "henry_hub_price"]].copy()
    df["price_lag1"] = df["henry_hub_price"].shift(1)
    df = df.dropna()
    if len(df) < 18 or LinearRegression is None:
        return fallback

    X = df[["price_lag1"]].to_numpy()
    y = df["production_bcf"].to_numpy()
    model = LinearRegression().fit(X, y)
    coef = float(model.coef_[0])

    # Guardrail: if regression is unstable or wrong-signed, use fallback.
    if not np.isfinite(coef) or coef <= 0:
        return fallback
    return coef


def estimate_demand_elasticity(monthly: pd.DataFrame, fallback: float = -90.0) -> float:
    """
    Estimate Bcf/month demand response to $/MMBtu after month seasonality.
    """
    df = monthly[["demand_bcf", "henry_hub_price"]].dropna().copy()
    if len(df) < 18 or LinearRegression is None:
        return fallback

    dummies = _month_dummies(df.index)
    X = pd.concat([df[["henry_hub_price"]], dummies], axis=1).to_numpy()
    y = df["demand_bcf"].to_numpy()
    model = LinearRegression().fit(X, y)
    coef = float(model.coef_[0])

    # Guardrail: demand elasticity should usually be negative in this simple model.
    if not np.isfinite(coef) or coef >= 0:
        return fallback
    return coef


def calibrate_from_monthly_panel(
    monthly: pd.DataFrame,
    config: Optional[CalibrationConfig] = None,
) -> Dict[str, Any]:
    """Create the simulator scenario dictionary from aligned monthly EIA data."""

    
    config = config or CalibrationConfig()

    
    monthly = monthly.copy().sort_index()
    monthly = monthly.dropna(subset=["production_bcf", "demand_bcf", "storage_bcf", "lng_exports_bcf", "henry_hub_price"])

    monthly = monthly.copy()
    monthly.index = pd.to_datetime(monthly.index)
    if monthly.empty:
        raise ValueError("No monthly calibration data available after cleaning.")

    seasonal_factors = estimate_seasonal_factors(monthly)
    residual = estimate_weather_residuals(monthly, seasonal_factors)

    storage_change = monthly["storage_bcf"].diff().dropna()
    max_injection = float(max(storage_change.max(), 0.0)) if not storage_change.empty else 300.0
    max_withdrawal = float(abs(min(storage_change.min(), 0.0))) if not storage_change.empty else 500.0

    production_q = monthly["production_bcf"].quantile(config.pipeline_capacity_quantile)
    pipeline_capacity = float(production_q)

    recent_storage = monthly["storage_bcf"].tail(config.recent_months_for_initial_storage).mean()

    scenario = {
        "base_price": float(monthly["henry_hub_price"].mean()),
        "base_supply": float(monthly["production_bcf"].mean()),
        "base_demand": float(monthly["demand_bcf"].mean()),
        "industrial_demand": float(monthly["industrial_demand_bcf"].mean()),
        "lng_exports": float(monthly["lng_exports_bcf"].mean()),
        "pipeline_capacity": pipeline_capacity,
        "initial_storage": float(recent_storage),
        "storage_capacity": float(monthly["storage_bcf"].max()),
        "max_injection": max_injection,
        "max_withdrawal": max_withdrawal,
        "supply_elasticity": estimate_supply_elasticity(monthly, config.fallback_supply_elasticity),
        "demand_elasticity": estimate_demand_elasticity(monthly, config.fallback_demand_elasticity),
        "weather_mean": float(residual.mean()),
        "weather_volatility": float(residual.std(ddof=1)),
        "seasonal_factors": seasonal_factors,
        "calibration_start": str(monthly.index.min().date()),
        "calibration_end": str(monthly.index.max().date()),
        "calibration_config": asdict(config),
    }

    return scenario


def calibrate_reference_scenario(
    start: str = "2018-01",
    end: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Convenience function: load EIA data through eia-ng and return:
      1. scenario dict for the simulator
      2. cleaned monthly calibration dataframe
    """
    config = CalibrationConfig(start=start, end=end, **kwargs)
    monthly = load_eia_monthly_panel(config)
    scenario = calibrate_from_monthly_panel(monthly, config)
    return scenario, monthly


# -----------------------------
# Scenario modifiers
# -----------------------------

def make_shock_scenario(
    base: Dict[str, Any],
    *,
    name: str = "Shock",
    lng_export_pct: float = 0.0,
    demand_sigma: float = 0.0,
    supply_pct: float = 0.0,
    pipeline_capacity_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Create a calibrated scenario variant.

    Example:
        cold_lng = make_shock_scenario(
            scenario,
            lng_export_pct=0.15,
            demand_sigma=1.0,
            pipeline_capacity_pct=-0.05,
        )
    """
    s = dict(base)
    s["scenario_name"] = name
    s["lng_exports"] = base["lng_exports"] * (1.0 + lng_export_pct)
    s["base_demand"] = base["base_demand"] + demand_sigma * base.get("weather_volatility", 0.0)
    s["base_supply"] = base["base_supply"] * (1.0 + supply_pct)
    s["pipeline_capacity"] = base["pipeline_capacity"] * (1.0 + pipeline_capacity_pct)
    return s


if __name__ == "__main__":
    scenario, monthly_df = calibrate_reference_scenario(start="2018-01")
    print("Calibrated scenario:")
    for k, v in scenario.items():
        if k == "seasonal_factors":
            print(f"{k}: {[round(x, 3) for x in v]}")
        elif isinstance(v, float):
            print(f"{k}: {v:,.3f}")
        else:
            print(f"{k}: {v}")
