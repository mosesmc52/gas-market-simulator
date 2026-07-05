from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from calibration import calibrate_from_monthly_panel

WEEKS_PER_MONTH = 4.345
WEEKS_PER_YEAR = 52


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.has_duplicates:
        return df.copy()
    return df.T.groupby(level=0).last().T


def _monthly_total_to_weekly(series: pd.Series, weekly_index: pd.Index) -> pd.Series:
    monthly = series.copy()
    monthly.index = pd.to_datetime(monthly.index).to_period("M").to_timestamp()
    monthly = monthly.sort_index()
    if monthly.empty:
        return pd.Series(index=pd.to_datetime(weekly_index), dtype=float, name=series.name)

    daily_parts: list[pd.Series] = []
    for date, value in monthly.items():
        days_in_month = int(date.days_in_month)
        daily_index = pd.date_range(date, date + pd.offsets.MonthEnd(0), freq="D")
        daily_value = float(value) / days_in_month if pd.notna(value) else np.nan
        daily_parts.append(pd.Series(daily_value, index=daily_index))

    daily = pd.concat(daily_parts).sort_index()
    weekly = daily.resample("W-FRI").sum(min_count=1)
    weekly = weekly.reindex(pd.to_datetime(weekly_index))
    weekly.name = series.name
    return weekly


def _weekly_history_to_monthly_panel(weekly_df: pd.DataFrame) -> pd.DataFrame:
    weekly = _ensure_datetime_index(weekly_df)
    aggregations = {
        "production_bcf": "sum",
        "demand_bcf": "sum",
        "industrial_demand_bcf": "sum",
        "lng_exports_bcf": "sum",
        "storage_bcf": "last",
        "henry_hub_price": "mean",
        "hdd": "sum",
        "cdd": "sum",
    }
    available = {column: how for column, how in aggregations.items() if column in weekly.columns}
    monthly = weekly.resample("MS").agg(available)
    if "storage_bcf" in monthly.columns:
        monthly["storage_change_bcf"] = monthly["storage_bcf"].diff()
    return monthly.dropna(subset=[c for c in ("production_bcf", "demand_bcf", "storage_bcf", "lng_exports_bcf", "henry_hub_price") if c in monthly.columns])


def build_weekly_history_panel(
    monthly_df: pd.DataFrame,
    weekly_storage_df: pd.DataFrame,
    weekly_price_df: pd.DataFrame,
) -> pd.DataFrame:
    monthly = _ensure_datetime_index(monthly_df)
    weekly_storage = _ensure_datetime_index(weekly_storage_df)
    weekly_price = _ensure_datetime_index(weekly_price_df)

    weekly_index = weekly_storage.index.union(weekly_price.index).sort_values()
    weekly = pd.DataFrame(index=weekly_index)

    for column in ("production_bcf", "demand_bcf", "industrial_demand_bcf", "lng_exports_bcf"):
        if column in monthly.columns:
            weekly[column] = _monthly_total_to_weekly(monthly[column], weekly.index)

    weekly = weekly.join(weekly_storage[["storage_bcf"]], how="left")
    weekly = weekly.join(weekly_price[["henry_hub_price"]], how="left")
    weekly["storage_change_bcf"] = weekly["storage_bcf"].diff()
    return weekly.dropna(
        subset=[
            c
            for c in ("production_bcf", "demand_bcf", "storage_bcf", "lng_exports_bcf", "henry_hub_price")
            if c in weekly.columns
        ]
    )


def augment_scenario_with_seasonal_limits(
    base_scenario: Dict[str, Any],
    monthly_df: pd.DataFrame,
) -> Dict[str, Any]:
    monthly = _ensure_datetime_index(monthly_df)
    out = dict(base_scenario)

    storage_change = monthly["storage_bcf"].diff()
    pos = storage_change.where(storage_change > 0)
    neg = (-storage_change.where(storage_change < 0))

    injection_by_month = pos.groupby(monthly.index.month).mean().reindex(range(1, 13))
    withdrawal_by_month = neg.groupby(monthly.index.month).mean().reindex(range(1, 13))

    default_injection = float(out.get("max_injection", 300.0))
    default_withdrawal = float(out.get("max_withdrawal", 500.0))

    out["monthly_injection_limits"] = (
        injection_by_month.fillna(default_injection).astype(float).tolist()
    )
    out["monthly_withdrawal_limits"] = (
        withdrawal_by_month.fillna(default_withdrawal).astype(float).tolist()
    )

    production_month_factor = (
        monthly.groupby(monthly.index.month)["production_bcf"].mean() / monthly["production_bcf"].mean()
    ).reindex(range(1, 13)).fillna(1.0)
    lng_month_factor = (
        monthly.groupby(monthly.index.month)["lng_exports_bcf"].mean() / monthly["lng_exports_bcf"].mean()
    ).reindex(range(1, 13)).fillna(1.0)

    out["production_month_factors"] = production_month_factor.astype(float).tolist()
    out["lng_month_factors"] = lng_month_factor.astype(float).tolist()
    return out


def fit_price_model(
    monthly_df: pd.DataFrame,
    storage_capacity: float | None = None,
) -> Dict[str, Any]:
    monthly = _ensure_datetime_index(monthly_df)
    df = _drop_duplicate_columns(monthly)

    if storage_capacity is None:
        storage_capacity = float(df["storage_bcf"].max())

    df["storage_ratio"] = df["storage_bcf"] / storage_capacity
    seasonal_storage_ratio = (
        df.groupby(df.index.month)["storage_ratio"].mean().reindex(range(1, 13)).fillna(df["storage_ratio"].mean())
    )
    df["seasonal_storage_ratio"] = df.index.month.map(seasonal_storage_ratio.to_dict())
    df["storage_gap_ratio"] = df["storage_ratio"] - df["seasonal_storage_ratio"]
    df["storage_change_bcf"] = df["storage_bcf"].diff()
    df["balance_proxy_bcf"] = df["production_bcf"] - (df["demand_bcf"] + df["lng_exports_bcf"])
    df["price_lag1"] = df["henry_hub_price"].shift(1)
    df["hdd"] = pd.to_numeric(df.get("hdd", 0.0), errors="coerce").fillna(0.0)
    df["cdd"] = pd.to_numeric(df.get("cdd", 0.0), errors="coerce").fillna(0.0)

    feature_cols = [
        "price_lag1",
        "storage_gap_ratio",
        "storage_change_bcf",
        "balance_proxy_bcf",
        "hdd",
        "cdd",
    ]
    train = df.dropna(subset=feature_cols + ["henry_hub_price"]).copy()
    if len(train) < 24:
        raise ValueError("Need at least 24 monthly observations to fit the price model.")

    X = train[feature_cols].to_numpy(dtype=float)
    y = train["henry_hub_price"].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(train)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    fitted = X_design @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "feature_names": feature_cols,
        "intercept": float(beta[0]),
        "coefficients": {
            feature: float(coef) for feature, coef in zip(feature_cols, beta[1:])
        },
        "r2": r2,
        "training_rows": int(len(train)),
        "storage_capacity": float(storage_capacity),
        "seasonal_storage_ratio_by_month": {
            int(month): float(value) for month, value in seasonal_storage_ratio.items()
        },
        "price_floor": 0.50,
        "price_cap": 12.00,
    }


def predict_price_from_model(
    price_model: Dict[str, Any],
    *,
    storage_bcf: float,
    storage_change_bcf: float,
    balance_bcf: float,
    prev_price: float,
    forecast_date: pd.Timestamp,
    hdd: float = 0.0,
    cdd: float = 0.0,
    scale_to_monthly: bool = False,
) -> float:
    storage_capacity = float(price_model["storage_capacity"])
    storage_ratio = storage_bcf / storage_capacity if storage_capacity > 0 else 0.5
    seasonal_storage_ratio = float(
        price_model["seasonal_storage_ratio_by_month"].get(int(forecast_date.month), storage_ratio)
    )
    factor = WEEKS_PER_MONTH if scale_to_monthly else 1.0
    features = {
        "price_lag1": float(prev_price),
        "storage_gap_ratio": float(storage_ratio - seasonal_storage_ratio),
        "storage_change_bcf": float(storage_change_bcf * factor),
        "balance_proxy_bcf": float(balance_bcf * factor),
        "hdd": float(hdd * factor),
        "cdd": float(cdd * factor),
    }

    estimate = float(price_model["intercept"])
    for feature_name in price_model["feature_names"]:
        estimate += float(price_model["coefficients"][feature_name]) * features[feature_name]

    return float(
        np.clip(estimate, price_model.get("price_floor", 0.50), price_model.get("price_cap", 12.00))
    )


def _seasonal_limit(
    scenario: Dict[str, Any],
    month: int,
    kind: str,
) -> float:
    if kind == "injection":
        limits = scenario.get("monthly_injection_limits")
        fallback = float(scenario.get("max_injection", 300.0))
    else:
        limits = scenario.get("monthly_withdrawal_limits")
        fallback = float(scenario.get("max_withdrawal", 500.0))

    if isinstance(limits, Sequence) and len(limits) >= 12:
        return float(limits[month - 1])
    return fallback


def simulate_monthly_backtest(
    panel_df: pd.DataFrame,
    scenario: Dict[str, Any],
    price_model: Dict[str, Any],
    *,
    initial_storage: float | None = None,
    initial_price: float | None = None,
) -> pd.DataFrame:
    panel = _drop_duplicate_columns(_ensure_datetime_index(panel_df))
    storage = float(initial_storage if initial_storage is not None else panel["storage_bcf"].iloc[0])
    price = float(initial_price if initial_price is not None else panel["henry_hub_price"].iloc[0])
    storage_capacity = float(scenario.get("storage_capacity", panel["storage_bcf"].max()))
    pipeline_capacity = float(scenario.get("pipeline_capacity", panel["production_bcf"].quantile(0.95)))
    response_factor = float(scenario.get("storage_response_factor", 0.95))

    rows = []
    for i, (date, row) in enumerate(panel.iterrows(), start=1):
        storage_start = storage
        raw_supply = float(row["production_bcf"])
        delivered_supply = min(raw_supply, pipeline_capacity)
        demand = float(row["demand_bcf"] + row["lng_exports_bcf"])
        balance = delivered_supply - demand

        max_injection = _seasonal_limit(scenario, date.month, "injection")
        max_withdrawal = _seasonal_limit(scenario, date.month, "withdrawal")
        if balance >= 0:
            storage_change = min(balance, storage_capacity - storage, max_injection)
        else:
            storage_change = -min(-balance, storage, max_withdrawal)
        storage_change *= response_factor
        storage = float(np.clip(storage + storage_change, 0.0, storage_capacity))

        price = predict_price_from_model(
            price_model,
            storage_bcf=storage,
            storage_change_bcf=storage_change,
            balance_bcf=balance,
            prev_price=price,
            forecast_date=pd.Timestamp(date),
            hdd=float(row.get("hdd", 0.0) or 0.0),
            cdd=float(row.get("cdd", 0.0) or 0.0),
            scale_to_monthly=False,
        )

        rows.append(
            {
                "month": i,
                "date": pd.Timestamp(date),
                "price": round(price, 3),
                "raw_supply_bcf": round(raw_supply, 1),
                "delivered_supply_bcf": round(delivered_supply, 1),
                "demand_bcf": round(demand, 1),
                "balance_bcf": round(balance, 1),
                "storage_start_bcf": round(storage_start, 1),
                "storage_bcf": round(storage, 1),
                "storage_change_bcf": round(storage_change, 1),
                "storage_ratio": round(storage / storage_capacity, 3) if storage_capacity > 0 else np.nan,
                "pipeline_capacity_bcf": pipeline_capacity,
                "lng_exports_bcf": round(float(row["lng_exports_bcf"]), 1),
            }
        )

    return pd.DataFrame(rows)


def build_monthly_comparison(panel_df: pd.DataFrame, simulated_df: pd.DataFrame) -> pd.DataFrame:
    historical = _ensure_datetime_index(panel_df).reset_index().rename(columns={"index": "date"})
    historical["month"] = range(1, len(historical) + 1)
    comparison = historical.merge(simulated_df, on=["month", "date"], suffixes=("_real", "_sim"))

    comparison = comparison.rename(
        columns={
            "production_bcf": "supply_real",
            "demand_bcf": "demand_real",
            "demand_bcf_real": "demand_real",
            "demand_bcf_sim": "demand_sim",
            "lng_exports_bcf_real": "lng_exports_real",
            "henry_hub_price": "price_real",
            "price": "price_pressure_sim",
            "storage_bcf_real": "storage_real",
            "storage_bcf_sim": "storage_end_sim",
            "storage_change_bcf_real": "storage_change_real",
            "storage_change_bcf_sim": "storage_change_sim",
            "raw_supply_bcf": "supply_sim",
            "lng_exports_bcf_sim": "lng_exports_sim",
        }
    )
    comparison["total_demand_real"] = comparison["demand_real"] + comparison["lng_exports_real"]
    return comparison


def score_monthly_comparison(comparison: pd.DataFrame) -> Dict[str, float]:
    df = comparison.copy()
    df["storage_error"] = df["storage_end_sim"] - df["storage_real"]
    df["storage_change_error"] = df["storage_change_sim"] - df["storage_change_real"]
    df["price_error"] = df["price_pressure_sim"] - df["price_real"]

    storage_mae = float(df["storage_error"].abs().mean())
    storage_change_mae = float(df["storage_change_error"].abs().mean())
    price_mae = float(df["price_error"].abs().mean())

    valid_storage_change = df.dropna(subset=["storage_change_real", "storage_change_sim"])
    storage_direction_accuracy = float(
        (np.sign(valid_storage_change["storage_change_real"]) == np.sign(valid_storage_change["storage_change_sim"])).mean()
    )

    df["real_price_change"] = df["price_real"].diff()
    df["sim_price_change"] = df["price_pressure_sim"].diff()
    valid_price = df.dropna(subset=["real_price_change", "sim_price_change"])
    price_direction_accuracy = float(
        (np.sign(valid_price["real_price_change"]) == np.sign(valid_price["sim_price_change"])).mean()
    )

    return {
        "storage_mae_bcf": round(storage_mae, 1),
        "storage_change_mae_bcf": round(storage_change_mae, 1),
        "price_mae": round(price_mae, 3),
        "storage_change_direction_accuracy": round(storage_direction_accuracy, 3),
        "price_direction_accuracy": round(price_direction_accuracy, 3),
    }


def rolling_origin_validation(
    monthly_df: pd.DataFrame,
    *,
    train_min_months: int = 36,
    horizons: Iterable[int] = (1, 3),
) -> pd.DataFrame:
    monthly = _ensure_datetime_index(monthly_df)
    rows: list[Dict[str, Any]] = []

    for horizon in horizons:
        last_start = len(monthly) - horizon
        for train_end in range(train_min_months, last_start + 1):
            train = monthly.iloc[:train_end].copy()
            future = monthly.iloc[train_end : train_end + horizon].copy()
            if len(future) < horizon:
                continue

            scenario = augment_scenario_with_seasonal_limits(
                calibrate_from_monthly_panel(train),
                train,
            )
            price_model = fit_price_model(train, storage_capacity=scenario["storage_capacity"])
            simulated = simulate_monthly_backtest(
                future,
                scenario,
                price_model,
                initial_storage=float(train["storage_bcf"].iloc[-1]),
                initial_price=float(train["henry_hub_price"].iloc[-1]),
            )
            comparison = build_monthly_comparison(future, simulated)
            score = score_monthly_comparison(comparison)
            rows.append(
                {
                    "train_end_date": pd.Timestamp(train.index[-1]),
                    "forecast_start_date": pd.Timestamp(future.index[0]),
                    "horizon_months": int(horizon),
                    **score,
                }
            )

    return pd.DataFrame(rows)


def simulate_weekly_historical_backtest(
    panel_df: pd.DataFrame,
    scenario: Dict[str, Any],
    price_model: Dict[str, Any],
    *,
    initial_storage: float | None = None,
    initial_price: float | None = None,
) -> pd.DataFrame:
    panel = _drop_duplicate_columns(_ensure_datetime_index(panel_df))
    storage = float(initial_storage if initial_storage is not None else panel["storage_bcf"].iloc[0])
    price = float(initial_price if initial_price is not None else panel["henry_hub_price"].iloc[0])
    storage_capacity = float(scenario.get("storage_capacity", panel["storage_bcf"].max()))
    pipeline_capacity = float(scenario.get("pipeline_capacity", panel["production_bcf"].quantile(0.95))) / WEEKS_PER_MONTH
    response_factor = float(scenario.get("storage_response_factor", 0.95))

    rows = []
    for i, (date, row) in enumerate(panel.iterrows(), start=1):
        storage_start = storage
        raw_supply = float(row["production_bcf"])
        delivered_supply = min(raw_supply, pipeline_capacity)
        demand = float(row["demand_bcf"] + row["lng_exports_bcf"])
        balance = delivered_supply - demand

        max_injection = _seasonal_limit(scenario, date.month, "injection") / WEEKS_PER_MONTH
        max_withdrawal = _seasonal_limit(scenario, date.month, "withdrawal") / WEEKS_PER_MONTH
        if balance >= 0:
            storage_change = min(balance, storage_capacity - storage, max_injection)
        else:
            storage_change = -min(-balance, storage, max_withdrawal)
        storage_change *= response_factor
        storage = float(np.clip(storage + storage_change, 0.0, storage_capacity))

        price = predict_price_from_model(
            price_model,
            storage_bcf=storage,
            storage_change_bcf=storage_change,
            balance_bcf=balance,
            prev_price=price,
            forecast_date=pd.Timestamp(date),
            hdd=float(row.get("hdd", 0.0) or 0.0),
            cdd=float(row.get("cdd", 0.0) or 0.0),
            scale_to_monthly=True,
        )

        rows.append(
            {
                "week": i,
                "date": pd.Timestamp(date),
                "price": round(price, 3),
                "raw_supply_bcf": round(raw_supply, 2),
                "delivered_supply_bcf": round(delivered_supply, 2),
                "demand_bcf": round(demand, 2),
                "balance_bcf": round(balance, 2),
                "storage_start_bcf": round(storage_start, 1),
                "storage_bcf": round(storage, 1),
                "storage_change_bcf": round(storage_change, 2),
                "storage_ratio": round(storage / storage_capacity, 3) if storage_capacity > 0 else np.nan,
                "pipeline_capacity_bcf": round(pipeline_capacity, 2),
                "lng_exports_bcf": round(float(row["lng_exports_bcf"]), 2),
            }
        )

    return pd.DataFrame(rows)


def build_weekly_comparison(panel_df: pd.DataFrame, simulated_df: pd.DataFrame) -> pd.DataFrame:
    historical = _ensure_datetime_index(panel_df).reset_index().rename(columns={"index": "date"})
    historical["week"] = range(1, len(historical) + 1)
    comparison = historical.merge(simulated_df, on=["week", "date"], suffixes=("_real", "_sim"))

    comparison = comparison.rename(
        columns={
            "production_bcf": "supply_real",
            "demand_bcf": "demand_real",
            "demand_bcf_real": "demand_real",
            "demand_bcf_sim": "demand_sim",
            "lng_exports_bcf_real": "lng_exports_real",
            "henry_hub_price": "price_real",
            "price": "price_pressure_sim",
            "storage_bcf_real": "storage_real",
            "storage_bcf_sim": "storage_end_sim",
            "storage_change_bcf_real": "storage_change_real",
            "storage_change_bcf_sim": "storage_change_sim",
            "raw_supply_bcf": "supply_sim",
            "lng_exports_bcf_sim": "lng_exports_sim",
        }
    )
    comparison["total_demand_real"] = comparison["demand_real"] + comparison["lng_exports_real"]
    return comparison


def score_weekly_comparison(comparison: pd.DataFrame) -> Dict[str, float]:
    return score_monthly_comparison(comparison)


def rolling_origin_validation_weekly(
    weekly_df: pd.DataFrame,
    *,
    train_min_weeks: int = 104,
    horizons: Iterable[int] = (1, 4, 13),
) -> pd.DataFrame:
    weekly = _ensure_datetime_index(weekly_df)
    rows: list[Dict[str, Any]] = []

    for horizon in horizons:
        last_start = len(weekly) - horizon
        for train_end in range(train_min_weeks, last_start + 1):
            train = _drop_duplicate_columns(weekly.iloc[:train_end])
            future = _drop_duplicate_columns(weekly.iloc[train_end : train_end + horizon])
            if len(future) < horizon:
                continue

            train_monthly = _weekly_history_to_monthly_panel(train)
            if len(train_monthly) < 12:
                continue

            scenario = augment_scenario_with_seasonal_limits(
                calibrate_from_monthly_panel(train_monthly),
                train_monthly,
            )
            price_model = fit_price_model(train, storage_capacity=scenario["storage_capacity"])
            simulated = simulate_weekly_historical_backtest(
                future,
                scenario,
                price_model,
                initial_storage=float(train["storage_bcf"].iloc[-1]),
                initial_price=float(train["henry_hub_price"].iloc[-1]),
            )
            comparison = build_weekly_comparison(future, simulated)
            score = score_weekly_comparison(comparison)
            rows.append(
                {
                    "train_end_date": pd.Timestamp(train.index[-1]),
                    "forecast_start_date": pd.Timestamp(future.index[0]),
                    "horizon_weeks": int(horizon),
                    **score,
                }
            )

    return pd.DataFrame(rows)


def _sample_weather_for_week(
    monthly_df: pd.DataFrame,
    *,
    month: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    candidates = monthly_df.loc[monthly_df.index.month == month, ["hdd", "cdd"]].dropna()
    if candidates.empty:
        candidates = monthly_df[["hdd", "cdd"]].dropna()
    choice = candidates.iloc[int(rng.integers(0, len(candidates)))]
    return float(choice["hdd"]) / WEEKS_PER_MONTH, float(choice["cdd"]) / WEEKS_PER_MONTH


def _weekly_supply(
    scenario: Dict[str, Any],
    month: int,
    rng: np.random.Generator,
) -> float:
    base_supply = float(scenario.get("base_supply", 3100.0))
    month_factor = float(scenario.get("production_month_factors", [1.0] * 12)[month - 1])
    weekly_supply = base_supply * month_factor / WEEKS_PER_MONTH
    noise_bcf = float(scenario.get("supply_noise_bcf", 0.0)) / np.sqrt(WEEKS_PER_MONTH)
    noise_pct = float(scenario.get("supply_noise_pct", 0.0))
    noise = 0.0
    if noise_bcf > 0:
        noise += float(rng.normal(0.0, noise_bcf))
    if noise_pct > 0:
        noise += float(rng.normal(0.0, weekly_supply * noise_pct))
    return max(0.0, weekly_supply + noise)


def _weekly_lng_exports(
    scenario: Dict[str, Any],
    month: int,
) -> float:
    base_lng = float(scenario.get("lng_exports", 400.0))
    month_factor = float(scenario.get("lng_month_factors", [1.0] * 12)[month - 1])
    return max(0.0, base_lng * month_factor / WEEKS_PER_MONTH)


def _weekly_domestic_demand(
    scenario: Dict[str, Any],
    month: int,
    hdd: float,
    cdd: float,
    rng: np.random.Generator,
) -> float:
    seasonal_factor = float(scenario.get("seasonal_factors", [1.0] * 12)[month - 1])
    base_demand = float(scenario.get("weather_base_demand", scenario.get("base_demand", 2900.0)))
    monthly_base = base_demand * seasonal_factor
    weather_component = (
        float(scenario.get("hdd_sensitivity", 0.0)) * float(scenario.get("hdd_sensitivity_scale", 1.0)) * hdd * WEEKS_PER_MONTH
        + float(scenario.get("cdd_sensitivity", 0.0)) * float(scenario.get("cdd_sensitivity_scale", 1.0)) * cdd * WEEKS_PER_MONTH
    )
    noise = float(rng.normal(0.0, float(scenario.get("demand_noise_bcf", 0.0)) / np.sqrt(WEEKS_PER_MONTH)))
    monthly_demand = monthly_base + weather_component + noise
    return max(0.0, monthly_demand / WEEKS_PER_MONTH)


def simulate_weekly_forward_scenario(
    name: str,
    scenario: Dict[str, Any],
    monthly_df: pd.DataFrame,
    price_model: Dict[str, Any] | None,
    latest_weekly_storage: Dict[str, Any],
    *,
    horizon_weeks: int = 13,
    n_sims: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    monthly = _ensure_datetime_index(monthly_df)
    storage_capacity = float(scenario.get("storage_capacity", monthly["storage_bcf"].max()))
    pipeline_capacity = float(scenario.get("pipeline_capacity", monthly["production_bcf"].quantile(0.95))) / WEEKS_PER_MONTH
    response_factor = float(scenario.get("storage_response_factor", 0.95))
    release_date = pd.Timestamp(latest_weekly_storage["release_date"])
    initial_storage = float(latest_weekly_storage["storage_bcf"])
    rng = np.random.default_rng(seed)

    rows: list[Dict[str, Any]] = []
    for simulation in range(n_sims):
        storage = initial_storage
        for week in range(1, horizon_weeks + 1):
            forecast_date = release_date + pd.Timedelta(days=7 * week)
            month = int(forecast_date.month)
            hdd, cdd = _sample_weather_for_week(monthly, month=month, rng=rng)
            raw_supply = _weekly_supply(scenario, month, rng)
            delivered_supply = min(raw_supply, pipeline_capacity)
            lng_exports = _weekly_lng_exports(scenario, month)
            domestic_demand = _weekly_domestic_demand(scenario, month, hdd, cdd, rng)
            demand = domestic_demand + lng_exports
            balance = delivered_supply - demand

            max_injection = _seasonal_limit(scenario, month, "injection") / WEEKS_PER_MONTH
            max_withdrawal = _seasonal_limit(scenario, month, "withdrawal") / WEEKS_PER_MONTH
            storage_start = storage
            if balance >= 0:
                storage_change = min(balance, storage_capacity - storage, max_injection)
            else:
                storage_change = -min(-balance, storage, max_withdrawal)
            storage_change *= response_factor
            storage = float(np.clip(storage + storage_change, 0.0, storage_capacity))

            rows.append(
                {
                    "scenario": name,
                    "simulation": simulation,
                    "week": week,
                    "forecast_date": forecast_date,
                    "raw_supply_bcf": round(raw_supply, 2),
                    "delivered_supply_bcf": round(delivered_supply, 2),
                    "demand_bcf": round(demand, 2),
                    "domestic_demand_bcf": round(domestic_demand, 2),
                    "lng_exports_bcf": round(lng_exports, 2),
                    "balance_bcf": round(balance, 2),
                    "storage_start_bcf": round(storage_start, 1),
                    "storage_bcf": round(storage, 1),
                    "storage_change_bcf": round(storage_change, 2),
                    "storage_ratio": round(storage / storage_capacity, 3) if storage_capacity > 0 else np.nan,
                    "hdd": round(hdd, 2),
                    "cdd": round(cdd, 2),
                }
            )

    return pd.DataFrame(rows)


def summarize_weekly_monte_carlo(mc_df: pd.DataFrame) -> pd.DataFrame:
    def q10(x: pd.Series) -> float:
        return float(x.quantile(0.10))

    def q90(x: pd.Series) -> float:
        return float(x.quantile(0.90))

    grouped = (
        mc_df.groupby(["scenario", "week"], as_index=False)
        .agg(
            forecast_date=("forecast_date", "first"),
            expected_storage_bcf=("storage_bcf", "mean"),
            min_storage_bcf=("storage_bcf", "min"),
            max_storage_bcf=("storage_bcf", "max"),
            p10_storage_bcf=("storage_bcf", q10),
            p90_storage_bcf=("storage_bcf", q90),
            expected_storage_change_bcf=("storage_change_bcf", "mean"),
            min_storage_change_bcf=("storage_change_bcf", "min"),
            max_storage_change_bcf=("storage_change_bcf", "max"),
            expected_balance_bcf=("balance_bcf", "mean"),
            min_balance_bcf=("balance_bcf", "min"),
            max_balance_bcf=("balance_bcf", "max"),
            p10_balance_bcf=("balance_bcf", q10),
            p90_balance_bcf=("balance_bcf", q90),
        )
        .sort_values(["scenario", "week"])
        .reset_index(drop=True)
    )

    numeric_cols = grouped.select_dtypes(include=[np.number]).columns
    grouped[numeric_cols] = grouped[numeric_cols].round(2)
    return grouped


def build_weekly_market_signal(weekly_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []

    for scenario, group in weekly_summary.groupby("scenario"):
        g = group.sort_values("week")
        ending = g.iloc[-1]
        total_expected_storage_change = float(g["expected_storage_change_bcf"].sum())
        avg_balance = float(g["expected_balance_bcf"].mean())
        ending_storage = float(ending["expected_storage_bcf"])
        storage_band_width = float(ending["p90_storage_bcf"] - ending["p10_storage_bcf"])
        storage_regime = (
            "Tight"
            if total_expected_storage_change <= -150
            else "Loose"
            if total_expected_storage_change >= 150
            else "Balanced"
        )
        extremeness = (
            "Extreme"
            if abs(total_expected_storage_change) >= 500
            else "Elevated"
            if abs(total_expected_storage_change) >= 150
            else "Expected"
        )
        storage_trend = (
            "Tightening"
            if total_expected_storage_change <= -150
            else "Loosening"
            if total_expected_storage_change >= 150
            else "Balanced"
        )
        confidence = float(
            np.clip(
                1.0 - min(storage_band_width / max(ending_storage, 500.0), 0.7),
                0.3,
                0.95,
            )
        )
        rows.append(
            {
                "scenario": scenario,
                "horizon_weeks": int(g["week"].max()),
                "horizon_months": round(float(g["week"].max()) / WEEKS_PER_MONTH, 1),
                "market_regime": storage_regime,
                "extremeness": extremeness,
                "storage_trend": storage_trend,
                "expected_storage_change_bcf": round(total_expected_storage_change, 1),
                "expected_ending_storage_bcf": round(float(ending["expected_storage_bcf"]), 1),
                "p10_ending_storage_bcf": round(float(ending["p10_storage_bcf"]), 1),
                "p90_ending_storage_bcf": round(float(ending["p90_storage_bcf"]), 1),
                "min_ending_storage_bcf": round(float(ending["min_storage_bcf"]), 1),
                "max_ending_storage_bcf": round(float(ending["max_storage_bcf"]), 1),
                "expected_avg_balance_bcf": round(avg_balance, 1),
                "confidence": round(confidence, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("expected_storage_change_bcf")
