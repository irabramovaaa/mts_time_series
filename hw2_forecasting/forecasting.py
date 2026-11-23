"""
Функции для ДЗ 2: прогнозирование продаж товара по дневному, недельному и месячному ряду.

Подготовка рядов повторяет правила из ДЗ 1 (hw1_eda/ts_analysis.py):
  * продажи = сумма Quantity без отмен и строк с нулевой ценой или количеством;
  * дневной ряд — только рабочие дни магазина (дни без единого счёта не считаются наблюдениями);
  * недельный и месячный ряды — без неполных крайних периодов и недели, когда магазин был закрыт.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

FREQ_NAMES = {"D": "дневной", "W": "недельный", "M": "месячный"}

# --------------------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------------------
def load_online_retail(data_dir: str | Path = "data") -> pd.DataFrame:
    """Читает датасет Online Retail; если файла нет — скачивает его с UCI."""
    data_dir = Path(data_dir)
    xlsx_path = data_dir / "Online Retail.xlsx"
    if not xlsx_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path, _ = urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/352/online+retail.zip", data_dir / "online_retail.zip")
        zipfile.ZipFile(zip_path).extractall(data_dir)

    df = pd.read_excel(xlsx_path, dtype={"InvoiceNo": str, "StockCode": str})
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    return df


def build_series(df: pd.DataFrame, stock_code: str) -> dict[str, pd.Series]:
    """Строит дневной, недельный и месячный ряды продаж товара (в штуках) по правилам из ДЗ 1."""
    sales = df[~df["InvoiceNo"].str.startswith("C") & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    date = sales["InvoiceDate"].dt.normalize()

    # рабочие дни магазина = дни, в которые был хотя бы один счёт по любому товару
    working_days = pd.DatetimeIndex(date.unique()).sort_values()
    is_product = sales["StockCode"] == stock_code
    daily = sales[is_product].groupby(date[is_product])["Quantity"].sum()
    daily = daily.reindex(working_days, fill_value=0).astype(float)

    # для агрегации разворачиваем на полный календарь: нерабочие дни = NaN
    calendar = daily.reindex(pd.date_range(working_days.min(), working_days.max(), freq="D"))
    start, end = calendar.index.min(), calendar.index.max()

    weekly = calendar.resample("W-SUN").agg(["sum", "count"])
    full_week = (weekly.index - pd.Timedelta(days=6) >= start) & (weekly.index <= end)
    weekly = weekly.loc[full_week & (weekly["count"] > 0), "sum"]

    monthly = calendar.resample("MS").sum(min_count=1)
    monthly = monthly[monthly.index + pd.offsets.MonthEnd(0) <= end]

    series = {"D": daily, "W": weekly, "M": monthly}
    for freq, s in series.items():
        s.name = f"{FREQ_NAMES[freq]} ряд, шт."
        s.index.name = "date"
    return series


def check_series(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Проверка индексации: порядок, уникальность дат, пропуски, шаг между наблюдениями."""
    rows = {}
    for freq, s in series.items():
        gaps = s.index.to_series().diff().dropna()
        rows[freq] = {
            "наблюдений": len(s),
            "начало": s.index.min().date(),
            "конец": s.index.max().date(),
            "даты по возрастанию": s.index.is_monotonic_increasing,
            "даты уникальны": s.index.is_unique,
            "пропусков (NaN)": int(s.isna().sum()),
            "типичный шаг": gaps.mode().iat[0],
            "макс. шаг": gaps.max(),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------
# Модели. Каждая функция принимает обучающий ряд и горизонт h и возвращает h прогнозов.
# ETS, SARIMA и бустинг обучаются на log1p(y): в ДЗ 1 разброс рос вместе с уровнем.
# --------------------------------------------------------------------------------------
def forecast_ets(train: pd.Series, h: int, season: int | None = None) -> np.ndarray:
    """Экспоненциальное сглаживание Холта–Уинтерса: затухающий тренд + (опционально) сезонность."""
    model = ExponentialSmoothing(
        np.log1p(train.values),
        trend="add",
        damped_trend=True,
        seasonal="add" if season else None,
        seasonal_periods=season,
        initialization_method="estimated",
    ).fit()
    return np.clip(np.expm1(model.forecast(h)), 0, None)


def forecast_sarima(train: pd.Series, h: int, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)) -> np.ndarray:
    """SARIMA с линейным трендом (trend='ct'): в ДЗ 1 ряд оказался стационарным вокруг тренда."""
    model = SARIMAX(np.log1p(train.values), order=order, seasonal_order=seasonal_order, trend="ct").fit(disp=False)
    return np.clip(np.expm1(model.forecast(h)), 0, None)


def forecast_boosting(train: pd.Series, h: int, lags=(1, 2), season: int | None = None,
                      min_samples_leaf: int = 3) -> np.ndarray:
    """
    Градиентный бустинг на лагах ряда, прогноз строится рекурсивно (шаг за шагом).

    Деревья не умеют продолжать тренд за пределы обучающих значений, поэтому сначала
    из log1p(y) вычитается линейный тренд, бустинг учится на остатках, а тренд
    потом добавляется обратно. Признаки: значения на заданных лагах, среднее за последние
    max(lags) шагов и (для дневного ряда) номер дня внутри недели.
    """
    def features(hist, t):
        x = [hist[-lag] for lag in lags] + [np.mean(hist[-max(lags):])]
        return x + [t % season] if season else x

    z = np.log1p(train.values)
    n = len(z)
    slope, intercept = np.polyfit(np.arange(n), z, 1)
    resid = list(z - (intercept + slope * np.arange(n)))

    X = [features(resid[:t], t) for t in range(max(lags), n)]
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                      min_samples_leaf=min_samples_leaf, subsample=0.8, random_state=0)
    model.fit(X, resid[max(lags):])

    preds = []
    for t in range(n, n + h):
        r = model.predict([features(resid, t)])[0]
        resid.append(r)   # спрогнозированный шаг становится лагом для следующего
        preds.append(r + intercept + slope * t)
    return np.clip(np.expm1(preds), 0, None)


# --------------------------------------------------------------------------------------
# Эксперимент: скользящее начало прогноза (rolling origin, расширяющееся окно)
# --------------------------------------------------------------------------------------
def backtest(y: pd.Series, models: dict, horizons: list[int], n_origins: int,
             step: int = 1, mase_season: int = 1) -> pd.DataFrame:
    """
    Для каждой точки старта модели обучаются только на данных ДО неё и прогнозируют
    max(horizons) шагов вперёд. Точки старта сдвигаются на step наблюдений,
    последняя выбрана так, чтобы весь прогноз попал в данные.

    models — словарь {название: функция (train, h) -> массив из h прогнозов}.
    Возвращает длинную таблицу: точка старта, модель, шаг, дата, факт, прогноз,
    а также масштаб для MASE (MAE наивного прогноза на обучающей части).
    """
    H = max(horizons)
    n = len(y)
    origins = [n - H - step * k for k in reversed(range(n_origins))]
    parts = []
    for o in origins:
        train, test = y.iloc[:o], y.iloc[o:o + H]
        scale = np.mean(np.abs(train.values[mase_season:] - train.values[:-mase_season]))
        for name, forecast in models.items():
            pred = forecast(train, H)
            parts.append(pd.DataFrame({
                "origin": test.index[0], "train_size": o, "model": name,
                "step": np.arange(1, H + 1), "date": test.index,
                "actual": test.values, "forecast": pred, "mase_scale": scale,
            }))
    return pd.concat(parts, ignore_index=True)


def evaluate(forecasts: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Метрики для прогноза на h шагов: ошибки всех шагов 1..h по всем точкам старта.
      MAE  — средняя абсолютная ошибка, в штуках;
      RMSE — сильнее штрафует большие промахи, в штуках;
      WAPE — сумма |ошибок| / сумма продаж, в %; не зависит от масштаба и работает при нулях;
      MASE — MAE модели / MAE наивного прогноза на обучении (< 1 — лучше наивного).
    """
    rows = []
    for h in horizons:
        sub = forecasts[forecasts["step"] <= h].assign(err=lambda d: d["forecast"] - d["actual"])
        for model, g in sub.groupby("model", sort=False):
            per_origin = g.groupby("origin").apply(lambda x: x["err"].abs().mean() / x["mase_scale"].iat[0])
            rows.append({
                "horizon": h, "model": model, "n_forecasts": len(g),
                "MAE": g["err"].abs().mean(),
                "RMSE": np.sqrt((g["err"] ** 2).mean()),
                "WAPE, %": 100 * g["err"].abs().sum() / g["actual"].sum(),
                "MASE": per_origin.mean(),
            })
    return pd.DataFrame(rows).round(2)


def best_models(summary: pd.DataFrame, metric: str = "WAPE, %") -> pd.DataFrame:
    """Лучшая модель для каждой частоты и горизонта по выбранной метрике."""
    idx = summary.groupby(["freq", "horizon"], sort=False)[metric].idxmin()
    return summary.loc[idx, ["freq", "horizon", "model", "MAE", "WAPE, %", "MASE"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------------------
def plot_series(series: dict[str, pd.Series], title: str) -> None:
    """Три ряда, на которых строятся прогнозы."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, (freq, s) in zip(axes, series.items()):
        ax.plot(s.index, s.values, marker="o" if freq != "D" else None, ms=3, lw=0.9, label=s.name)
        ax.set(title=f"{title}: {FREQ_NAMES[freq]} ряд", xlabel="Дата", ylabel="Продажи, шт.")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_split(y: pd.Series, forecasts: pd.DataFrame, title: str) -> None:
    """Где в ряду находятся точки старта прогнозов и проверочная часть."""
    first_origin = forecasts["origin"].min()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(y.index, y.values, lw=0.9, color="tab:blue", label="ряд")
    ax.axvspan(first_origin, y.index.max(), color="tab:orange", alpha=0.15, label="проверочная часть")
    for i, o in enumerate(forecasts["origin"].unique()):
        ax.axvline(o, color="tab:orange", lw=0.8, ls="--", label="точки старта прогноза" if i == 0 else None)
    ax.set(title=title, xlabel="Дата", ylabel="Продажи, шт.")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_last_forecast(y: pd.Series, forecasts: pd.DataFrame, title: str, history: int = 60) -> None:
    """Факт и прогнозы всех моделей из последней точки старта (на максимальный горизонт)."""
    last = forecasts[forecasts["origin"] == forecasts["origin"].max()]
    hist = y[y.index < last["origin"].iat[0]].iloc[-history:]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(hist.index, hist.values, color="gray", lw=1, label="история (обучение)")
    fact = last.drop_duplicates("date")
    ax.plot(fact["date"], fact["actual"], color="black", lw=2, marker="o", ms=3, label="факт")
    for model, g in last.groupby("model", sort=False):
        ax.plot(g["date"], g["forecast"], lw=1.6, marker="o", ms=2.5, label=model)
    ax.set(title=title, xlabel="Дата", ylabel="Продажи, шт.")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_one_step(forecasts: pd.DataFrame, title: str) -> None:
    """Прогнозы на 1 шаг из всех точек старта против факта."""
    one = forecasts[forecasts["step"] == 1]
    fig, ax = plt.subplots(figsize=(14, 4))
    fact = one.drop_duplicates("date")
    ax.plot(fact["date"], fact["actual"], color="black", lw=2, marker="o", label="факт")
    for model, g in one.groupby("model", sort=False):
        ax.plot(g["date"], g["forecast"], lw=1.2, marker="o", ms=3, label=model)
    ax.set(title=title, xlabel="Дата", ylabel="Продажи, шт.")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metric_by_horizon(summary: pd.DataFrame, metric: str, title: str) -> None:
    """Как меняется метрика с ростом горизонта: по одному графику на частоту."""
    freqs = list(summary["freq"].unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(5 * len(freqs), 4))
    for ax, freq in zip(np.atleast_1d(axes), freqs):
        sub = summary[summary["freq"] == freq]
        for model, g in sub.groupby("model", sort=False):
            ax.plot(g["horizon"], g[metric], marker="o", label=model)
        ax.set_xticks(sorted(sub["horizon"].unique()))
        ax.set(title=f"{FREQ_NAMES[freq].capitalize()} ряд", xlabel="Горизонт, шагов", ylabel=metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
