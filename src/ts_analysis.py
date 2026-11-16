"""
Функции для анализа и предобработки временного ряда продаж (датасет Online Retail, UCI).

Модуль покрывает все этапы EDA:
    1. загрузка и очистка транзакций;
    2. построение рядов с дневной, недельной и месячной агрегацией;
    3. диагностика пропусков и выбросов;
    4. описательные и дополнительные характеристики ряда;
    5. декомпозиция (STL) и оценка силы сезонности;
    6. проверка стационарности (ADF + KPSS) и преобразование к стационарному виду.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

FREQ_NAMES = {"D": "день", "W": "неделя", "M": "месяц"}
WEEKDAY_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# --------------------------------------------------------------------------------------
# 1. Загрузка и очистка
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


def describe_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Краткая структура датасета: тип, число пропусков и уникальных значений по полям."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "share_missing": df.isna().mean().round(4),
            "n_unique": df.nunique(),
            "example": df.iloc[0],
        }
    )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Отделяет «нормальные» продажи от служебных строк.

    Удаляются (и учитываются в отчёте):
      * отмены — номер счёта начинается с 'C' (отрицательное Quantity);
      * прочие строки с Quantity <= 0 (корректировки склада, списания);
      * строки с UnitPrice <= 0 (бесплатные отгрузки, списания долгов).
    Совпадающие строки НЕ удаляются: в датасете нет идентификатора строки,
    поэтому их нельзя отличить от повторной позиции в том же счёте.
    Возвращает (очищенный df, отчёт о количестве удалённых строк).
    """
    is_cancel = df["InvoiceNo"].str.startswith("C")
    nonpos_qty = (df["Quantity"] <= 0) & ~is_cancel
    nonpos_price = (df["UnitPrice"] <= 0) & ~is_cancel & ~nonpos_qty

    report = pd.DataFrame(
        {"rows": [is_cancel.sum(), nonpos_qty.sum(), nonpos_price.sum()]},
        index=["Отмены (InvoiceNo = C...)", "Quantity <= 0 (не отмены)", "UnitPrice <= 0"],
    )
    report["share"] = (report["rows"] / len(df)).round(4)

    clean = df[~(is_cancel | nonpos_qty | nonpos_price)].copy()
    clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]
    clean["Date"] = clean["InvoiceDate"].dt.normalize()
    return clean, report


def top_products(df_clean: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Топ товаров по числу дней с продажами — помогает выбрать товар с «плотным» рядом."""
    g = df_clean.groupby("StockCode")
    out = pd.DataFrame(
        {
            "description": g["Description"].agg(lambda s: s.mode().iat[0] if s.notna().any() else ""),
            "days_with_sales": g["Date"].nunique(),
            "n_invoices": g["InvoiceNo"].nunique(),
            "total_qty": g["Quantity"].sum(),
            "total_revenue": g["Revenue"].sum().round(0),
        }
    )
    return out.sort_values("days_with_sales", ascending=False).head(n)


def cancellations_for_product(df_raw: pd.DataFrame, stock_code: str) -> pd.Series:
    """Объём отмен по товару — дополнительная характеристика качества спроса."""
    p = df_raw[df_raw["StockCode"] == stock_code]
    cancel = p[p["InvoiceNo"].str.startswith("C")]
    sold = p[(~p["InvoiceNo"].str.startswith("C")) & (p["Quantity"] > 0)]
    return pd.Series(
        {
            "Продано, шт.": sold["Quantity"].sum(),
            "Отменено, шт.": -cancel["Quantity"].sum(),
            "Доля отмен, %": round(100 * -cancel["Quantity"].sum() / sold["Quantity"].sum(), 2),
            "Число счетов-отмен": cancel["InvoiceNo"].nunique(),
        }
    )


# --------------------------------------------------------------------------------------
# 2. Построение временных рядов
# --------------------------------------------------------------------------------------
def store_working_days(df_clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Дни, когда магазин работал = дни, в которые был хотя бы один счёт по ЛЮБОМУ товару."""
    return pd.DatetimeIndex(sorted(df_clean["Date"].unique()))


def build_daily_series(df_clean: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Дневной ряд по товару на полном календаре.

    Ключевое решение про пропуски:
      * день, когда магазин работал, но товар не купили  -> 0 (реальное отсутствие спроса);
      * день, когда магазин не работал (нет ни одного счёта) -> NaN (наблюдения не существует).
    """
    working = store_working_days(df_clean)
    full_idx = pd.date_range(working.min(), working.max(), freq="D")
    p = df_clean[df_clean["StockCode"] == stock_code]

    daily = p.groupby("Date").agg(
        quantity=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
        n_invoices=("InvoiceNo", "nunique"),
        n_customers=("CustomerID", "nunique"),
    )
    daily = daily.reindex(full_idx)
    daily.index.name = "date"
    daily["store_open"] = daily.index.isin(working)
    sold_cols = ["quantity", "revenue", "n_invoices", "n_customers"]
    daily.loc[daily["store_open"], sold_cols] = daily.loc[daily["store_open"], sold_cols].fillna(0)
    daily["avg_price"] = daily["revenue"] / daily["quantity"].replace(0, np.nan)
    return daily


def aggregate_series(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Агрегация дневного ряда в недельный ('W', неделя Пн–Вс) или месячный ('M').
    Дополнительно считается число рабочих дней в периоде и флаг неполного периода
    (первая/последняя неделя или месяц, попавшие в выборку частично).
    """
    rule = {"W": "W-SUN", "M": "MS"}[freq]
    agg = daily.resample(rule).agg(
        {"quantity": "sum", "revenue": "sum", "n_invoices": "sum", "store_open": "sum"}
    )
    agg = agg.rename(columns={"store_open": "n_work_days"})
    agg["avg_price"] = agg["revenue"] / agg["quantity"].replace(0, np.nan)
    agg["qty_per_work_day"] = agg["quantity"] / agg["n_work_days"].replace(0, np.nan)

    # неполные крайние периоды: календарь периода выходит за границы наблюдений
    start, end = daily.index.min(), daily.index.max()
    if freq == "W":
        period_start = agg.index - pd.Timedelta(days=6)
        period_end = agg.index
    else:
        period_start = agg.index
        period_end = agg.index + pd.offsets.MonthEnd(0)
    agg["is_partial"] = (period_start < start) | (period_end > end)
    agg.index.name = "period"
    return agg


def build_all_series(df_clean: pd.DataFrame, stock_code: str) -> dict[str, pd.DataFrame]:
    """Строит все три агрегации сразу: {'D': ..., 'W': ..., 'M': ...}."""
    daily = build_daily_series(df_clean, stock_code)
    return {"D": daily, "W": aggregate_series(daily, "W"), "M": aggregate_series(daily, "M")}


def analysis_series(frames: dict[str, pd.DataFrame], freq: str, col: str = "quantity") -> pd.Series:
    """
    Ряд, используемый для дальнейшего анализа:
      * D — только рабочие дни магазина (нерабочие дни не являются наблюдениями);
      * W, M — без неполных крайних периодов (иначе они выглядят как ложные провалы)
        и без периодов, когда магазин не работал ни дня (рождественская неделя).
    """
    f = frames[freq]
    if freq == "D":
        s = f.loc[f["store_open"], col]
    else:
        s = f.loc[~f["is_partial"] & (f["n_work_days"] > 0), col]
    s.name = f"{col} ({FREQ_NAMES[freq]})"
    return s.astype(float)


# --------------------------------------------------------------------------------------
# 3. Пропуски и выбросы
# --------------------------------------------------------------------------------------
def missing_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Сводка по пропускам/нулям для всех частот."""
    rows = {}
    d = frames["D"]
    rows["D"] = {
        "всего периодов": len(d),
        "магазин закрыт (NaN)": int((~d["store_open"]).sum()),
        "рабочих периодов": int(d["store_open"].sum()),
        "рабочих периодов без продаж (0)": int(((d["quantity"] == 0) & d["store_open"]).sum()),
        "неполных периодов": 0,
    }
    for fr in ["W", "M"]:
        f = frames[fr]
        rows[fr] = {
            "всего периодов": len(f),
            "магазин закрыт (NaN)": int((f["n_work_days"] == 0).sum()),
            "рабочих периодов": int((f["n_work_days"] > 0).sum()),
            "рабочих периодов без продаж (0)": int(((f["quantity"] == 0) & (f["n_work_days"] > 0)).sum()),
            "неполных периодов": int(f["is_partial"].sum()),
        }
    return pd.DataFrame(rows).T


def closed_days_breakdown(daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Где расположены нерабочие дни: распределение по дням недели и непрерывные блоки."""
    closed = daily.index[~daily["store_open"]]
    by_weekday = (
        pd.Series(closed.dayofweek).map(dict(enumerate(WEEKDAY_NAMES))).value_counts()
        .reindex(WEEKDAY_NAMES, fill_value=0)
    )
    # непрерывные блоки закрытых дней длиной > 1, не являющиеся обычной субботой
    is_closed = ~daily["store_open"]
    block_id = (is_closed != is_closed.shift()).cumsum()
    blocks = (
        daily[is_closed].groupby(block_id[is_closed])
        .apply(lambda g: pd.Series({"start": g.index.min(), "end": g.index.max(), "days": len(g)}))
    )
    blocks = blocks[(blocks["days"] > 1) | (pd.DatetimeIndex(blocks["start"]).dayofweek != 5)]
    return by_weekday, blocks.reset_index(drop=True)


def detect_outliers(s: pd.Series, window: int = 15, k_iqr: float = 1.5, k_mad: float = 3.5) -> pd.DataFrame:
    """
    Два метода выявления выбросов:
      * IQR (Тьюки) — глобальный: x > Q3 + k·IQR или x < Q1 − k·IQR;
      * Hampel (скользящая медиана ± k·MAD) — локальный, устойчив к тренду.
    Наблюдение считается выбросом, если оба метода согласны (consensus).
    """
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_flag = (s > q3 + k_iqr * iqr) | (s < q1 - k_iqr * iqr)

    med = s.rolling(window, center=True, min_periods=window // 2).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 2).median()
    robust_z = 0.6745 * (s - med) / mad.replace(0, np.nan)
    hampel_flag = robust_z.abs() > k_mad

    return pd.DataFrame(
        {
            "value": s,
            "rolling_median": med,
            "robust_z": robust_z.round(2),
            "iqr_outlier": iqr_flag,
            "hampel_outlier": hampel_flag.fillna(False),
            "outlier": iqr_flag & hampel_flag.fillna(False),
        }
    )


def outlier_summary(outl: pd.DataFrame) -> pd.Series:
    """Количество выбросов по методам."""
    return pd.Series(
        {
            "наблюдений": len(outl),
            "IQR": int(outl["iqr_outlier"].sum()),
            "Hampel": int(outl["hampel_outlier"].sum()),
            "оба метода": int(outl["outlier"].sum()),
            "доля (оба), %": round(100 * outl["outlier"].mean(), 2),
        }
    )


def explain_outliers(outl: pd.DataFrame, df_clean: pd.DataFrame, stock_code: str, top: int = 10) -> pd.DataFrame:
    """Для дневных выбросов показывает, из чего они состоят: крупнейший заказ и его доля в дне."""
    days = outl.index[outl["outlier"]]
    p = df_clean[(df_clean["StockCode"] == stock_code) & (df_clean["Date"].isin(days))]
    g = p.groupby("Date")
    res = pd.DataFrame(
        {
            "qty_day": g["Quantity"].sum(),
            "n_invoices": g["InvoiceNo"].nunique(),
            "max_single_order": g["Quantity"].max(),
            "country_of_max": g.apply(lambda x: x.loc[x["Quantity"].idxmax(), "Country"]),
        }
    )
    res["share_of_max_order"] = (res["max_single_order"] / res["qty_day"]).round(2)
    return res.sort_values("qty_day", ascending=False).head(top)


def winsorize_outliers(outl: pd.DataFrame) -> pd.Series:
    """Альтернативная (не применяемая по умолчанию) обработка: замена выброса скользящей медианой."""
    s = outl["value"].copy()
    s[outl["outlier"]] = outl.loc[outl["outlier"], "rolling_median"]
    return s


# --------------------------------------------------------------------------------------
# 4. Описательные и дополнительные характеристики
# --------------------------------------------------------------------------------------
def descriptive_stats(s: pd.Series) -> pd.Series:
    """Описательные статистики + характеристики спроса (ADI, CV² по Syntetos–Boylan)."""
    s = s.dropna()
    nz = s[s > 0]
    adi = len(s) / max(len(nz), 1)  # средний интервал между ненулевыми продажами
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    if adi < 1.32 and cv2 < 0.49:
        demand = "smooth (регулярный)"
    elif adi < 1.32:
        demand = "erratic (нерегулярный по объёму)"
    elif cv2 < 0.49:
        demand = "intermittent (прерывистый)"
    else:
        demand = "lumpy (комковатый)"
    return pd.Series(
        {
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "Q1": s.quantile(0.25),
            "Q3": s.quantile(0.75),
            "max": s.max(),
            "CV (std/mean)": s.std() / s.mean(),
            "skewness": stats.skew(s),
            "kurtosis (excess)": stats.kurtosis(s),
            "доля нулей": (s == 0).mean(),
            "ADI": adi,
            "CV² (ненулевые)": cv2,
            "тип спроса": demand,
            "ACF lag 1": acf(s, nlags=1, fft=True)[1],
        }
    )


def stats_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Описательные статистики для нескольких рядов сразу (столбец = частота)."""
    t = pd.DataFrame({k: descriptive_stats(v) for k, v in series.items()})
    return t.map(lambda x: round(x, 3) if isinstance(x, (float, np.floating)) else x)


def level_variance_by_period(s: pd.Series, rule: str = "QS") -> tuple[pd.DataFrame, pd.Series]:
    """
    Изменение уровня и разброса по кварталам + формальные тесты:
      * Kruskal–Wallis — различаются ли уровни (медианы) между кварталами;
      * Levene (по медиане, Brown–Forsythe) — различаются ли дисперсии.
    """
    groups = {k: g.dropna().values for k, g in s.groupby(pd.Grouper(freq=rule)) if len(g.dropna()) > 2}
    table = pd.DataFrame(
        {
            "n": {k.date(): len(v) for k, v in groups.items()},
            "mean": {k.date(): v.mean() for k, v in groups.items()},
            "median": {k.date(): np.median(v) for k, v in groups.items()},
            "std": {k.date(): v.std(ddof=1) for k, v in groups.items()},
        }
    ).round(1)
    table["CV"] = (table["std"] / table["mean"]).round(2)
    kw = stats.kruskal(*groups.values())
    lev = stats.levene(*groups.values(), center="median")
    tests = pd.Series(
        {
            "Kruskal–Wallis p-value (уровень)": round(kw.pvalue, 5),
            "Levene p-value (дисперсия)": round(lev.pvalue, 5),
            "corr(mean, std) по периодам": round(np.corrcoef(table["mean"], table["std"])[0, 1], 3),
        }
    )
    return table, tests


def weekday_profile(daily_s: pd.Series) -> tuple[pd.DataFrame, float]:
    """Профиль продаж по дням недели (рабочие дни) + тест Kruskal–Wallis на различие дней."""
    df = daily_s.dropna().to_frame("value")
    df["weekday"] = df.index.dayofweek
    prof = df.groupby("weekday")["value"].agg(["count", "mean", "median", "std"]).round(1)
    prof.index = [WEEKDAY_NAMES[i] for i in prof.index]
    groups = [g.values for _, g in df.groupby("weekday")["value"]]
    pvalue = stats.kruskal(*groups).pvalue
    return prof, pvalue


def price_quantity_relation(frames: dict[str, pd.DataFrame], freq: str = "W") -> pd.Series:
    """Связь средней цены и объёма продаж (ранговая корреляция Спирмена)."""
    f = frames[freq]
    f = f[~f["is_partial"]] if "is_partial" in f else f
    f = f.dropna(subset=["avg_price", "quantity"])
    rho, p = stats.spearmanr(f["avg_price"], f["quantity"])
    return pd.Series({"Spearman rho": round(rho, 3), "p-value": round(p, 5),
                      "мин. ср. цена": round(f["avg_price"].min(), 2), "макс. ср. цена": round(f["avg_price"].max(), 2)})


def country_structure(df_clean: pd.DataFrame, stock_code: str, top: int = 5) -> pd.DataFrame:
    """Структура продаж товара по странам."""
    p = df_clean[df_clean["StockCode"] == stock_code]
    t = p.groupby("Country")["Quantity"].sum().sort_values(ascending=False)
    out = pd.DataFrame({"quantity": t, "share, %": (100 * t / t.sum()).round(1)})
    return out.head(top)


def order_size_profile(df_clean: pd.DataFrame, stock_code: str) -> pd.Series:
    """Распределение размера одной строки заказа: показывает оптовую природу продаж."""
    q = df_clean.loc[df_clean["StockCode"] == stock_code, "Quantity"]
    return q.describe(percentiles=[0.5, 0.9, 0.99]).round(1)


# --------------------------------------------------------------------------------------
# 5. Декомпозиция и сезонность
# --------------------------------------------------------------------------------------
def stl_decompose(s: pd.Series, period: int, robust: bool = True):
    """STL-декомпозиция (устойчивая к выбросам) — возвращает объект результата statsmodels."""
    s = s.dropna()
    return STL(s.values, period=period, robust=robust).fit(), s.index


def decomposition_strength(res) -> pd.Series:
    """
    Сила тренда и сезонности по Hyndman & Athanasopoulos (FPP3):
        F_T = max(0, 1 − Var(R) / Var(T + R)),  F_S = max(0, 1 − Var(R) / Var(S + R)).
    F_S > ~0.64 обычно трактуют как выраженную сезонность, < ~0.4 — как слабую.
    """
    r, t, sn = res.resid, res.trend, res.seasonal
    ft = max(0.0, 1 - np.var(r) / np.var(t + r))
    fs = max(0.0, 1 - np.var(r) / np.var(sn + r))
    return pd.Series({"сила тренда F_T": round(ft, 3), "сила сезонности F_S": round(fs, 3)})


def seasonal_acf_check(s: pd.Series, lags: list[int], nlags: int | None = None) -> pd.DataFrame:
    """Значения ACF на предполагаемых сезонных лагах и 95%-граница значимости ±1.96/√n."""
    s = s.dropna()
    nlags = nlags or max(lags)
    values = acf(s, nlags=nlags, fft=True)
    bound = 1.96 / np.sqrt(len(s))
    return pd.DataFrame(
        {"lag": lags, "acf": [round(values[l], 3) for l in lags], "граница 95%": round(bound, 3),
         "значимо": [abs(values[l]) > bound for l in lags]}
    ).set_index("lag")


# --------------------------------------------------------------------------------------
# 6. Стационарность
# --------------------------------------------------------------------------------------
def stationarity_tests(s: pd.Series, name: str = "") -> pd.Series:
    """
    ADF       : H0 — есть единичный корень (ряд НЕстационарен). p < 0.05 -> отвергаем H0.
    KPSS (c)  : H0 — ряд стационарен вокруг константы.        p < 0.05 -> отвергаем H0.
    KPSS (ct) : H0 — ряд стационарен вокруг линейного тренда (справочно).
    Ряд считаем стационарным, если ADF отвергает H0, а KPSS (c) — нет.
    Если ADF отвергает единичный корень, KPSS(c) отвергает стационарность вокруг константы,
    но KPSS(ct) её не отвергает — ряд тренд-стационарен (стационарен вокруг тренда).
    Значения p-value KPSS ограничены таблицей: 0.01 означает «≤ 0.01», 0.1 — «≥ 0.1».
    """
    import warnings

    s = s.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS предупреждает, если p вне таблицы (0.01 / 0.1)
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        _, kpss_ct_p, *_ = kpss(s, regression="ct", nlags="auto")
    adf_ok, kpss_ok, kpss_ct_ok = adf_p < 0.05, kpss_p >= 0.05, kpss_ct_p >= 0.05
    if adf_ok and kpss_ok:
        verdict = "стационарен"
    elif adf_ok and kpss_ct_ok:
        verdict = "тренд-стационарен (нестационарен по уровню)"
    elif not adf_ok and not kpss_ok:
        verdict = "нестационарен"
    else:
        verdict = "тесты расходятся"
    return pd.Series(
        {"n": len(s), "ADF stat": round(adf_stat, 3), "ADF p-value": round(adf_p, 4),
         "KPSS(c) stat": round(kpss_stat, 3), "KPSS(c) p-value": round(kpss_p, 4),
         "KPSS(ct) p-value": round(kpss_ct_p, 4), "ACF lag 1": round(acf(s, nlags=1, fft=True)[1], 3),
         "вывод": verdict},
        name=name,
    )


def trend_test(s: pd.Series) -> pd.Series:
    """
    Проверка наличия монотонного тренда:
      * тест Манна–Кендалла (τ Кендалла между временем и значением), H0 — тренда нет;
      * наклон OLS-регрессии значения на номер периода и его p-value.
    """
    s = s.dropna()
    t = np.arange(len(s))
    tau, p_mk = stats.kendalltau(t, s.values)
    lr = stats.linregress(t, s.values)
    return pd.Series(
        {"n": len(s), "Kendall tau": round(tau, 3), "Mann–Kendall p-value": round(p_mk, 5),
         "наклон OLS за период": round(lr.slope, 2), "OLS p-value": round(lr.pvalue, 5),
         "рост за период, % от среднего": round(100 * lr.slope / s.mean(), 2)},
        name=s.name,
    )


def transform_series(s: pd.Series, log: bool = True, detrend: bool = False,
                     diff: int = 0, seasonal_diff: int = 0) -> pd.Series:
    """
    Последовательно применяет: log1p -> удаление линейного тренда (остатки OLS по номеру
    наблюдения) -> сезонное дифференцирование -> обычное дифференцирование.
    """
    out = s.dropna().astype(float)
    if log:
        out = np.log1p(out)
    if detrend:
        t = np.arange(len(out))
        lr = stats.linregress(t, out.values)
        out = out - (lr.intercept + lr.slope * t)
    if seasonal_diff:
        out = out.diff(seasonal_diff)
    for _ in range(diff):
        out = out.diff()
    return out.dropna()


def stationarity_pipeline(s: pd.Series, candidates: dict[str, dict]) -> pd.DataFrame:
    """Проверяет исходный ряд и набор преобразований, возвращает сводную таблицу."""
    rows = [stationarity_tests(s, "исходный")]
    for name, kw in candidates.items():
        rows.append(stationarity_tests(transform_series(s, **kw), name))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Визуализация
# --------------------------------------------------------------------------------------
def plot_raw_aggregations(frames: dict[str, pd.DataFrame], title: str) -> None:
    """Ряд до обработки на трёх частотах; нерабочие дни и неполные периоды подсвечены."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    d = frames["D"]
    ax = axes[0]
    ax.plot(d.index, d["quantity"], lw=0.8, color="tab:blue", label="продажи, шт. (рабочие дни)")
    closed = d.index[~d["store_open"]]
    ax.scatter(closed, np.zeros(len(closed)), marker="|", color="tab:red", s=40, label="магазин закрыт (NaN)")
    ax.set_title(f"{title}: дневная агрегация")
    for ax, fr in zip(axes[1:], ["W", "M"]):
        f = frames[fr]
        ax.plot(f.index, f["quantity"], marker="o", ms=3, color="tab:blue", label="продажи, шт.")
        part = f[f["is_partial"]]
        ax.scatter(part.index, part["quantity"], color="tab:orange", s=70, zorder=3, label="неполный период")
        ax.set_title(f"{title}: {'недельная' if fr == 'W' else 'месячная'} агрегация")
    for ax in axes:
        ax.set_xlabel("Дата")
        ax.set_ylabel("Количество, шт.")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_outliers(outl: pd.DataFrame, title: str) -> None:
    """Ряд, скользящая медиана и найденные выбросы (по каждому методу)."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(outl.index, outl["value"], lw=0.8, label="ряд")
    ax.plot(outl.index, outl["rolling_median"], lw=1.5, color="black", label="скользящая медиана")
    only_one = outl[(outl["iqr_outlier"] | outl["hampel_outlier"]) & ~outl["outlier"]]
    both = outl[outl["outlier"]]
    ax.scatter(only_one.index, only_one["value"], color="gold", edgecolor="k", s=35, zorder=3,
               label="выброс по одному методу")
    ax.scatter(both.index, both["value"], color="red", s=55, zorder=4, label="выброс по обоим методам")
    ax.set(title=title, xlabel="Дата", ylabel="Количество, шт.")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_distribution(series: dict[str, pd.Series], title: str) -> None:
    """Гистограммы и boxplot для каждой частоты."""
    fig, axes = plt.subplots(2, len(series), figsize=(5 * len(series), 7))
    for j, (name, s) in enumerate(series.items()):
        axes[0, j].hist(s.dropna(), bins=30, color="tab:blue", alpha=0.8, label=name)
        axes[0, j].axvline(s.mean(), color="red", ls="--", label="среднее")
        axes[0, j].axvline(s.median(), color="green", ls="--", label="медиана")
        axes[0, j].set(title=f"Гистограмма: {name}", xlabel="Количество, шт.", ylabel="Частота")
        axes[0, j].legend()
        axes[1, j].boxplot(s.dropna(), vert=False)
        axes[1, j].set(title=f"Boxplot: {name}", xlabel="Количество, шт.", yticks=[])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_rolling(s: pd.Series, window: int, title: str, unit: str) -> None:
    """Скользящие среднее и стандартное отклонение — изменение уровня и разброса во времени."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(s.index, s, lw=0.7, alpha=0.6, label="ряд")
    ax.plot(s.index, s.rolling(window).mean(), lw=2, color="tab:red", label=f"скользящее среднее ({window} {unit})")
    ax.plot(s.index, s.rolling(window).std(), lw=2, color="tab:green", label=f"скользящее std ({window} {unit})")
    ax.set(title=title, xlabel="Дата", ylabel="Количество, шт.")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weekday_profile(daily_s: pd.Series, title: str) -> None:
    """Boxplot продаж по дням недели."""
    df = daily_s.dropna().to_frame("value")
    df["weekday"] = df.index.dayofweek
    present = sorted(df["weekday"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([df.loc[df["weekday"] == d, "value"] for d in present], showfliers=False)
    ax.plot(range(1, len(present) + 1), [df.loc[df["weekday"] == d, "value"].mean() for d in present],
            "o--", color="red", label="среднее")
    ax.set_xticks(range(1, len(present) + 1), [WEEKDAY_NAMES[d] for d in present])
    ax.set(title=title, xlabel="День недели", ylabel="Количество, шт.")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_extra_characteristics(frames: dict[str, pd.DataFrame], title: str) -> None:
    """Недельные: число счетов, средняя цена и продажи на рабочий день."""
    w = frames["W"][~frames["W"]["is_partial"]]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(w.index, w["n_invoices"], marker="o", ms=3, label="число счетов с товаром")
    axes[0].set_ylabel("Счетов, шт.")
    axes[1].plot(w.index, w["avg_price"], marker="o", ms=3, color="tab:purple", label="средняя цена")
    axes[1].set_ylabel("Цена, £")
    axes[2].plot(w.index, w["qty_per_work_day"], marker="o", ms=3, color="tab:green",
                 label="продажи на рабочий день")
    axes[2].set_ylabel("Шт. / раб. день")
    axes[2].set_xlabel("Неделя")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(s: pd.Series, lags: int, title: str) -> None:
    """ACF и PACF рядом."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(s.dropna(), lags=lags, ax=axes[0], title=f"ACF: {title}")
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], title=f"PACF: {title}", method="ywm")
    for ax in axes:
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stl(res, index, title: str) -> None:
    """Графики компонент STL: наблюдения, тренд, сезонность, остаток."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    parts = [("Наблюдения", res.observed, "tab:blue"), ("Тренд", res.trend, "tab:red"),
             ("Сезонная компонента", res.seasonal, "tab:green"), ("Остаток", res.resid, "tab:gray")]
    for ax, (name, values, color) in zip(axes, parts):
        if name == "Остаток":
            ax.scatter(index, values, s=6, color=color, label=name)
            ax.axhline(0, color="black", lw=0.8)
        else:
            ax.plot(index, values, color=color, label=name)
        ax.set_ylabel(name)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Дата")
    plt.tight_layout()
    plt.show()


def plot_transformed(s: pd.Series, transformed: pd.Series, title: str) -> None:
    """Исходный и преобразованный ряд рядом."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(s.index, s, lw=0.8)
    axes[0].set(title=f"Исходный: {title}", xlabel="Дата", ylabel="Количество, шт.")
    axes[1].plot(transformed.index, transformed, lw=0.8, color="tab:green")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title=f"Преобразованный: {transformed.name or title}", xlabel="Дата", ylabel="Значение")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
