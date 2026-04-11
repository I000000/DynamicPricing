import os
import time
import pandas as pd
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from catboost import CatBoostRegressor
from io import StringIO
from typing import List, Dict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging

# ------------------------- Инициализация -------------------------
app = FastAPI(title="Dynamic Pricing Inference API")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model_tuned.cbm")
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

FEATURES = [
    'avg_price', 'avg_unit_cost',
    'year', 'month', 'quarter', 'day_of_week', 'is_weekend',
    'lag_1', 'lag_7', 'lag_30',
    'rolling_mean_7', 'rolling_mean_30',
    'sales_trend', 'price_change_pct'
]
CAT_FEATURES = ['ProductKey', 'day_of_week', 'month', 'year', 'quarter']

# ------------------------- Метрики Prometheus -------------------------
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    method = request.method
    path = request.url.path
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(method, path).inc()
    REQUEST_DURATION.labels(method, path).observe(duration)
    return response

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------------- Предобработка (из preprocess.py) -------------------------
def load_and_clean_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка сырых транзакций"""
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['UnitCost'] > 0)]
    if 'ExchangeRate' not in df.columns:
        df['ExchangeRate'] = 1.0
    df['UnitPrice_USD'] = df['UnitPrice'] / df['ExchangeRate']
    df['UnitCost_USD'] = df['UnitCost'] / df['ExchangeRate']
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['DateOnly'] = df['OrderDate'].dt.date
    return df

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(['ProductKey', 'DateOnly']).agg(
        total_quantity=('Quantity', 'sum'),
        avg_price=('UnitPrice_USD', 'mean'),
        avg_unit_cost=('UnitCost_USD', 'mean'),
        transaction_count=('OrderKey', 'nunique')
    ).reset_index()
    daily['DateOnly'] = pd.to_datetime(daily['DateOnly'])
    return daily

def fill_missing_dates_for_product(df_daily: pd.DataFrame) -> pd.DataFrame:
    all_products = df_daily['ProductKey'].unique()
    full_dfs = []
    for prod in all_products:
        prod_df = df_daily[df_daily['ProductKey'] == prod].copy().sort_values('DateOnly')
        min_date = prod_df['DateOnly'].min()
        max_date = prod_df['DateOnly'].max()
        full_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        full_df = pd.DataFrame({'DateOnly': full_dates})
        merged = full_df.merge(prod_df, on='DateOnly', how='left')
        merged['ProductKey'] = prod
        merged['total_quantity'] = merged['total_quantity'].fillna(0)
        merged['avg_price'] = merged['avg_price'].ffill().bfill()
        merged['avg_unit_cost'] = merged['avg_unit_cost'].ffill().bfill()
        merged['transaction_count'] = merged['transaction_count'].fillna(0)
        full_dfs.append(merged)
    return pd.concat(full_dfs, ignore_index=True)

def remove_outliers_iqr(df: pd.DataFrame, group_col='ProductKey', value_col='total_quantity', multiplier=3.0) -> pd.DataFrame:
    def filter_group(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return group[(group[value_col] >= lower) & (group[value_col] <= upper)]
    return df.groupby(group_col, group_keys=False).apply(filter_group)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['DateOnly'] = pd.to_datetime(df['DateOnly'])
    df['year'] = df['DateOnly'].dt.year
    df['month'] = df['DateOnly'].dt.month
    df['day_of_week'] = df['DateOnly'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['DateOnly'].dt.quarter
    df = df.sort_values(['ProductKey', 'DateOnly']).reset_index(drop=True)
    df['lag_1'] = df.groupby('ProductKey')['total_quantity'].shift(1)
    df['lag_7'] = df.groupby('ProductKey')['total_quantity'].shift(7)
    df['lag_30'] = df.groupby('ProductKey')['total_quantity'].shift(30)
    df['rolling_mean_7'] = (df.groupby('ProductKey')['total_quantity']
                            .shift(1).rolling(7, min_periods=1).mean()
                            .reset_index(level=0, drop=True))
    df['rolling_mean_30'] = (df.groupby('ProductKey')['total_quantity']
                             .shift(1).rolling(30, min_periods=1).mean()
                             .reset_index(level=0, drop=True))
    df['sales_trend'] = df['rolling_mean_7'] - df['rolling_mean_30']
    df['price_change_pct'] = df.groupby('ProductKey')['avg_price'].pct_change().fillna(0)
    return df

def full_preprocessing(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн предобработки из сырых транзакций до признаков."""
    df_clean = load_and_clean_data_from_df(raw_df)
    df_daily = aggregate_to_daily(df_clean)
    df_filled = fill_missing_dates_for_product(df_daily)
    df_filtered = remove_outliers_iqr(df_filled, multiplier=3.0)
    df_feat = add_features(df_filtered)
    return df_feat

# ------------------------- Оптимизация цены -------------------------
def generate_price_grid(current_price: float, cost: float, n_steps: int = 30, min_margin: float = 0.05) -> List[float]:
    """
    Генерирует сетку цен в пределах ±20% от текущей цены,
    но не ниже cost + min_margin.
    """
    min_price = max(cost + min_margin, current_price * 0.7)
    max_price = current_price * 1.2
    # Если max_price оказалась ниже min_price (например, при высокой себестоимости),
    # расширяем верхнюю границу
    if max_price < min_price:
        max_price = min_price * 1.5
    return np.linspace(min_price, max_price, n_steps).tolist()

def optimize_price_for_product(product_df: pd.DataFrame):
    last_row = product_df.iloc[-1:].copy()
    current_price = last_row['avg_price'].values[0]
    cost = last_row['avg_unit_cost'].values[0]

    X_current = last_row[FEATURES + ['ProductKey']]
    current_demand = model.predict(X_current)[0]
    current_profit = (current_price - cost) * current_demand

    price_candidates = generate_price_grid(current_price, cost)
    best_price = current_price
    best_profit = current_profit
    best_demand = current_demand

    for new_price in price_candidates:
        test_row = last_row.copy()
        test_row['avg_price'] = new_price
        if len(product_df) > 1:
            prev_price = product_df.iloc[-2]['avg_price']
            test_row['price_change_pct'] = (new_price - prev_price) / prev_price if prev_price != 0 else 0.0
        else:
            test_row['price_change_pct'] = 0.0
        X = test_row[FEATURES + ['ProductKey']]
        pred_qty = model.predict(X)[0]
        profit = (new_price - cost) * pred_qty
        if profit > best_profit:
            best_profit = profit
            best_price = new_price
            best_demand = pred_qty

    return {
        "product_key": int(last_row['ProductKey'].values[0]),
        "current_price": float(current_price),
        "unit_cost": float(cost),
        "current_profit": float(current_profit),
        "optimal_price": float(best_price),
        "expected_demand": float(best_demand),
        "expected_profit": float(best_profit),
    }

# ------------------------- Эндпоинты -------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Принимает сырой CSV транзакций, возвращает прогноз спроса на следующий день для каждого товара."""
    try:
        contents = await file.read()
        raw_df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logging.info(f"Получен файл с {len(raw_df)} транзакциями")
        
        # Полный пайплайн предобработки
        processed_df = full_preprocessing(raw_df)
        # Для predict нужны только последние строки каждого товара (текущий день)
        last_day_df = processed_df.sort_values(['ProductKey', 'DateOnly']).groupby('ProductKey').tail(1)
        X = last_day_df[FEATURES + ['ProductKey']]
        preds = model.predict(X)
        last_day_df['predicted_quantity'] = preds
        result = last_day_df[['ProductKey', 'predicted_quantity', 'avg_price', 'avg_unit_cost']].to_dict(orient='records')
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        logging.error(f"Ошибка predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_prices(file: UploadFile = File(...)):
    """Принимает сырой CSV, возвращает оптимальные цены для каждого товара."""
    try:
        contents = await file.read()
        raw_df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logging.info(f"Получен файл с {len(raw_df)} транзакциями")
        
        processed_df = full_preprocessing(raw_df)
        processed_df = processed_df.sort_values(['ProductKey', 'DateOnly'])
        results = []
        for product_key, group in processed_df.groupby('ProductKey'):
            if len(group) < 2:
                continue
            opt = optimize_price_for_product(group)
            results.append(opt)
        return JSONResponse(content={"optimized_prices": results})
    except Exception as e:
        logging.error(f"Ошибка optimize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)