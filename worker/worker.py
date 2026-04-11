import os
import json
import logging
import time
import pandas as pd
import numpy as np
from io import StringIO
from typing import List
import pika
import redis
import psycopg2
from catboost import CatBoostRegressor
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import threading

# Метрики
TASKS_PROCESSED = Counter('worker_tasks_processed_total', 'Total tasks processed', ['status'])
TASK_DURATION = Histogram('worker_task_duration_seconds', 'Task processing duration')
TASKS_IN_PROGRESS = Gauge('worker_tasks_in_progress', 'Tasks currently being processed')

def start_metrics_server():
    try:
        start_http_server(8001)
        print("Prometheus metrics server started on port 8001", flush=True)
        logger.info("Prometheus metrics server started on port 8001")
    except Exception as e:
        print(f"Metrics server failed: {e}", flush=True)
        logger.error(f"Metrics server failed: {e}")

# ------------------------- Конфигурация -------------------------
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@postgres:5432/prices")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/baseline_model.cbm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Загрузка модели -------------------------
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

FEATURES = [
    'avg_price', 'avg_unit_cost',
    'year', 'month', 'quarter', 'day_of_week', 'is_weekend',
    'lag_1', 'lag_7', 'lag_30',
    'rolling_mean_7', 'rolling_mean_30',
    'sales_trend', 'price_change_pct'
]

# ------------------------- Предобработка -------------------------
def load_and_clean_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
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

# ------------------------- Обработка задачи -------------------------
def process_task(body: bytes, task_id: str):
    TASKS_IN_PROGRESS.inc()
    start = time.time()
    logger.info(f"Processing task {task_id}")
    r = redis.from_url(REDIS_URL)
    try:
        raw_df = pd.read_csv(StringIO(body.decode('utf-8')))
        processed_df = full_preprocessing(raw_df)
        processed_df = processed_df.sort_values(['ProductKey', 'DateOnly'])

        results = []
        
        for product_key, group in processed_df.groupby('ProductKey'):
            if len(group) < 2:
                continue
            opt = optimize_price_for_product(group)
            results.append(opt)

        # Сохраняем результат в БД
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                task_id UUID PRIMARY KEY,
                status VARCHAR(20) DEFAULT 'pending',
                result JSONB,
                error TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            INSERT INTO optimization_results (task_id, status, result)
            VALUES (%s, 'completed', %s)
            ON CONFLICT (task_id) DO UPDATE SET
                status = EXCLUDED.status,
                result = EXCLUDED.result,
                updated_at = NOW()
        """, (task_id, json.dumps(results)))
        conn.commit()
        cur.close()
        conn.close()

        # Кешируем в Redis
        r.setex(f"task:{task_id}", 3600, json.dumps({"status": "completed", "result": results}))
        logger.info(f"Task {task_id} completed")
        TASKS_PROCESSED.labels(status='success').inc()

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        error_msg = str(e)

        # Обновляем статус в БД
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO optimization_results (task_id, status, error)
                VALUES (%s, 'failed', %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    error = EXCLUDED.error,
                    updated_at = NOW()
            """, (task_id, error_msg))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            logger.error(f"Failed to update DB with error status: {db_err}")

        # Обновляем статус в Redis
        r.setex(f"task:{task_id}", 3600, json.dumps({"status": "failed", "error": error_msg}))
        TASKS_PROCESSED.labels(status='error').inc()
    finally:
        TASKS_IN_PROGRESS.dec()
        TASK_DURATION.observe(time.time() - start)

# ------------------------- Запуск воркера -------------------------
def main():
    connection = None
    for attempt in range(1, 11):
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            break
        except pika.exceptions.AMQPConnectionError:
            logger.warning(f"RabbitMQ not ready, retrying ({attempt}/10)...")
            time.sleep(2)
    if connection is None:
        logger.error("Could not connect to RabbitMQ after 10 attempts")
        return

    channel = connection.channel()
    channel.queue_declare(queue='tasks', durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        task_id = properties.headers.get("task_id")
        process_task(body, task_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue='tasks', on_message_callback=callback)
    logger.info("Worker started. Waiting for messages...")
    channel.start_consuming()

if __name__ == "__main__":
    threading.Thread(target=start_metrics_server, daemon=True).start()
    main()
