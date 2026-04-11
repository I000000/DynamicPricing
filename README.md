
# Сервис динамического ценообразования
Сервис позволяет загружать исторические транзакции (CSV) и получать рекомендации по оптимальным ценам для максимизации валовой прибыли.
## Возможности
- Автоматическая предобработка и генерация признаков
- Прогноз спроса с помощью CatBoost на CPU/GPU
- Подбор оптимальной цены 
- Асинхронная обработка через RabbitMQ
- Веб-интерфейс на React + TypeScript
- Мониторинг Prometheus + Grafana
## Требования
- Docker 20.10+
- Docker Compose 2.0+
- 8+ ГБ свободной RAM (для GPU-ускорения требуется NVIDIA Docker)
## Установка и запуск
1. Склонируйте репозиторий.
2. Поместите обученную модель `catboost_model_tuned.cbm` в папку `models/`.
3. Выполните в корне проекта:
 ```bash
 docker-compose up --build
 ```
 ## Использование

Откройте в браузере http://localhost

### Через веб-интерфейс
1. Нажмите **«Выберите файл»** и загрузите CSV с транзакциями.  
   Формат файла:  
   `OrderKey,OrderDate,ProductKey,Quantity,UnitPrice,UnitCost,CurrencyCode,ExchangeRate`
2. Нажмите **«Загрузить и оптимизировать»**.
3. Дождитесь статуса `completed` – отобразится таблица с оптимальными ценами и ожидаемой прибылью.

### Через API Gateway (асинхронно)
```bash
curl -X POST http://localhost:8080/upload -F "file=@sample.csv"
# Вернёт task_id
curl http://localhost:8080/task/{task_id}
```
### Через Inference API (синхронно)
```bash
curl -X POST http://localhost:8000/optimize -F "file=@sample.csv"
```
## Мониторинг
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)  
  Дашборд **«Dynamic Pricing Worker Monitoring»** показывает метрики обработки задач.

## Архитектура

```mermaid
graph TD
    subgraph Frontend
        UI[React Frontend]
    end

    subgraph Backend
        API[Go API Gateway]
        DB[(PostgreSQL)]
        Queue[RabbitMQ]
        Redis[(Redis Cache)]
    end

    subgraph ML Service
        Inference[Inference API<br>FastAPI]
        Worker[Python ML Worker]
        Model[CatBoost Model]
    end

    subgraph Monitoring
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    subgraph External
        User[Пользователь]
    end

    User -->|загружает CSV| UI
    UI -->|HTTP POST /api/upload| API
    API -->|сохраняет файл| DB
    API -->|кэширует статус| Redis
    API -->|публикует задачу| Queue
    Queue -->|задача| Worker
    Worker -->|читает файл| DB
    Worker -->|инференс| Model
    Worker -->|сохраняет результат| DB
    Worker -->|кэширует результат| Redis
    API -->|проверяет статус| DB
    API -->|получает отчёт| DB
    API -->|JSON| UI
    UI -->|отображает отчёт| User

    Inference -->|синхронный инференс| Model
    User -->|HTTP POST /optimize| Inference

    Prometheus -->|scrape /metrics| API
    Prometheus -->|scrape /metrics| Worker
    Prometheus -->|scrape /metrics| Inference
    Grafana -->|источник данных| Prometheus
```

Подробная документация по лабораторным работам находится в папке `docs/`.

---