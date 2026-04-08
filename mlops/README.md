# MLOps Stack for Text Classifier

Полноценный MLOps стек для мониторинга текстового классификатора.

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Grafana (3000)                            │
│                    Dashboards & Visualization                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Prometheus   │   │     Loki      │   │   Promtail    │
│    (9090)     │   │    (3100)     │   │   (Docker)    │
│   Metrics     │   │     Logs      │   │  Log Shipper  │
└───────┬───────┘   └───────────────┘   └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         ML Services                                │
├─────────────────────────────┬─────────────────────────────────────┤
│      Classifier (8000)      │     Drift Detector (8001)           │
│   BERT Text Classification  │   Data & Concept Drift Detection    │
└─────────────────────────────┴─────────────────────────────────────┘
```

## Сервисы

### 1. Classifier Service (порт 8000)

FastAPI сервис для текстовой классификации на основе BERT.

**Endpoints:**
- `POST /classify` - классификация одного текста
- `POST /classify/batch` - пакетная классификация
- `GET /health` - проверка здоровья
- `GET /metrics` - Prometheus метрики
- `POST /reload` - перезагрузка модели
- `GET /model/info` - информация о модели

**Метрики:**
- `classifier_requests_total` - количество запросов
- `classifier_request_latency_seconds` - латентность (P50, P95, P99)
- `classifier_predictions_total` - предсказания по классам
- `classifier_confidence_score` - распределение confidence
- `classifier_low_confidence_total` - предсказания с низкой уверенностью
- `classifier_model_loaded` - статус загрузки модели

### 2. Drift Detector Service (порт 8001)

Сервис для обнаружения Data Drift и Concept Drift.

**Endpoints:**
- `POST /sample` - добавить сэмпл для анализа
- `POST /samples` - добавить пакет сэмплов
- `GET /drift` - проверить наличие дрифта
- `GET /distributions` - текущее и референсное распределения
- `POST /reset-reference` - сбросить референсное распределение
- `POST /model-updated` - уведомить об обновлении модели

**Метрики:**
- `drift_data_drift_detected` - флаг обнаружения data drift
- `drift_concept_drift_detected` - флаг обнаружения concept drift
- `drift_psi_score` - Population Stability Index
- `drift_ks_statistic` - Kolmogorov-Smirnov статистика
- `drift_class_distribution_*` - распределение классов
- `drift_model_age_seconds` - возраст модели

### 3. Prometheus (порт 9090)

Сбор и хранение метрик. Настроены алерты для:
- Высокий error rate (>10%)
- Высокая латентность (P95 > 2s)
- Модель не загружена
- Нет предсказаний более 1 часа
- Высокий % низкой уверенности (>30%)
- Аномалия распределения классов

### 4. Loki (порт 3100)

Агрегация логов со всех сервисов.

### 5. Promtail

Сбор логов из Docker контейнеров и отправка в Loki.

### 6. Grafana (порт 3000)

Визуализация метрик и логов.

**Dashboards:**
- **ML Classifier Dashboard** - основные метрики классификатора
- **Drift Detection Dashboard** - метрики дрифта

**Доступ:**
- URL: http://localhost:3000
- Login: admin
- Password: admin

## Быстрый старт

### 1. Запуск всего стека

```bash
docker-compose up -d
```

### 2. Запуск только MLOps компонентов

```bash
docker-compose up -d classifier drift-detector prometheus loki promtail grafana
```

### 3. Проверка статуса

```bash
# Classifier health
curl http://localhost:8000/health

# Drift detector health
curl http://localhost:8001/health

# Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## Использование Classifier API

### Классификация текста

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Build a model to predict house prices using tabular data"}'
```

Ответ:
```json
{
  "label": "CLASSIC ML",
  "confidence": 0.92,
  "latency_ms": 45.2
}
```

### Пакетная классификация

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Image classification for medical diagnosis",
      "Sentiment analysis of tweets"
    ]
  }'
```

## Интеграция с Drift Detector

При каждой классификации рекомендуется отправлять результат в Drift Detector:

```python
import httpx

# После классификации
result = classifier_response

# Отправить в drift detector
httpx.post("http://drift-detector:8001/sample", json={
    "predicted_class": result["label"],
    "confidence": result["confidence"],
    "input_length": len(text)
})
```

## Алерты

Настроены следующие алерты (см. `prometheus/alerts/classifier_alerts.yml`):

| Alert | Severity | Описание |
|-------|----------|----------|
| ClassifierHighErrorRate | critical | Error rate > 10% за 5 мин |
| ClassifierHighLatency | warning | P95 latency > 2s за 5 мин |
| ClassifierModelNotLoaded | critical | Модель не загружена 2+ мин |
| ClassifierNoRecentPredictions | warning | Нет предсказаний 1+ час |
| ClassifierHighLowConfidenceRate | warning | >30% низкая уверенность |
| DataDriftDetected | warning | Обнаружен data drift |
| ConceptDriftDetected | warning | Обнаружен concept drift |

## Drift Detection

### Методы обнаружения

1. **Population Stability Index (PSI)**
   - Сравнивает распределения confidence и длины входа
   - Порог: PSI > 0.2 = drift

2. **Kolmogorov-Smirnov Test**
   - Статистический тест на различие распределений
   - Порог: p-value < 0.05 = drift

3. **Jensen-Shannon Divergence**
   - Для распределения классов
   - Порог: JS > 0.1 = concept drift

### Когда дрифт обнаружен

1. Проверить логи на аномалии во входных данных
2. Сравнить распределения классов (current vs reference)
3. Проанализировать примеры с низкой уверенностью
4. Рассмотреть переобучение модели

### Сброс референса после переобучения

```bash
# После обновления модели
curl -X POST http://localhost:8000/reload
curl -X POST http://localhost:8001/model-updated
curl -X POST http://localhost:8001/reset-reference
```

## Логирование

Все сервисы используют JSON-формат логов:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "classifier",
  "message": "Classification: CLASSIC ML (0.92)",
  "label": "CLASSIC ML",
  "confidence": 0.92,
  "latency_ms": 45.2
}
```

Просмотр логов в Grafana:
1. Перейти на дашборд "ML Classifier Dashboard"
2. Скроллить до панели "Classifier Logs"
3. Или использовать Explore с источником Loki

## Структура файлов

```
mlops/
├── classifier/
│   ├── Dockerfile
│   ├── main.py           # FastAPI сервис
│   └── requirements.txt
├── drift_detector/
│   ├── Dockerfile
│   ├── main.py           # Drift detection сервис
│   └── requirements.txt
├── prometheus/
│   ├── prometheus.yml    # Конфиг Prometheus
│   └── alerts/
│       └── classifier_alerts.yml
├── loki/
│   └── loki-config.yml
├── promtail/
│   └── promtail-config.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasources.yml
│       └── dashboards/
│           ├── dashboards.yml
│           └── json/
│               ├── classifier_dashboard.json
│               └── drift_dashboard.json
└── README.md
```

## Рекомендации по продакшену

1. **Секреты**: Использовать Docker secrets или Vault для паролей
2. **Масштабирование**: Добавить load balancer перед classifier
3. **Бэкапы**: Настроить бэкап Prometheus и Grafana данных
4. **Alertmanager**: Добавить для email/Slack уведомлений
5. **mTLS**: Настроить TLS между сервисами
6. **Resource limits**: Добавить limits в docker-compose

## Полезные PromQL запросы

```promql
# Request rate
sum(rate(classifier_requests_total[5m]))

# Error rate
sum(rate(classifier_requests_total{status="error"}[5m])) /
sum(rate(classifier_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, rate(classifier_request_latency_seconds_bucket[5m]))

# Class distribution
sum(increase(classifier_predictions_total[1h])) by (predicted_class)

# Low confidence ratio
sum(rate(classifier_low_confidence_total[1h])) /
sum(rate(classifier_predictions_total[1h]))
```
