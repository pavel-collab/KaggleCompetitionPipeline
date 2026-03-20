# Kaggle Competitions Pipeline

Автоматический мониторинг Kaggle соревнований с уведомлениями в Telegram.

## Что делает пайплайн

1. **Собирает** новые соревнования с Kaggle API
2. **Классифицирует** их через LLM (CLASSIC ML / LLM-NLP / CV / Other)
3. **Фильтрует** только интересные категории
4. **Сохраняет** в PostgreSQL
5. **Переводит** описание на русский
6. **Отправляет** уведомление в Telegram

---

## Архитектура

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Scheduler  │────▶│ classify_queue   │────▶│ Classify Worker  │
│  (cron)     │     │    (RabbitMQ)    │     │   (LLM + DB)     │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Telegram   │◀────│  Notify Worker   │◀────│   PostgreSQL     │
│             │     │  (LLM + TG Bot)  │     │                  │
└─────────────┘     └──────────────────┘     └──────────────────┘
                            ▲
                            │
                    ┌───────┴────────┐
                    │  notify_queue  │
                    │   (RabbitMQ)   │
                    └────────────────┘
```

### Сервисы

| Сервис | Описание |
|--------|----------|
| `postgres` | База данных для хранения соревнований |
| `rabbitmq` | Брокер сообщений для очередей задач |
| `scheduler` | Запускается каждые 24ч, создаёт задачи |
| `classify-worker` | Обрабатывает очередь классификации |
| `notify-worker` | Обрабатывает очередь уведомлений |

---

## Требования

- Docker и Docker Compose
- Telegram Bot Token (получить у [@BotFather](https://t.me/BotFather))
- OpenRouter API Key (или OpenAI API Key)
- Kaggle API сервис на порту 8000 (опционально)

---

## Быстрый старт

### 1. Клонирование и переход в директорию

```bash
cd kaggle_pipeline
```

### 2. Создание конфигурации

```bash
cp .env.example .env
```

### 3. Заполнение .env файла

Откройте `.env` в редакторе и заполните обязательные поля:

```bash
# Обязательные параметры:
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=448541908
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxx
```

#### Как получить TELEGRAM_BOT_TOKEN

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям, задайте имя бота
4. Скопируйте токен вида `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

#### Как получить TELEGRAM_CHAT_ID

1. Напишите что-нибудь вашему боту
2. Откройте в браузере: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Найдите `"chat":{"id":123456789}` — это ваш chat_id

#### Как получить OPENAI_API_KEY (OpenRouter)

1. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai)
2. Перейдите в Keys → Create Key
3. Скопируйте ключ

### 4. Запуск

```bash
docker-compose up --build
```

Для запуска в фоне:

```bash
docker-compose up --build -d
```

### 5. Проверка статуса

```bash
# Логи всех сервисов
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f scheduler
docker-compose logs -f classify-worker
docker-compose logs -f notify-worker
```

---

## Управление

### Остановка

```bash
docker-compose down
```

### Остановка с удалением данных

```bash
docker-compose down -v
```

### Перезапуск одного сервиса

```bash
docker-compose restart notify-worker
```

### Пересборка после изменений кода

```bash
docker-compose up --build -d
```

---

## Мониторинг

### RabbitMQ Management UI

Веб-интерфейс для просмотра очередей:

- URL: http://localhost:15672
- Login: `guest`
- Password: `guest`

Здесь можно увидеть:
- Количество сообщений в очередях
- Скорость обработки
- Статус consumers (workers)

### PostgreSQL

Подключение к базе данных:

```bash
# Через docker
docker-compose exec postgres psql -U kaggle -d kaggle

# Или напрямую (если установлен psql)
psql -h localhost -p 5432 -U kaggle -d kaggle
```

Полезные запросы:

```sql
-- Все соревнования
SELECT * FROM kaggle_competitions;

-- Только новые
SELECT title, type, status FROM kaggle_competitions WHERE status = 'new';

-- Статистика по типам
SELECT type, COUNT(*) FROM kaggle_competitions GROUP BY type;

-- Статистика по статусам
SELECT status, COUNT(*) FROM kaggle_competitions GROUP BY status;
```

---

## Конфигурация

### Все параметры .env

| Параметр | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `TELEGRAM_BOT_TOKEN` | Да | - | Токен Telegram бота |
| `TELEGRAM_CHAT_ID` | Да | - | ID чата для уведомлений |
| `OPENAI_API_KEY` | Да | - | API ключ OpenRouter/OpenAI |
| `OPENAI_API_BASE` | Нет | `https://openrouter.ai/api/v1` | Base URL для API |
| `OPENAI_MODEL` | Нет | `openai/gpt-4o-mini` | Модель для классификации |
| `POSTGRES_HOST` | Нет | `postgres` | Хост PostgreSQL |
| `POSTGRES_PORT` | Нет | `5432` | Порт PostgreSQL |
| `POSTGRES_USER` | Нет | `kaggle` | Пользователь БД |
| `POSTGRES_PASSWORD` | Нет | `kaggle` | Пароль БД |
| `POSTGRES_DB` | Нет | `kaggle` | Имя базы данных |
| `RABBITMQ_HOST` | Нет | `rabbitmq` | Хост RabbitMQ |
| `RABBITMQ_PORT` | Нет | `5672` | Порт RabbitMQ |
| `RABBITMQ_USER` | Нет | `guest` | Пользователь RabbitMQ |
| `RABBITMQ_PASSWORD` | Нет | `guest` | Пароль RabbitMQ |
| `KAGGLE_API_URL` | Нет | `http://host.docker.internal:8000` | URL Kaggle API сервиса |

### Изменение интервала запуска

В файле `app/main.py` измените константу:

```python
# Каждые 24 часа (по умолчанию)
SCHEDULER_INTERVAL = 24 * 60 * 60

# Каждый час
SCHEDULER_INTERVAL = 60 * 60

# Каждые 5 минут (для тестирования)
SCHEDULER_INTERVAL = 5 * 60
```

После изменения пересоберите:

```bash
docker-compose up --build -d
```

---

## Разработка

### Локальный запуск без Docker

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Запустить PostgreSQL и RabbitMQ (можно через Docker)
docker-compose up -d postgres rabbitmq

# Настроить .env для локального запуска
# POSTGRES_HOST=localhost
# RABBITMQ_HOST=localhost

# Инициализировать БД
python -m app.main init-db

# Запустить scheduler
python -m app.main scheduler

# В другом терминале - classify worker
python -m app.main classify

# В третьем терминале - notify worker
python -m app.main notify
```

### Структура проекта

```
kaggle_pipeline/
├── app/
│   ├── __init__.py
│   ├── config.py      # Настройки (pydantic-settings)
│   ├── models.py      # SQLAlchemy модели
│   ├── database.py    # CRUD операции
│   ├── llm.py         # LangChain + классификация
│   ├── telegram.py    # Telegram Bot API
│   ├── worker.py      # RabbitMQ producers/consumers
│   └── main.py        # Entry point
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Ошибка подключения к RabbitMQ

```
pika.exceptions.AMQPConnectionError: Connection refused
```

**Решение:** RabbitMQ ещё не запустился. Подождите 10-15 секунд или проверьте:

```bash
docker-compose ps
docker-compose logs rabbitmq
```

### Ошибка Telegram: Chat not found

```
telegram.error.BadRequest: Chat not found
```

**Решение:** Убедитесь, что вы написали боту хотя бы одно сообщение перед запуском.

### Ошибка OpenRouter: Invalid API Key

```
openai.AuthenticationError: Invalid API Key
```

**Решение:** Проверьте правильность `OPENAI_API_KEY` в `.env`

### База данных пустая после перезапуска

По умолчанию данные сохраняются в Docker volumes. Если вы запускали `docker-compose down -v`, данные были удалены.

---

## Логика работы (подробно)

### Scheduler (каждые 24 часа)

```
1. Подсчитать соревнования со статусом 'new' или 'queued'

2. Если >= 5 в очереди:
   → Взять 3 из БД
   → Отправить в notify_queue

3. Если < 5 в очереди:
   → Проверить health Kaggle API
   → Если недоступен → отправить ошибку в Telegram
   → Если доступен → загрузить соревнования
   → Отправить каждое в classify_queue
```

### Classify Worker

```
1. Получить задачу из classify_queue

2. Отправить в LLM для классификации:
   - Входные данные: title, description, tags
   - Результат: тип (CLASSIC ML / LLM-NLP / CV / Other)

3. Если тип интересный (не Other):
   → Сохранить в БД со статусом 'new'

4. Подтвердить обработку задачи (ack)
```

### Notify Worker

```
1. Получить задачу из notify_queue

2. Перевести описание на русский через LLM

3. Отправить сообщение в Telegram:
   - Title, Link, Deadline, Type
   - Описание на русском

4. Обновить статус в БД на 'shown'

5. Подтвердить обработку задачи (ack)
```

---

## Масштабирование

### Запуск нескольких workers

```bash
docker-compose up --build -d --scale classify-worker=3 --scale notify-worker=2
```

Это запустит:
- 3 classify-worker (параллельная классификация)
- 2 notify-worker (параллельная отправка)

RabbitMQ автоматически распределит задачи между workers.
