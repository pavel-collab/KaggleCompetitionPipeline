"""
RabbitMQ worker for processing Kaggle competition tasks.

============================================================
RabbitMQ BASICS (для понимания кода ниже):
============================================================

RabbitMQ - это брокер сообщений, работает по принципу "очередей".

Основные концепции:
1. PRODUCER (отправитель) - отправляет сообщения в очередь
2. QUEUE (очередь) - хранит сообщения, пока их не обработают
3. CONSUMER (получатель/worker) - читает и обрабатывает сообщения

Как это работает:
    [Producer] --> [Queue] --> [Consumer]

    Например:
    [main.py публикует задачу] --> [competitions_queue] --> [worker.py обрабатывает]

Преимущества:
- Задачи не теряются (хранятся в очереди)
- Можно запустить несколько workers для параллельной обработки
- Producer не ждёт завершения задачи (асинхронность)

============================================================
PIKA - это Python клиент для RabbitMQ
============================================================

Основные методы:
- connection = pika.BlockingConnection(params) - подключение к RabbitMQ
- channel = connection.channel() - создание канала для работы
- channel.queue_declare(queue='name') - создание очереди
- channel.basic_publish(...) - отправка сообщения в очередь
- channel.basic_consume(...) - подписка на очередь для получения сообщений
- channel.start_consuming() - начало бесконечного цикла обработки

============================================================
"""

import json
import time
from datetime import datetime
from typing import Any

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.config import settings
from app.database import mark_as_shown, upsert_competition
from app.llm import classify_competition, translate_description
from app.telegram import send_competition_notification, send_error_notification


# ============================================================
# Названия очередей
# ============================================================

# Очередь для классификации новых соревнований
CLASSIFY_QUEUE = "classify_competition"

# Очередь для отправки уведомлений в Telegram
NOTIFY_QUEUE = "notify_competition"


# ============================================================
# Подключение к RabbitMQ
# ============================================================


def get_connection() -> pika.BlockingConnection:
    """
    Создаёт подключение к RabbitMQ.

    BlockingConnection - синхронное подключение, проще в использовании.
    Для production можно использовать SelectConnection (асинхронное).
    """
    # Параметры подключения берём из настроек
    credentials = pika.PlainCredentials(
        settings.rabbitmq_user,
        settings.rabbitmq_password,
    )
    parameters = pika.ConnectionParameters(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        credentials=credentials,
        # Heartbeat - проверка что соединение живо (в секундах)
        heartbeat=600,
        # Таймаут на блокирующие операции
        blocked_connection_timeout=300,
    )
    return pika.BlockingConnection(parameters)


def wait_for_rabbitmq(max_retries: int = 30, delay: int = 2) -> None:
    """
    Ждёт пока RabbitMQ станет доступен.
    Полезно при запуске в Docker - RabbitMQ может стартовать дольше.
    """
    for attempt in range(max_retries):
        try:
            connection = get_connection()
            connection.close()
            print("RabbitMQ is ready!")
            return
        except pika.exceptions.AMQPConnectionError:
            print(f"Waiting for RabbitMQ... attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to RabbitMQ")


# ============================================================
# Публикация задач (Producer)
# ============================================================


def publish_task(queue_name: str, data: dict[str, Any]) -> None:
    """
    Публикует задачу в указанную очередь.

    Args:
        queue_name: Имя очереди (например, 'classify_competition')
        data: Данные задачи в виде словаря (будут сериализованы в JSON)
    """
    connection = get_connection()
    channel = connection.channel()

    # queue_declare - создаёт очередь если её нет, или просто проверяет что есть
    # durable=True - очередь сохранится при перезапуске RabbitMQ
    channel.queue_declare(queue=queue_name, durable=True)

    # Сериализуем данные в JSON
    message = json.dumps(data, ensure_ascii=False)

    # Публикуем сообщение
    # exchange='' - используем default exchange (прямая отправка в очередь)
    # routing_key - имя очереди куда отправляем
    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=message.encode("utf-8"),
        # delivery_mode=2 - сообщение сохранится на диск (persistent)
        properties=pika.BasicProperties(delivery_mode=2),
    )

    print(f"Published task to {queue_name}: {data.get('title', 'unknown')}")
    connection.close()


def publish_classify_task(competition_data: dict[str, Any]) -> None:
    """Публикует задачу на классификацию соревнования."""
    publish_task(CLASSIFY_QUEUE, competition_data)


def publish_notify_task(competition_data: dict[str, Any]) -> None:
    """Публикует задачу на отправку уведомления."""
    publish_task(NOTIFY_QUEUE, competition_data)


# ============================================================
# Обработка задач (Consumer)
# ============================================================


def process_classify_task(data: dict[str, Any]) -> None:
    """
    Обрабатывает задачу классификации.

    1. Вызывает LLM для определения типа соревнования
    2. Фильтрует по нужным типам (CLASSIC ML, LLM/NLP, CV)
    3. Сохраняет в БД
    4. Создаёт задачу на уведомление
    """
    print(f"Processing classification for: {data.get('competition_title')}")

    try:
        # Классифицируем через LLM
        result = classify_competition(
            title=data.get("competition_title", ""),
            link=data.get("link", ""),
            date_start=data.get("date_start", ""),
            deadline=data.get("deadline", ""),
            description=data.get("description", ""),
            tags=data.get("tags", ""),
        )

        # Фильтруем - пропускаем только интересные типы
        if result.type in ["CLASSIC ML", "LLM/NLP", "CV"]:
            # Парсим даты
            date_start = None
            deadline = None
            try:
                date_start = datetime.strptime(result.date_start, "%Y-%m-%d")
            except ValueError:
                pass
            try:
                deadline = datetime.strptime(result.deadline, "%Y-%m-%d")
            except ValueError:
                pass

            # Сохраняем в БД
            upsert_competition(
                title=result.title,
                link=result.link,
                date_start=date_start,
                deadline=deadline,
                description=result.description,
                competition_type=result.type,
            )

            print(f"Saved competition: {result.title} (type: {result.type})")
        else:
            print(f"Skipped competition (type={result.type}): {result.title}")

    except Exception as e:
        print(f"Error processing classification: {e}")
        send_error_notification(f"Classification error: {e}")


def process_notify_task(data: dict[str, Any]) -> None:
    """
    Обрабатывает задачу отправки уведомления.

    1. Переводит описание на русский через LLM
    2. Отправляет в Telegram
    3. Помечает соревнование как показанное
    """
    print(f"Processing notification for: {data.get('title')}")

    try:
        # Переводим описание на русский
        description_ru = translate_description(data.get("description", ""))

        # Отправляем в Telegram
        send_competition_notification(
            title=data["title"],
            link=data["link"],
            deadline=str(data.get("deadline", "")),
            competition_type=data.get("type", ""),
            description_ru=description_ru,
        )

        # Помечаем как показанное
        mark_as_shown(data["title"])

        print(f"Notified about: {data['title']}")

    except Exception as e:
        print(f"Error processing notification: {e}")
        send_error_notification(f"Notification error: {e}")


# ============================================================
# Worker (Consumer) - бесконечный цикл обработки задач
# ============================================================


def make_callback(process_func):
    """
    Создаёт callback-функцию для обработки сообщений из очереди.

    Callback вызывается каждый раз когда приходит новое сообщение.

    Параметры callback (передаются RabbitMQ автоматически):
    - channel: канал через который пришло сообщение
    - method: метаданные доставки (включая delivery_tag для подтверждения)
    - properties: свойства сообщения
    - body: тело сообщения (bytes)
    """
    def callback(
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        # Десериализуем JSON
        data = json.loads(body.decode("utf-8"))

        try:
            # Обрабатываем задачу
            process_func(data)

            # basic_ack - подтверждаем что сообщение обработано
            # После этого RabbitMQ удалит его из очереди
            # delivery_tag - уникальный ID этого сообщения
            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Error in callback: {e}")
            # basic_nack - сообщаем что не смогли обработать
            # requeue=True - вернуть сообщение в очередь для повторной попытки
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    return callback


def run_worker(queue_name: str, process_func) -> None:
    """
    Запускает worker для обработки задач из очереди.

    Worker работает бесконечно, обрабатывая задачи по мере поступления.
    """
    print(f"Starting worker for queue: {queue_name}")

    # Ждём пока RabbitMQ станет доступен
    wait_for_rabbitmq()

    connection = get_connection()
    channel = connection.channel()

    # Создаём очередь (если её ещё нет)
    channel.queue_declare(queue=queue_name, durable=True)

    # prefetch_count=1 - обрабатываем по одному сообщению за раз
    # Это важно для равномерного распределения нагрузки между workers
    channel.basic_qos(prefetch_count=1)

    # Подписываемся на очередь
    # on_message_callback - функция которая будет вызвана при получении сообщения
    channel.basic_consume(
        queue=queue_name,
        on_message_callback=make_callback(process_func),
    )

    print(f"Worker ready. Waiting for tasks in {queue_name}...")

    # Запускаем бесконечный цикл обработки
    # Это блокирующий вызов - программа будет работать пока не остановят
    channel.start_consuming()


def run_classify_worker() -> None:
    """Запускает worker для классификации соревнований."""
    run_worker(CLASSIFY_QUEUE, process_classify_task)


def run_notify_worker() -> None:
    """Запускает worker для отправки уведомлений."""
    run_worker(NOTIFY_QUEUE, process_notify_task)
