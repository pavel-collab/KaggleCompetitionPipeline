# Training Pipeline for Competition Classification

MVP для обучения локальной модели классификации соревнований Kaggle.

## Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Пайплайн обучения](#пайплайн-обучения)
- [Интеграция в основной проект](#интеграция-в-основной-проект)
- [Troubleshooting](#troubleshooting)

## Структура

```
training/
├── config.py           # Конфигурация (пути, гиперпараметры)
├── collect_data.py     # Сбор данных с Kaggle API
├── label_data.py       # Разметка через LLM (OpenRouter)
├── prepare_dataset.py  # Подготовка train/val/test
├── train_bert.py       # Fine-tuning BERT (CPU/GPU)
├── train_lora.py       # Fine-tuning LoRA (GPU only)
├── evaluate.py         # Оценка моделей и сравнение с baseline
├── inference.py        # Inference обученных моделей
├── requirements.txt    # Зависимости
└── data/
    ├── raw/            # Сырые данные с Kaggle
    └── processed/      # Размеченные и подготовленные данные
```

## Требования

### Минимальные требования
- Python 3.10+
- 8GB RAM (для BERT)
- Настроенный Kaggle API

### Для LoRA обучения (опционально)
- NVIDIA GPU с 8GB+ VRAM
- CUDA 11.8+

## Установка

### 1. Создание виртуального окружения

```bash
cd training
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

Для LoRA обучения дополнительно:
```bash
pip install unsloth trl
```

### 3. Настройка Kaggle API

1. Зайди на https://www.kaggle.com/settings
2. В секции "API" нажми "Create New Token"
3. Скачается файл `kaggle.json`
4. Помести его в нужную директорию:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Положи в C:\Users\<username>\.kaggle\kaggle.json
```

### 4. Настройка OpenRouter (для разметки)

Получи API ключ на https://openrouter.ai/keys

```bash
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_MODEL="openai/gpt-4o-mini"
```

## Пайплайн обучения

### Шаг 1: Сбор данных

```bash
cd training
python collect_data.py
```

**Что делает:**
- Подключается к Kaggle API
- Скачивает метаданные соревнований из разных категорий
- Сохраняет в `data/raw/competitions.json`

**Ожидаемый результат:**
```
Kaggle Competition Collector
============================================================
Authenticating with Kaggle API...

Fetching category: all
  [1] Titanic - Machine Learning from Disaster...
  [2] House Prices - Advanced Regression...
  ...
Saved 500 competitions to data/raw/competitions.json
```

### Шаг 2: Разметка данных

```bash
# Убедись, что переменные окружения установлены
export OPENAI_API_KEY="sk-or-v1-..."

python label_data.py
```

**Что делает:**
- Загружает сырые данные
- Отправляет каждое соревнование в LLM для классификации
- Сохраняет результаты в `data/processed/labeled_competitions.json`
- Поддерживает прерывание и возобновление (сохраняет прогресс каждые 10 записей)

**Ожидаемый результат:**
```
Competition Labeler
============================================================
Loaded 500 competitions
Already labeled: 0

[1/500] Titanic - Machine Learning from Disaster... -> CLASSIC ML
[2/500] House Prices - Advanced Regression... -> CLASSIC ML
[3/500] Stable Diffusion - Image to Prompts... -> CV
...

Labeling Statistics:
============================================================
  CLASSIC ML: 180
  LLM/NLP: 95
  CV: 120
  Other: 105
  Total: 500
```

**Стоимость:** ~$0.10-0.30 за 500 соревнований (gpt-4o-mini)

### Шаг 3: Подготовка датасета

```bash
python prepare_dataset.py
```

**Что делает:**
- Форматирует данные для обучения
- Делит на train/val/test (80/10/10)
- Применяет стратификацию по классам

**Ожидаемый результат:**
```
Dataset Preparation
============================================================
Loaded 500 labeled competitions
Prepared 500 valid samples

Train (400 samples):
  CLASSIC ML: 144 (36.0%)
  CV: 96 (24.0%)
  LLM/NLP: 76 (19.0%)
  Other: 84 (21.0%)

Validation (50 samples):
  ...

Test (50 samples):
  ...
```

### Шаг 4: Обучение модели

#### Вариант A: BERT (рекомендуется)

```bash
python train_bert.py
```

**Характеристики:**
- Модель: DistilBERT (~67M параметров)
- Размер: ~250MB
- Работает на CPU и GPU
- Время обучения: 5-15 минут (GPU) / 1-2 часа (CPU)
- Inference: 100-500ms на CPU

**Ожидаемый результат:**
```
BERT Fine-tuning for Competition Classification
============================================================
Using device: cuda

Loading model: distilbert-base-uncased
Loading datasets...
  Train: 400 samples
  Val:   50 samples
  Test:  50 samples

Starting training...
Epoch 1/3: loss=1.2, accuracy=0.65
Epoch 2/3: loss=0.5, accuracy=0.85
Epoch 3/3: loss=0.2, accuracy=0.92

Test Results:
  accuracy: 0.9000
  f1_macro: 0.8850

Classification Report:
              precision    recall  f1-score   support
  CLASSIC ML       0.92      0.95      0.93        20
         CV       0.88      0.90      0.89        15
     LLM/NLP       0.90      0.85      0.87        10
       Other       0.85      0.80      0.82         5

Saving model to: models/bert_classifier
```

#### Вариант B: LoRA (только GPU)

```bash
pip install unsloth trl
python train_lora.py
```

**Характеристики:**
- Модель: TinyLlama с LoRA адаптерами
- Размер: ~1-2GB (4-bit квантизация)
- Требует GPU с 8GB+ VRAM
- Генеративный подход (instruction-tuning)

### Шаг 5: Тестирование

```bash
# BERT
python inference.py --model bert "Predict house prices based on size, location and age"
# Output: CLASSIC ML

python inference.py --model bert "Classify images of cats and dogs"
# Output: CV

python inference.py --model bert "Summarize customer reviews"
# Output: LLM/NLP
```

### Шаг 6: Оценка моделей

```bash
# Оценить все доступные модели
python evaluate.py

# Только BERT
python evaluate.py --bert-only

# Только LoRA
python evaluate.py --lora-only

# Только OpenRouter LLM (baseline)
python evaluate.py --llm-only
```

**Что делает:**
- Загружает тестовый набор данных
- Оценивает BERT классификатор (если обучен)
- Оценивает LoRA классификатор (если обучен и есть GPU)
- Оценивает OpenRouter LLM как baseline
- Выводит сравнительную таблицу метрик
- Сохраняет результаты в `evaluation_results.json`

**Ожидаемый результат:**
```
Model Evaluation
======================================================================

Loading test data...
Loaded 50 test samples

Evaluating BERT classifier...
  Loading BERT model...
  Running inference...

Evaluating OpenRouter LLM (baseline)...
  Using model: openai/gpt-4o-mini
  Running inference...

======================================================================
EVALUATION RESULTS
======================================================================

Model                Accuracy   F1 (macro)  F1 (weighted)    Samples
----------------------------------------------------------------------
BERT                   0.9000       0.8850         0.8920      50/50
OpenRouter LLM         0.9400       0.9320         0.9380      50/50

======================================================================
COMPARISON WITH BASELINE (OpenRouter LLM)
======================================================================

BERT vs OpenRouter LLM:
  Accuracy:  -0.0400 (-4.00%)
  F1 macro:  -0.0470 (-4.70%)
```

**Метрики:**
- **Accuracy:** Доля правильных ответов
- **F1 macro:** Среднее F1 по всем классам (не учитывает дисбаланс)
- **F1 weighted:** Взвешенное F1 (учитывает дисбаланс классов)

Сравнение с baseline показывает, насколько локальная модель уступает (или превосходит) облачному LLM.

## Интеграция в основной проект

После обучения замени вызов LLM в `app/llm.py`:

```python
# === Было ===
from langchain_openai import ChatOpenAI

def classify_competition(...) -> CompetitionClassification:
    llm = ChatOpenAI(...)
    chain = CLASSIFICATION_PROMPT | llm.with_structured_output(...)
    result = chain.invoke({...})
    return result


# === Стало ===
from transformers import pipeline

# Загрузи модель один раз при старте
classifier = pipeline(
    "text-classification",
    model="./training/models/bert_classifier",
    device=-1,  # CPU, или 0 для GPU
)

def classify_competition(
    title: str,
    link: str,
    date_start: str,
    deadline: str,
    description: str,
    tags: str,
) -> CompetitionClassification:
    # Форматируем текст как при обучении
    text = f"{title} [SEP] Tags: {tags} [SEP] {description[:1500]}"

    # Классификация
    result = classifier(text, truncation=True, max_length=512)[0]

    return CompetitionClassification(
        title=title,
        link=link,
        date_start=date_start,
        deadline=deadline,
        description=description,
        type=result["label"],
    )
```

## Гиперпараметры

Настраиваются в `config.py`:

| Параметр | BERT | LoRA | Описание |
|----------|------|------|----------|
| MODEL_NAME | distilbert-base-uncased | unsloth/tinyllama-bnb-4bit | Базовая модель |
| BATCH_SIZE | 16 | 4 | Размер батча |
| EPOCHS | 3 | 3 | Количество эпох |
| LEARNING_RATE | 2e-5 | 2e-4 | Learning rate |
| MAX_LENGTH | 512 | 512 | Максимальная длина текста |

## Troubleshooting

### Ошибка: "kaggle.json not found"
```
OSError: Could not find kaggle.json
```
**Решение:** Настрой Kaggle API (см. секцию "Установка")

### Ошибка: "OPENAI_API_KEY not set"
```
Error: OPENAI_API_KEY not set!
```
**Решение:**
```bash
export OPENAI_API_KEY="sk-or-v1-..."
```

### Ошибка: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Решение:** Уменьши batch_size в `config.py`:
```python
BERT_BATCH_SIZE = 8  # вместо 16
```

### Ошибка: "unsloth not installed"
```
ImportError: unsloth not installed
```
**Решение:**
```bash
pip install unsloth trl
```

### Медленное обучение на CPU
Это нормально. BERT обучение на CPU занимает 1-2 часа.
Для ускорения используй GPU или Google Colab.

## Рекомендации

1. **Минимум данных:** 100-200 примеров на класс для хорошего качества
2. **Проверка разметки:** Просмотри ~50 примеров вручную перед обучением
3. **Raspberry Pi:** Используй BERT, опционально с ONNX-оптимизацией
4. **Дообучение:** При появлении новых соревнований можно дообучить модель
