"""
TGUAI — Модуль сегментации и классификации отзывов
====================================================
Демонстрирует проблему неконтролируемой генерации LLM
и решает её через Output Manipulation:
  - Структурированный JSON-вывод
  - Guardrails по множеству допустимых меток
  - Верификация и восстановление при ошибках
  - Сравнение prompt-стратегий (baseline vs structured vs few-shot)

Провайдер: OpenRouter (openrouter.ai)
  - Бесплатный tier, регистрация через email или Google
  - Получить ключ: openrouter.ai → Keys → Create Key
"""

import json
import re
import time
import random
from typing import Optional
import urllib.request


# ─── Конфигурация ────────────────────────────────────────────────────────────
# Вставь свой ключ от OpenRouter: openrouter.ai → Keys → Create Key

PROVIDER = "openrouter"

API_KEY = "ВСТАВЬ_КЛЮЧ_СЮДА"  # openrouter.ai → Keys → Create Key

_CONFIGS = {
    "deepseek": {
        "url":   "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",          # DeepSeek-V3
        "fmt":   "openai",
    },
    "openrouter": {
        "url":   "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-oss-120b:free",  # бесплатная квота
        "fmt":   "openai",
    },
}

_CFG = _CONFIGS[PROVIDER]

VALID_LABELS = {
    "Эмоции", "Вывод", "Анализ", "Читательские ожидания",
    "Сюжет", "Герои", "Экспозиция", "Нарратив", "Стиль",
    "Миры", "Сравнение", "Автор", "Жанр", "Контекст",
    "Аудитория", "Мораль", "Интертекст", "Озвучка",
    "Цитаты", "Спойлер", "Оформление", "Перевод"
}

LABEL_LIST_STR = "\n".join(f"- {l}" for l in sorted(VALID_LABELS))


# ─── Три стратегии промптинга ─────────────────────────────────────────────────

def prompt_baseline(text: str) -> str:
    """Baseline: никаких ограничений на формат — демонстрирует нестабильность."""
    return (
        f"Раздели следующий отзыв на смысловые части и определи тему каждой части.\n\n"
        f"Отзыв: {text}"
    )


def prompt_structured(text: str) -> str:
    """Structured output: JSON-схема + список допустимых меток."""
    return (
        f"Раздели отзыв на смысловые спаны. Для каждого спана определи метку.\n\n"
        f"ДОПУСТИМЫЕ МЕТКИ (только из этого списка):\n{LABEL_LIST_STR}\n\n"
        f"ФОРМАТ ОТВЕТА — строго JSON, без лишнего текста:\n"
        f'{{"spans": [{{"text": "...", "label": "...", "sentiment": "positive|negative|neutral"}}]}}\n\n'
        f"Отзыв: {text}"
    )


def prompt_few_shot(text: str, examples: list[dict]) -> str:
    """Few-shot: 2 примера из реального датасета + строгая схема."""
    ex_str = ""
    for ex in examples[:2]:
        spans_json = [
            {"text": s["text"][:80], "label": s["labels"][0], "sentiment": "positive"}
            for s in ex["spans"][:3] if s["labels"]
        ]
        ex_str += (
            f"Отзыв: {ex['text'][:120]}\n"
            f"Ответ: {json.dumps({'spans': spans_json}, ensure_ascii=False)}\n\n"
        )

    return (
        f"Задача: разбить отзыв на спаны и назначить каждому метку из фиксированного списка.\n\n"
        f"ДОПУСТИМЫЕ МЕТКИ:\n{LABEL_LIST_STR}\n\n"
        f"ПРИМЕРЫ:\n{ex_str}"
        f"ФОРМАТ: только JSON {{'spans': [{{'text':..., 'label':..., 'sentiment': positive|negative|neutral}}]}}\n\n"
        f"Отзыв: {text}"
    )


# ─── API-клиент ───────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "", max_tokens: int = 1000) -> str:
    """Универсальный HTTP-клиент для OpenAI-совместимых провайдеров (OpenRouter)."""
    cfg = _CFG

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": cfg["model"],
        "max_tokens": max_tokens,
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/tguai-project",
        "X-Title": "TGUAI Segmenter",
    }

    data = json.dumps(body).encode()
    req  = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


# ─── Парсинг и валидация вывода ───────────────────────────────────────────────

def parse_json_response(raw: str) -> Optional[dict]:
    """Извлекает JSON из ответа, даже если модель добавила текст вокруг."""
    # Попытка 1: весь ответ — JSON
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Попытка 2: ищем ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Попытка 3: ищем первый {...}
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_spans(parsed: Optional[dict]) -> dict:
    """
    Guardrail: проверяет и исправляет вывод модели.
    Возвращает словарь с полями:
      spans     — список провалидированных спанов
      errors    — список найденных нарушений
      fixed     — сколько спанов было исправлено
    """
    errors = []
    fixed  = 0
    spans  = []

    if parsed is None:
        return {"spans": [], "errors": ["Не удалось распарсить JSON"], "fixed": 0}

    raw_spans = parsed.get("spans", [])
    if not isinstance(raw_spans, list):
        return {"spans": [], "errors": ["Поле 'spans' не является списком"], "fixed": 0}

    for i, span in enumerate(raw_spans):
        label     = span.get("label", "")
        text      = span.get("text", "").strip()
        sentiment = span.get("sentiment", "neutral")

        # Проверка метки
        if label not in VALID_LABELS:
            errors.append(f"Спан {i}: неизвестная метка '{label}'")
            # Попытка исправить через нечёткое совпадение
            best = _fuzzy_match_label(label)
            if best:
                span["label"] = best
                span["_fixed_label"] = True
                fixed += 1
            else:
                span["label"] = "Анализ"  # fallback
                fixed += 1

        # Проверка текста
        if not text:
            errors.append(f"Спан {i}: пустой текст")

        # Проверка sentiment
        if sentiment not in {"positive", "negative", "neutral"}:
            span["sentiment"] = "neutral"
            errors.append(f"Спан {i}: неверный sentiment '{sentiment}', заменён на neutral")
            fixed += 1

        spans.append(span)

    return {"spans": spans, "errors": errors, "fixed": fixed}


def _fuzzy_match_label(label: str) -> Optional[str]:
    """Простое нечёткое совпадение по подстроке."""
    label_lower = label.lower()
    for valid in VALID_LABELS:
        if valid.lower() in label_lower or label_lower in valid.lower():
            return valid
    return None


# ─── Основная функция ─────────────────────────────────────────────────────────

def segment_review(
    text: str,
    strategy: str = "structured",
    examples: Optional[list] = None,
) -> dict:
    """
    Сегментирует отзыв одной из трёх стратегий.

    strategy: 'baseline' | 'structured' | 'few_shot'
    examples: список отзывов из датасета (нужен для few_shot)

    Возвращает:
        {
          strategy, raw_response, parsed, validation,
          success, n_spans, n_errors, n_fixed
        }
    """
    # Формируем промпт
    if strategy == "baseline":
        prompt = prompt_baseline(text)
        system = ""
    elif strategy == "structured":
        prompt = prompt_structured(text)
        system = "Ты — эксперт по анализу книжных отзывов. Отвечай строго в формате JSON."
    elif strategy == "few_shot":
        if not examples:
            raise ValueError("few_shot требует список examples")
        prompt = prompt_few_shot(text, examples)
        system = "Ты — эксперт по анализу книжных отзывов. Отвечай строго в формате JSON."
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    # Вызов API
    t0  = time.time()
    raw = call_llm(prompt, system=system)
    elapsed = round(time.time() - t0, 2)

    # Парсинг и валидация
    parsed     = parse_json_response(raw)
    validation = validate_spans(parsed)

    return {
        "strategy":     strategy,
        "provider":     PROVIDER,
        "model":        _CFG["model"],
        "raw_response": raw,
        "parsed":       parsed,
        "validation":   validation,
        "success":      parsed is not None,
        "n_spans":      len(validation["spans"]),
        "n_errors":     len(validation["errors"]),
        "n_fixed":      validation["fixed"],
        "elapsed_sec":  elapsed,
    }


# ─── Загрузка сырых отзывов ───────────────────────────────────────────────────

def load_raw_reviews(path: str, text_column: str = "Текст отзыва") -> list[str]:
    """
    Загружает сырые отзывы (без аннотаций) из файла.
    Поддерживаемые форматы:
      - .csv : колонка text_column (по умолчанию 'Текст отзыва')
      - .txt : каждая непустая строка — отдельный отзыв
      - .json: массив строк ["отзыв1", "отзыв2", ...]
               или массив объектов [{"text": "..."}, ...]
    """
    import csv

    with open(path, encoding="utf-8") as f:
        if path.endswith(".csv"):
            reader = csv.DictReader(f)
            return [
                row[text_column].strip()
                for row in reader
                if row.get(text_column, "").strip()
            ]
        elif path.endswith(".json"):
            data = json.load(f)
            if data and isinstance(data[0], str):
                return data
            return [item["text"] for item in data if item.get("text")]
        else:
            return [line.strip() for line in f if line.strip()]


def process_raw_file(
    raw_path: str,
    library_path: str = "merged_1209.json",
    strategy: str = "structured",
    output_path: Optional[str] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Сегментирует отзывы из сырого файла, используя аннотированную библиотеку
    только как источник few-shot примеров.

    raw_path     — файл с сырыми отзывами (.txt или .json)
    library_path — аннотированный датасет для few-shot примеров
    strategy     — 'baseline' | 'structured' | 'few_shot'
    output_path  — если указан, сохраняет результаты в JSON
    """
    reviews = load_raw_reviews(raw_path)

    examples = []
    if strategy == "few_shot":
        with open(library_path, encoding="utf-8") as f:
            library = json.load(f)
        random.seed(seed)
        examples = random.sample(library, min(5, len(library)))

    results = []
    for i, text in enumerate(reviews):
        print(f"[{i+1}/{len(reviews)}] Обрабатываем отзыв ({len(text)} симв.)...")
        try:
            res = segment_review(text, strategy=strategy, examples=examples)
            res["review_text"] = text
            res["sample_idx"]  = i
            results.append(res)
            print(f"  → {res['n_spans']} спанов, {res['n_errors']} ошибок "
                  f"({res['elapsed_sec']}s)")
        except Exception as e:
            print(f"  → ОШИБКА: {e}")
        time.sleep(0.5)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "analytics": analyze_segments(results)},
                      f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены: {output_path}")

    return results


def analyze_segments(results: list[dict]) -> dict:
    """
    Лёгкая аналитика по сегментам:
      - всего отзывов и спанов
      - топ-10 меток
      - распределение sentiment
      - среднее спанов на отзыв
    """
    from collections import Counter

    label_counter     = Counter()
    sentiment_counter = Counter()
    total_spans       = 0
    success_count     = sum(1 for r in results if r["success"])

    for r in results:
        spans = r.get("validation", {}).get("spans", [])
        total_spans += len(spans)
        for span in spans:
            label_counter[span.get("label", "?")] += 1
            sentiment_counter[span.get("sentiment", "neutral")] += 1

    n = len(results) or 1
    analytics = {
        "total_reviews":    len(results),
        "success_rate":     round(success_count / n * 100, 1),
        "total_spans":      total_spans,
        "avg_spans":        round(total_spans / n, 2),
        "top_labels":       label_counter.most_common(10),
        "sentiment_dist":   dict(sentiment_counter),
    }

    print("\n=== Аналитика сегментов ===")
    print(f"Отзывов обработано : {analytics['total_reviews']}")
    print(f"Успешных парсингов : {analytics['success_rate']}%")
    print(f"Всего спанов       : {analytics['total_spans']}  (avg {analytics['avg_spans']} на отзыв)")
    print(f"Sentiment          : {analytics['sentiment_dist']}")
    print("Топ-10 меток:")
    for label, cnt in analytics["top_labels"]:
        print(f"  {label:<30} {cnt}")

    return analytics


# ─── Эксперимент: сравнение стратегий ────────────────────────────────────────

def run_experiment(dataset_path: str, n_samples: int = 10, seed: int = 42) -> list[dict]:
    """
    Прогоняет n_samples отзывов через все три стратегии.
    Возвращает список результатов для анализа в Jupyter.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    random.seed(seed)
    samples  = random.sample(data, min(n_samples, len(data)))
    examples = random.sample(data, 5)  # few-shot примеры
    results  = []

    for i, item in enumerate(samples):
        print(f"[{i+1}/{n_samples}] Обрабатываем отзыв ({len(item['text'])} симв.)...")
        for strategy in ["baseline", "structured", "few_shot"]:
            try:
                res = segment_review(
                    item["text"],
                    strategy=strategy,
                    examples=examples,
                )
                res["review_text"]    = item["text"]
                res["ground_truth"]   = item["spans"]
                res["sample_idx"]     = i
                results.append(res)
                print(f"  {strategy:12s} → {res['n_spans']} спанов, "
                      f"{res['n_errors']} ошибок, {res['n_fixed']} исправлено "
                      f"({res['elapsed_sec']}s)")
            except Exception as e:
                print(f"  {strategy:12s} → ОШИБКА: {e}")
            time.sleep(0.5)  # rate limit

    return results


def summarize_results(results: list[dict]) -> dict:
    """Агрегирует результаты эксперимента по стратегиям."""
    from collections import defaultdict
    summary = defaultdict(lambda: {
        "total": 0, "success": 0,
        "avg_spans": 0, "avg_errors": 0, "avg_fixed": 0
    })

    for r in results:
        s = summary[r["strategy"]]
        s["total"]      += 1
        s["success"]    += int(r["success"])
        s["avg_spans"]  += r["n_spans"]
        s["avg_errors"] += r["n_errors"]
        s["avg_fixed"]  += r["n_fixed"]

    for strat, s in summary.items():
        n = s["total"] or 1
        s["success_rate"] = round(s["success"] / n * 100, 1)
        s["avg_spans"]    = round(s["avg_spans"] / n, 1)
        s["avg_errors"]   = round(s["avg_errors"] / n, 1)
        s["avg_fixed"]    = round(s["avg_fixed"] / n, 1)

    return dict(summary)


# ─── Точка входа (для тестирования без Jupyter) ───────────────────────────────

if __name__ == "__main__":
    import sys

    # Режим 1: python segmenter_module.py --raw reviews.txt [strategy] [output.json]
    # Режим 2: python segmenter_module.py [dataset.json]   (эксперимент сравнения стратегий)
    if len(sys.argv) > 1 and sys.argv[1] == "--raw":
        raw_path    = sys.argv[2] if len(sys.argv) > 2 else "reviews.txt"
        strategy    = sys.argv[3] if len(sys.argv) > 3 else "structured"
        output_path = sys.argv[4] if len(sys.argv) > 4 else "segmented_results.json"
        print(f"=== Сегментация сырых отзывов: {raw_path} (стратегия: {strategy}) ===\n")
        results = process_raw_file(raw_path, strategy=strategy, output_path=output_path)
    else:
        dataset = sys.argv[1] if len(sys.argv) > 1 else "merged_1209.json"
        print("=== Запуск эксперимента (15 отзывов × 3 стратегии) ===\n")
        results = run_experiment(dataset, n_samples=15)
        summary = summarize_results(results)

        print("\n=== Итоги ===")
        for strat, s in summary.items():
            print(f"{strat:12s} | успех {s['success_rate']}% | "
                  f"спанов {s['avg_spans']} | ошибок {s['avg_errors']} | "
                  f"исправлено {s['avg_fixed']}")
