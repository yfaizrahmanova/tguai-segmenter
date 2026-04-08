"""
TGUAI — Модуль сегментации и классификации отзывов
====================================================
Демонстрирует проблему неконтролируемой генерации LLM
и решает её через Output Manipulation:
  - Структурированный JSON-вывод
  - Guardrails по множеству допустимых меток
  - Верификация и восстановление при ошибках
  - Сравнение prompt-стратегий (baseline vs structured vs few-shot)

Поддерживаемые провайдеры (выбрать в PROVIDER ниже):
  "deepseek"   — api.deepseek.com, бесплатно 5М токенов при регистрации
                 регистрация: platform.deepseek.com (только email, без VPN)
  "openrouter" — openrouter.ai, бесплатный tier для deepseek-v3
                 регистрация: openrouter.ai (email или Google)
  "anthropic"  — для ВКР, требует платного ключа
"""

import json
import re
import time
import random
from typing import Optional
import urllib.request


# ─── Конфигурация ────────────────────────────────────────────────────────────
# Выбери провайдера и вставь свой ключ:

PROVIDER = "openrouter"   # "deepseek" | "openrouter" | "anthropic"

# DeepSeek: platform.deepseek.com → API Keys → Create
# OpenRouter: openrouter.ai → Keys → Create Key
# Anthropic: console.anthropic.com → API Keys
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
    "anthropic": {
        "url":   "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-20250514",
        "fmt":   "anthropic",
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
    """
    Универсальный HTTP-клиент.
    Поддерживает OpenAI-совместимый формат (DeepSeek, OpenRouter)
    и Anthropic формат.
    """
    cfg = _CFG

    if cfg["fmt"] == "openai":
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
        }
        # OpenRouter требует дополнительный заголовок
        if PROVIDER == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/tguai-project"
            headers["X-Title"] = "TGUAI Segmenter"

        data = json.dumps(body).encode()
        req  = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]

    else:  # anthropic
        messages = [{"role": "user", "content": prompt}]
        body = {
            "model": cfg["model"],
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        }
        data = json.dumps(body).encode()
        req  = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return result["content"][0]["text"]


# Алиас для обратной совместимости
call_claude = call_llm


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

    dataset = sys.argv[1] if len(sys.argv) > 1 else "merged_1209.json"
    print("=== Запуск эксперимента (15 отзывов × 3 стратегии) ===\n")
    results = run_experiment(dataset, n_samples=15)
    summary = summarize_results(results)

    print("\n=== Итоги ===")
    for strat, s in summary.items():
        print(f"{strat:12s} | успех {s['success_rate']}% | "
              f"спанов {s['avg_spans']} | ошибок {s['avg_errors']} | "
              f"исправлено {s['avg_fixed']}")
