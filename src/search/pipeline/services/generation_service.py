"""
GenerationService — финальный ответ LLM по контексту.

Контекст состоит из четырёх блоков:
1. Primary — топ-результаты Weaviate (vector + BM25 hybrid).
2. Grep — точные текстовые вхождения идентификаторов через ripgrep.
3. Graph — связанные ноды через cypher path-traversal (UI↔DB).
4. Cypher facts — структурные факты от NL→Cypher chain'а.

Промпт жёстко anti-hallucination:
- Явный список Sources (доступных файлов) перед контекстом.
- Запрет упоминать файлы/функции вне Sources.
- Запрет домысливать синтаксис, импорты, имена.
- Требование ссылок ``file.py:lineno`` на каждое утверждение.
"""

from typing import Any, Dict, Iterable, List, Optional

from ._llm import LLMRole


PROMPT = """Ты — старший разработчик. Отвечаешь по коду, на русском языке.

КРИТИЧЕСКИ ВАЖНО: твой ответ начинается СРАЗУ с заголовка "**Краткий ответ**".
Никаких вступлений, рассуждений, планов. Не пиши "We need to", "Let's",
"Я должен", "Сначала", "Provide", "Must use" и подобное — даже если кажется
что нужно подумать. Думай молча, в ответ выдавай только готовый текст.

## Вопрос
{query}

## Sources (единственные допустимые файлы)
Эти файлы извлечены из индекса проекта. ЛЮБОЙ файл вне этого списка —
галлюцинация и запрещён в ответе.

{sources_list}

## Контекст из кодовой базы

### 1. Прямые совпадения (Weaviate hybrid: vector + BM25)
{primary_context}

### 2. Точные текстовые вхождения (ripgrep по идентификаторам)
{grep_context}

### 3. Связанный код (граф зависимостей)
{graph_context}

### 4. Структурные факты (cypher по графу)
{cypher_context}

## Правила

- Упоминай только файлы из списка Sources. Любой другой путь = галлюцинация.
- Не выдумывай функции, методы, синтаксис. Если в контексте нет кода —
  напиши "в индексе нет деталей реализации".
- Каждое утверждение про код сопровождай ссылкой `path/to/file.py:42`.
- Если контекст не отвечает на вопрос напрямую — напиши: "Не нашёл
  прямого ответа в индексе. Возможно нужна переиндексация или уточнение
  формулировки."
- Используй Graph секцию: там методы класса (CONTAINS), callees (CALLS),
  и call-tree. Если в Graph есть подходящий метод — процитируй его наравне
  с Primary/Grep.
- **Полное покрытие**. Перед написанием раздела «Варианты реализации»
  СНАЧАЛА перечитай весь контекст и найди ВСЕ функции/методы, которые
  что-либо делают с темой запроса. Цель — описать **минимум 7-10 разных
  кейсов, если их есть в контексте**. НЕ останавливайся на 3-5 первых
  попавшихся. Особенно важно: если в Graph секции есть методы с похожим
  именем (CONTAINS-методы класса, например ``_get_position_*_by_book``
  с суффиксами bond / structured_bond / repo / autocall / protected_
  participation / equity / dividend) — каждый из них представляет
  отдельный case (свой тип инструмента) и обязан быть в списке.

## Формат ответа

**Краткий ответ**
2-3 предложения. Со ссылками `file.py:lineno`.

**Найденные варианты** (TL;DR-список ВСЕХ кейсов в контексте, по 1 строке)
Сначала просто перечисли все варианты по 1 строке БЕЗ кода:
- `<file_path>:<line>` — `<функция/метод>` — короткое описание (5-10 слов)
- `<file_path>:<line>` — `<функция/метод>` — короткое описание
- ... (минимум 7-10 строк, если в контексте 7-10+ вариантов)

**Варианты реализации** (детально с кодом)
Для КАЖДОГО пункта из «Найденных вариантов» — отдельная секция:

#### `<file_path>` — `<Function/Method>` (line N)
```python
3-15 строк ключевого фрагмента
```
Что отличает этот вариант: "для bonds: net_pos += quantity";
"для structured_bond: умножение на notional"; "для cash_transfers:
net_pos = transfer.amount"; и т.п.

**Связи кода**
Как элементы связаны (вызовы, наследование, граф). Если в контексте
есть Call tree — используй его.

**Файлы**
Маркированный список — только из Sources. Все файлы, упомянутые в
"Варианты реализации", должны быть здесь."""


def _format_chunks(
    chunks: List[Dict[str, Any]],
    *,
    empty_message: str,
    show_relationship: bool = False,
) -> str:
    """Однообразный рендер chunk'ов в markdown-блок с метой."""
    if not chunks:
        return empty_message
    pieces = []
    for c in chunks:
        name = c.get('name') or 'Unknown'
        ctype = c.get('type') or 'Unknown'
        file_ = c.get('file') or ''
        line = c.get('line')
        code = str(c.get('code') or '')
        head = f"### {name} ({ctype})"
        if show_relationship:
            head += f" — {c.get('relationship', 'RELATED')}"
        loc = f"`{file_}`" + (f":{line}" if line else '')
        pieces.append(f"{head}\n**Файл:** {loc}\n```\n{code}\n```")
    return "\n\n".join(pieces)


def _collect_sources(*chunk_lists: Iterable[Dict[str, Any]]) -> List[str]:
    """Уникальный отсортированный список путей файлов из всех чанков."""
    seen: set = set()
    for chunks in chunk_lists:
        for c in chunks:
            f = c.get('file')
            if f and f not in seen:
                seen.add(f)
    return sorted(seen)


class GenerationService:
    def __init__(self, llm: LLMRole):
        self.llm = llm

    @staticmethod
    def build_prompt(
        *,
        query: str,
        primary: List[Dict[str, Any]],
        graph: List[Dict[str, Any]],
        cypher: Optional[List[Dict[str, Any]]] = None,
        grep: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, List[str]]:
        """
        Собирает финальный prompt и список Sources.

        Вынесено отдельно от ``generate()``, чтобы caller (``GenerateAnswerTask``)
        мог положить тот же prompt в trace_input для Langfuse — мы видим
        что реально ушло в LLM, без догадок.

        Возвращает ``(prompt, sources)``.
        """
        cypher = cypher or []
        grep = grep or []

        primary_context = _format_chunks(
            primary, empty_message="Прямых совпадений не найдено.",
        )
        grep_context = _format_chunks(
            grep, empty_message="Точных текстовых вхождений не найдено.",
        )
        graph_context = _format_chunks(
            graph, empty_message="Связанный код через граф не найден.",
            show_relationship=True,
        )
        cypher_context = "\n".join(
            f"- {c.get('code') or c.get('name') or ''}"
            for c in cypher
        ) or "Структурных фактов из графа не получено."

        sources = _collect_sources(primary, grep, graph)
        sources_list = "\n".join(f"- `{s}`" for s in sources) or "(нет источников)"

        prompt = PROMPT.format(
            query=query,
            sources_list=sources_list,
            primary_context=primary_context,
            grep_context=grep_context,
            graph_context=graph_context,
            cypher_context=cypher_context,
        )
        return prompt, sources

    def generate(
        self,
        *,
        query: str,
        primary: List[Dict[str, Any]],
        graph: List[Dict[str, Any]],
        cypher: Optional[List[Dict[str, Any]]] = None,
        grep: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, str, List[str]]:
        """
        Генерирует ответ.

        Возвращает кортеж ``(answer, prompt, sources)`` — caller'ы
        (``GenerateAnswerTask``) используют ``prompt``/``sources`` для дампа
        в Langfuse trace + опционально в файл.
        """
        prompt, sources = self.build_prompt(
            query=query,
            primary=primary,
            graph=graph,
            cypher=cypher,
            grep=grep,
        )

        answer = self.llm.call(
            prompt,
            name="answer_generator",
            # 12288: новый формат ответа — TL;DR-список + детальные секции
            # для каждого варианта (10+ кейсов, каждый с цитатой кода).
            # При 5 вариантах хватало 8192, при 10 кейсах нужен запас.
            max_tokens=12288,
        )
        return answer, prompt, sources
