"""
Выбор SearchStrategy по запросу — чистая, без LLM.

После cleanup'а стратегий осталось три:
- UI_TO_DATABASE — запрос упоминает UI и backend/data одновременно.
- DATABASE_TO_UI — запрос про модели/таблицы и где они отображаются.
- SEMANTIC_ONLY — всё остальное (по умолчанию).
"""

from src.search.retrieval import SearchStrategy

_UI_KEYWORDS = (
    "ui", "view", "frontend", "button", "form", "component",
    "template", "html", "css", "react", "vue", "angular",
    "page", "screen", "widget", "render", "display", "show",
)
_DB_CONNECTION_KEYWORDS = (
    "database", "model", "data", "api", "backend", "server",
)
_DB_KEYWORDS = (
    "database", "model", "table", "schema", "migration", "orm",
    "django model", "sqlalchemy", "entity", "repository", "dao",
)
_UI_USAGE_KEYWORDS = (
    "used", "displayed", "shown", "view", "frontend", "ui",
)


class StrategyService:
    @staticmethod
    def detect(query: str) -> SearchStrategy:
        q = query.lower()
        # DATABASE_TO_UI проверяется первым — более специфичная пара
        # (model/table + used/displayed). Иначе слово "view" в UI_KEYWORDS
        # перетянет вопросы "где используется User model в view" в UI_TO_DATABASE.
        if any(k in q for k in _DB_KEYWORDS) and any(k in q for k in _UI_USAGE_KEYWORDS):
            return SearchStrategy.DATABASE_TO_UI
        if any(k in q for k in _UI_KEYWORDS) and any(k in q for k in _DB_CONNECTION_KEYWORDS):
            return SearchStrategy.UI_TO_DATABASE
        return SearchStrategy.SEMANTIC_ONLY
