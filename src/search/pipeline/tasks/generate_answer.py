import os
from datetime import datetime
from pathlib import Path

from src.infra.logger import get_logger
from ..base.task import SearchTask
from ..services import GenerationService
from .rag_controller import RagControllerTask

logger = get_logger(__name__)


# Сколько символов промпта максимум кладём в Langfuse span.input.
# Полный промпт может быть очень длинным (десятки KB) — UI Langfuse не любит
# гигантские payloads. На диск (через ``RAG_DUMP_PROMPT``) пишем целиком.
_PROMPT_TRACE_LIMIT = 12_000


class GenerateAnswerTask(SearchTask):
    """
    One-shot. Финальная генерация ответа после выхода из loop'а.

    deps: RagControllerTask (а тот тянет DetectStrategyTask).

    context["answer"]            ← str.
    context["sources"]           ← list of {name, file, type, source}.
    context["_answer_prompt"]    ← полный prompt, отправленный в LLM.
    context["_answer_sources"]   ← список путей файлов (для трассы).
    """

    dependencies = [RagControllerTask]

    def run(self, filters):
        chunks = self.context.get("context", [])
        primary = [c for c in chunks if c.get("source") == "primary"]
        graph = [c for c in chunks if c.get("source") == "graph"]
        cypher = [c for c in chunks if c.get("source") == "cypher"]
        grep = [c for c in chunks if c.get("source") == "grep"]

        service = GenerationService(llm=self.executor.answer_llm)
        answer, prompt, sources_files = service.generate(
            query=filters["query"],
            primary=primary,
            graph=graph,
            cypher=cypher,
            grep=grep,
        )

        # Сохраняем в context — увидим и в trace_input/output, и в API response
        # (через _build_result, если захотим).
        self.context["answer"] = answer
        self.context["_answer_prompt"] = prompt
        self.context["_answer_sources"] = sources_files
        # В sources показываем все источники с file_path (primary/graph/grep).
        # Cypher-row'ы пропускаем — у них нет файлов, только агрегаты.
        self.context["sources"] = [
            {
                "name": c.get("name"),
                "file": c.get("file"),
                "type": c.get("type"),
                "source": c.get("source"),
            }
            for c in chunks
            if c.get("source") in ("primary", "graph", "grep")
        ]

        # Опциональный дамп промпта + ответа на диск — для офлайн-разбора
        # без захода в Langfuse. Включается через env RAG_DUMP_PROMPT=1
        # или RAG_DUMP_PROMPT=/path/to/dir.
        _maybe_dump_prompt(prompt=prompt, answer=answer, query=filters.get("query") or "")

    @staticmethod
    def trace_input(context, filters):
        chunks = context.get("context", [])
        prompt = context.get("_answer_prompt") or ""
        sources_files = context.get("_answer_sources") or []

        # Аккуратно режем prompt для UI Langfuse — на диске лежит целиком,
        # см. RAG_DUMP_PROMPT.
        truncated = prompt
        was_truncated = False
        if len(truncated) > _PROMPT_TRACE_LIMIT:
            truncated = truncated[:_PROMPT_TRACE_LIMIT] + "\n\n…[truncated for trace UI]"
            was_truncated = True

        return {
            "query": filters.get("query"),
            "primary_chunks": sum(1 for c in chunks if c.get("source") == "primary"),
            "graph_chunks": sum(1 for c in chunks if c.get("source") == "graph"),
            "cypher_chunks": sum(1 for c in chunks if c.get("source") == "cypher"),
            "grep_chunks": sum(1 for c in chunks if c.get("source") == "grep"),
            "sources_files": sources_files,
            "prompt_chars": len(prompt),
            "prompt_truncated": was_truncated,
            "prompt": truncated,
        }

    @staticmethod
    def trace_output(context):
        answer = context.get("answer") or ""
        return {
            "answer_length": len(answer),
            "sources": len(context.get("sources") or []),
            "answer": answer,  # полный ответ
        }


def _maybe_dump_prompt(*, prompt: str, answer: str, query: str) -> None:
    """
    Дамп промпта + ответа на диск, если задан ``RAG_DUMP_PROMPT``.

    Значения env:
    - ``""`` / unset / ``"0"`` — выключено.
    - ``"1"`` / ``"true"`` — включить, пишем в ``./logs/prompts/``.
    - любая другая строка — трактуется как путь к директории.
    """
    flag = (os.getenv("RAG_DUMP_PROMPT") or "").strip()
    if flag in ("", "0", "false", "False"):
        return

    if flag in ("1", "true", "True"):
        target_dir = Path("logs") / "prompts"
    else:
        target_dir = Path(flag)

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = target_dir / f"prompt_{ts}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Query\n\n{query}\n\n")
            f.write(f"# Prompt ({len(prompt)} chars)\n\n{prompt}\n\n")
            f.write(f"# Answer ({len(answer)} chars)\n\n{answer}\n")
        logger.info(f"[generate_answer] prompt dumped to {path}")
    except Exception as e:
        logger.warning(f"[generate_answer] dump failed: {e}")
