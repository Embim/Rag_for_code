from ..base.task import SearchTask
from ..services import QualityService


class CheckQualityTask(SearchTask):
    """
    Repeatable. Оценивает качество найденного контекста через LLM.

    context["quality_score"] ← float.
    context["quality_feedback"] ← str.
    """

    dependencies = []

    def run(self, filters):
        service = QualityService(llm=self.executor.quality_llm)
        score, feedback = service.assess(
            query=filters["query"],
            chunks=self.context.get("context", []),
        )
        self.context["quality_score"] = score
        self.context["quality_feedback"] = feedback

    @staticmethod
    def trace_input(context, filters):
        return {
            "query": filters.get("query", ""),
            "context_chunks": len(context.get("context", [])),
        }

    @staticmethod
    def trace_output(context):
        return {
            "score": context.get("quality_score"),
            "feedback": context.get("quality_feedback"),
        }
