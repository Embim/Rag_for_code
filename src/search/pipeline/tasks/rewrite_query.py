from ..base.task import SearchTask
from ..services import RewriterService


class RewriteQueryTask(SearchTask):
    """
    Repeatable. Переписывает текущий запрос на основе фидбэка качества.

    context["current_query"] ← новый текст запроса.
    """

    dependencies = []

    def run(self, filters):
        service = RewriterService(llm=self.executor.rewrite_llm)
        new_query = service.rewrite(
            original=filters["query"],
            current=self.context.get("current_query") or filters["query"],
            feedback=self.context.get("quality_feedback", ""),
        )
        self.context["current_query"] = new_query

    @staticmethod
    def trace_input(context, filters):
        return {
            "query": context.get("current_query") or filters.get("query"),
            "feedback": context.get("quality_feedback", ""),
        }

    @staticmethod
    def trace_output(context):
        return {"new_query": context.get("current_query")}
