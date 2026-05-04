from ..base.task import SearchTask
from ..services import StrategyService


class DetectStrategyTask(SearchTask):
    """
    One-shot. Выбирает SearchStrategy по тексту запроса. В loop'е стратегия
    не меняется — пишется в context один раз.

    context["strategy"] ← SearchStrategy.
    """

    dependencies = []

    def run(self, filters):
        self.context["strategy"] = StrategyService.detect(filters["query"])

    @staticmethod
    def trace_input(context, filters):
        return {"query": filters.get("query", "")}

    @staticmethod
    def trace_output(context):
        strategy = context.get("strategy")
        return {"strategy": getattr(strategy, "value", str(strategy))}
