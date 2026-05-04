from .link_apis import LinkApisTask


class RelinkApisTask(LinkApisTask):
    """
    Перепривязка API-вызовов поверх уже существующего графа в Neo4j.

    Логика идентична LinkApisTask, но без зависимостей: подразумевается,
    что граф уже собран и трогать парсинг/сборку не нужно.
    """

    dependencies = []
