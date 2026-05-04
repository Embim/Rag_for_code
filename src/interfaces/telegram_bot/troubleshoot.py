"""
Troubleshooting Assistant for Code RAG System.

Helps diagnose and fix common problems:
- Neo4j connection issues
- Weaviate errors
- Parser failures
- Slow performance
- Index problems
"""

from typing import Dict, Any, List
import asyncio

from src.infra.logger import get_logger


logger = get_logger(__name__)


class TroubleshootingAssistant:
    """
    Helps users diagnose and fix system problems.

    Provides:
    - Problem detection
    - Step-by-step diagnostics
    - Suggested solutions
    """

    def __init__(self, neo4j_client, weaviate_indexer):
        """
        Initialize troubleshooter.

        Args:
            neo4j_client: Neo4j client for health checks
            weaviate_indexer: Weaviate client for health checks
        """
        self.neo4j = neo4j_client
        self.weaviate = weaviate_indexer

    async def diagnose(self, problem_description: str) -> str:
        """
        Diagnose a problem and provide solutions.

        Args:
            problem_description: User's description of the problem

        Returns:
            Diagnosis and suggested solutions
        """
        problem_lower = problem_description.lower()

        # Detect problem type
        if 'neo4j' in problem_lower or 'граф' in problem_lower:
            return await self._diagnose_neo4j()

        elif 'weaviate' in problem_lower or 'вектор' in problem_lower:
            return await self._diagnose_weaviate()

        elif 'парсер' in problem_lower or 'parser' in problem_lower:
            return self._diagnose_parser(problem_description)

        elif 'медленн' in problem_lower or 'slow' in problem_lower:
            return self._diagnose_performance()

        elif 'индекс' in problem_lower or 'index' in problem_lower:
            return await self._diagnose_index()

        else:
            # General diagnostics
            return await self._general_diagnostics()

    async def _diagnose_neo4j(self) -> str:
        """Diagnose Neo4j connection issues."""
        try:
            # Try to connect
            self.neo4j.execute_cypher("RETURN 1")
            return (
                "✅ **Neo4j работает нормально**\n\n"
                "Соединение успешно установлено."
            )

        except Exception as e:
            error_msg = str(e).lower()

            if 'connection refused' in error_msg or 'could not connect' in error_msg:
                return """
❌ **Neo4j не отвечает**

**Проблема:** Не удается подключиться к Neo4j

**Проверьте:**
1. Neo4j запущен?
   ```bash
   docker ps | grep neo4j
   # или
   systemctl status neo4j
   ```

2. Правильный порт в config (default: 7687)?
   `NEO4J_URI=bolt://localhost:7687`

3. Firewall не блокирует?

**Решение:**
```bash
# Запустить Neo4j через Docker
docker run -d \\
  --name neo4j \\
  -p 7474:7474 -p 7687:7687 \\
  -e NEO4J_AUTH=neo4j/your_password \\
  neo4j:latest
```
                """

            elif 'authentication failed' in error_msg:
                return """
❌ **Ошибка аутентификации Neo4j**

**Проблема:** Неверный логин/пароль

**Решение:**
1. Проверьте .env файл:
   ```
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

2. Или сбросьте пароль:
   ```bash
   docker exec -it neo4j neo4j-admin set-initial-password new_password
   ```
                """

            else:
                return f"""
❌ **Neo4j ошибка**

**Детали:** {str(e)}

**Общие решения:**
1. Перезапустите Neo4j
2. Проверьте логи: `docker logs neo4j`
3. Проверьте конфигурацию
                """

    async def _diagnose_weaviate(self) -> str:
        """Diagnose Weaviate issues."""
        try:
            # Try to query
            self.weaviate.client.schema.get()
            return (
                "✅ **Weaviate работает нормально**\n\n"
                "Соединение успешно установлено."
            )

        except Exception as e:
            error_msg = str(e).lower()

            if 'connection' in error_msg:
                return """
❌ **Weaviate не отвечает**

**Проверьте:**
1. Weaviate запущен?
   ```bash
   docker ps | grep weaviate
   ```

2. Правильный URL в config?
   `WEAVIATE_URL=http://localhost:8080`

**Решение:**
```bash
# Запустить Weaviate через Docker
docker run -d \\
  --name weaviate \\
  -p 8080:8080 \\
  -e QUERY_DEFAULTS_LIMIT=25 \\
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \\
  semitechnologies/weaviate:latest
```
                """

            else:
                return f"""
❌ **Weaviate ошибка**

**Детали:** {str(e)}

**Решения:**
1. Перезапустите Weaviate
2. Проверьте логи: `docker logs weaviate`
3. Проверьте схему: индекс создан?
                """

    def _diagnose_parser(self, description: str) -> str:
        """Diagnose parser issues."""
        desc_lower = description.lower()

        if 'react' in desc_lower or 'jsx' in desc_lower or 'tsx' in desc_lower:
            return """
🔧 **Проблемы с React Parser**

**Если babel parser не работает:**

1. Проверьте Node.js:
   ```bash
   node --version  # должно быть >= 16
   ```

2. Установите зависимости:
   ```bash
   cd src/code_rag/parsers
   npm install
   ```

3. Fallback на regex:
   - Автоматически включается если babel недоступен
   - Менее точный, но работает везде

**Логи:**
Смотрите в outputs/pipeline.log для деталей
            """

        elif 'python' in desc_lower or 'django' in desc_lower or 'fastapi' in desc_lower:
            return """
🔧 **Проблемы с Python Parser**

**Возможные причины:**
1. Синтаксические ошибки в коде
   - Парсер пропускает файлы с ошибками
   - Проверьте: `python -m py_compile your_file.py`

2. Нестандартный синтаксис
   - Декораторы без скобок
   - F-strings в старом Python

**Решение:**
- Проверьте версию Python (>=3.8)
- Посмотрите warnings в логах
            """

        else:
            return """
🔧 **Общие проблемы парсеров**

**Checklist:**
✓ Файлы не игнорируются (.ragignore)?
✓ Правильные расширения (.py, .tsx, .jsx)?
✓ Нет permission errors?

**Debugging:**
```bash
# Проверить что файлы найдены
ls -la data/repos/your_repo

# Посмотреть логи парсинга
tail -f outputs/pipeline.log | grep -i parse
```
            """

    def _diagnose_performance(self) -> str:
        """Diagnose performance issues."""
        return """
⚡ **Оптимизация производительности**

**Медленный поиск?**

1. **Multi-hop слишком глубокий:**
   - Уменьшите `max_hops` в config (default: 4 → 2)
   - Включите `early_stopping` (уже включен по умолчанию)

2. **Слишком большой граф:**
   - Используйте scope detection (автоматически)
   - Фильтруйте по типу узлов
   - Ограничьте repositories в запросе

3. **Cache не эффективен:**
   - Проверьте stats: `/stats`
   - Hit rate < 30%? Увеличьте cache size

**Рекомендуемые настройки:**
```yaml
# config/profiles/code.yaml
search:
  top_k_dense: 10  # было 15
  top_k_bm25: 15   # было 25

features:
  max_hops: 2  # было 4
  enable_multi_hop: true
```

**GPU для embeddings:**
```yaml
embedding:
  device: "cuda"  # вместо "cpu"
```
        """

    async def _diagnose_index(self) -> str:
        """Diagnose index issues."""
        return """
📇 **Проблемы с индексом**

**Индекс пустой или неполный?**

1. **Проверьте репозитории:**
   ```python
   # В Python shell
   from src.core.repo_loader import RepositoryLoader
   loader = RepositoryLoader()
   repos = list(loader.repos_dir.glob('*'))
   print(f"Found {len(repos)} repos")
   ```

2. **Переиндексируйте:**
   ```bash
   # Удалите старый индекс
   # WARNING: Это удалит все данные!
   docker exec -it neo4j cypher-shell -u neo4j -p your_password \\
     "MATCH (n) DETACH DELETE n"

   # Запустите индексацию заново
   python -m src.pipeline.index_repos
   ```

3. **Проверьте размер индекса:**
   ```cypher
   // В Neo4j Browser (localhost:7474)
   MATCH (n) RETURN labels(n), count(n)
   ```

**Должно быть:**
- Repository nodes
- File nodes
- Function/Class/Component nodes
- Relationships между ними
        """

    async def _general_diagnostics(self) -> str:
        """Run general system diagnostics."""
        results = []

        # Check Neo4j
        try:
            self.neo4j.execute_cypher("RETURN 1")
            results.append("✅ Neo4j: OK")
        except Exception as e:
            results.append(f"❌ Neo4j: {str(e)[:50]}")

        # Check Weaviate
        try:
            self.weaviate.client.schema.get()
            results.append("✅ Weaviate: OK")
        except Exception as e:
            results.append(f"❌ Weaviate: {str(e)[:50]}")

        return (
            "🔍 **Общая диагностика:**\n\n" +
            "\n".join(results) +
            "\n\nИспользуйте /troubleshoot с описанием конкретной проблемы для детальной помощи."
        )
