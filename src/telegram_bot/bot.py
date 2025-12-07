"""
Telegram Bot for Code RAG System.

Features:
1. Q&A - Ask questions about codebase and get answers with code blocks
2. Troubleshooting - Help diagnose and fix system issues
3. Graph Visualization - Generate Mermaid diagrams showing code relationships
4. Repository Management - Add/remove/reindex repositories
5. Traceback Analysis - Parse and analyze Python tracebacks
6. Logging - Track all questions and answers

Usage:
    python -m src.telegram_bot.bot
"""

import os
import logging
import time
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from ..code_rag.retrieval import CodeRetriever, SearchStrategy, ScopeDetector
from ..code_rag.graph import Neo4jClient, WeaviateIndexer
from ..agents import (
    CodeExplorerAgent,
    AgentConfig,
    QueryOrchestrator,
    AgentCache,
    CacheConfig,
)
from ..agents.tools import (
    SemanticSearchTool,
    ExactSearchTool,
    GetEntityDetailsTool,
    GetRelatedEntitiesTool,
    ListFilesTool,
    ReadFileTool,
    GrepTool,
    GetGraphPathTool,
)
from .troubleshoot import TroubleshootingAssistant
from .visualizer import MermaidDiagramGenerator
from .logging_db import BotDatabase, ChatLog, get_bot_database
from ..analyzers import TracebackAnalyzer
from ..agents.business_agent import BusinessAgent
from ..logger import get_logger
from pathlib import Path


logger = get_logger(__name__)


class CodeRAGBot:
    """
    Telegram bot for Code RAG system.

    Provides:
    - Natural language Q&A about code
    - Troubleshooting help
    - Visual diagrams
    - Repository management
    """

    def __init__(
        self,
        token: str,
        neo4j_client: Neo4jClient,
        weaviate_indexer: WeaviateIndexer,
        openrouter_api_key: Optional[str] = None,
        enable_agents: bool = True
    ):
        """
        Initialize bot.

        Args:
            token: Telegram bot token
            neo4j_client: Neo4j client for graph operations
            weaviate_indexer: Weaviate client for search
            openrouter_api_key: API key for agent LLM calls (optional)
            enable_agents: Enable intelligent agent system
        """
        self.token = token
        self.neo4j = neo4j_client
        self.weaviate = weaviate_indexer

        # Initialize components
        self.retriever = CodeRetriever(neo4j_client, weaviate_indexer)
        self.scope_detector = ScopeDetector()
        self.troubleshooter = TroubleshootingAssistant(neo4j_client, weaviate_indexer)
        self.visualizer = MermaidDiagramGenerator(neo4j_client)
        
        # Initialize logging database
        self.db = get_bot_database()
        logger.info("✅ Bot database initialized")
        
        # Initialize OpenAI client for LLM (via OpenRouter)
        self.llm_client = None
        if openrouter_api_key:
            self.llm_client = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        
        # Initialize traceback analyzer
        self.traceback_analyzer = TracebackAnalyzer(
            neo4j_client=neo4j_client,
            weaviate_indexer=weaviate_indexer,
            llm_client=self.llm_client,
            config={'model': os.getenv('ANALYSIS_MODEL', 'tngtech/tng-r1t-chimera:free')}
        )
        logger.info("✅ Traceback analyzer initialized")

        # Initialize business agent (for business users)
        self.business_agent = BusinessAgent(
            screenshot_service=None,  # Will be set later if configured
            retriever=self.retriever,
            llm_client=self.llm_client,
            config={'model': os.getenv('ANALYSIS_MODEL', 'tngtech/tng-r1t-chimera:free')}
        )
        logger.info("✅ Business agent initialized")

        # Initialize agent system (Phase 6)
        self.enable_agents = enable_agents and openrouter_api_key
        self.orchestrator = None

        if self.enable_agents:
            try:
                # Initialize tools
                repos_dir = Path("data/repos")
                tools = [
                    SemanticSearchTool(self.retriever),
                    ExactSearchTool(neo4j_client),
                    GetEntityDetailsTool(neo4j_client),
                    GetRelatedEntitiesTool(neo4j_client),
                    ListFilesTool(repos_dir),
                    ReadFileTool(repos_dir),
                    GrepTool(neo4j_client),
                    GetGraphPathTool(neo4j_client),
                ]

                # Initialize Code Explorer Agent
                agent_config = AgentConfig(
                    max_iterations=int(os.getenv('AGENT_MAX_ITERATIONS', '15')),
                    timeout_seconds=float(os.getenv('AGENT_TIMEOUT', '180')),
                    temperature=0.1,
                    model=os.getenv('CODE_EXPLORER_MODEL', 'tngtech/tng-r1t-chimera:free'),
                )

                code_explorer = CodeExplorerAgent(
                    tools=tools,
                    api_key=openrouter_api_key,
                    config=agent_config,
                )

                # Initialize cache
                cache_config = CacheConfig(
                    enabled=True,
                    backend="memory",
                    query_ttl=86400,
                    tool_result_ttl=3600,
                )
                self.agent_cache = AgentCache(cache_config)

                # Initialize orchestrator
                self.orchestrator = QueryOrchestrator(
                    code_explorer=code_explorer,
                    api_key=openrouter_api_key,
                    model=os.getenv('ORCHESTRATOR_MODEL', 'deepseek/deepseek-r1:free'),
                )

                logger.info("✅ Agent system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize agents: {e}")
                self.enable_agents = False

        # Build application
        self.app = Application.builder().token(token).build()

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register command and message handlers."""
        # Commands
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("search", self.search_command))
        self.app.add_handler(CommandHandler("ask", self.ask_command))  # Agent-powered Q&A
        self.app.add_handler(CommandHandler("analyze", self.analyze_command))  # Traceback analysis
        self.app.add_handler(CommandHandler("troubleshoot", self.troubleshoot_command))
        self.app.add_handler(CommandHandler("visualize", self.visualize_command))
        self.app.add_handler(CommandHandler("repos", self.repos_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("history", self.history_command))
        self.app.add_handler(CommandHandler("role", self.role_command))
        self.app.add_handler(CommandHandler("guide", self.guide_command))  # Business user guide

        # Message handler for Q&A
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_question)
        )

        # Callback queries (for inline buttons)
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        # Register user
        user = update.effective_user
        self.db.get_or_create_user(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name
        )
        
        welcome_message = """
👋 Привет! Я бот для работы с Code RAG системой.

Я помогу вам:
• 💬 Отвечать на вопросы о кодовой базе
• 🔧 Диагностировать проблемы с системой  
• 🐛 Анализировать traceback ошибок
• 📊 Визуализировать связи в коде
• 📦 Управлять репозиториями

**Просто задайте вопрос** или используйте команды:
/help - Список команд
/search - Поиск по коду
/analyze - Анализ traceback ошибки
/troubleshoot - Помощь с проблемами
/visualize - Создать диаграмму
/repos - Управление репозиториями
/stats - Статистика системы

Спрашивайте что угодно о коде! 🚀
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        agent_status = "🤖 **ВКЛЮЧЕНЫ**" if self.enable_agents else "⏸️ **ВЫКЛЮЧЕНЫ**"

        help_text = f"""
📚 **Доступные команды:**

**Поиск и вопросы:**
/search <запрос> - Быстрый поиск по кодовой базе
/ask <вопрос> - Умный агент (исследует код итеративно) {agent_status}
Просто напишите вопрос - получите ответ

**Анализ ошибок:**
/analyze <traceback> - Анализ Python traceback
Или просто скиньте traceback - бот распознает автоматически

**Для бизнес-пользователей:**
/guide <вопрос> - Пошаговая инструкция "как сделать X"
- "как добавить клиента"
- "как найти заказ"
- "где настройки"

**Troubleshooting:**
/troubleshoot - Диагностика проблем
- "Neo4j не подключается"
- "Weaviate выдает ошибку"
- "Парсер не работает"

**Визуализация:**
/visualize <что показать> - Создать диаграмму
- "как работает корзина"
- "связи для ProductCard"
- "поток данных для заказа"

**Управление:**
/repos - Список репозиториев
/stats - Статистика индекса
/history - Ваша история запросов
/role - Профиль и роль (developer/business)

**Разница между /search и /ask:**
• /search - быстрый поиск одним запросом
• /ask - агент исследует код в несколько шагов, собирает контекст

**Примеры вопросов:**
• "Как реализована авторизация?"
• "Покажи все эндпоинты для продуктов"
• "Какие компоненты используют ProductCard?"
• "Что сломается если изменить модель Order?"
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command."""
        # Extract query from command
        query = ' '.join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text(
                "Использование: /search <ваш запрос>\n\n"
                "Пример: /search как работает корзина"
            )
            return

        # Perform search
        await self._handle_search(update, query)

    async def ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask command (agent-powered)."""
        if not self.enable_agents:
            await update.message.reply_text(
                "❌ Агенты не включены.\n\n"
                "Для включения агентов нужен OPENROUTER_API_KEY.\n"
                "Используйте /search для обычного поиска."
            )
            return

        query = ' '.join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text(
                "🤖 **Умный агент для Code RAG**\n\n"
                "Использование: /ask <ваш вопрос>\n\n"
                "Агент исследует код итеративно:\n"
                "1. Анализирует вопрос\n"
                "2. Планирует исследование\n"
                "3. Использует нужные инструменты\n"
                "4. Собирает контекст\n"
                "5. Формирует полный ответ\n\n"
                "Пример: /ask как работает процесс оформления заказа",
                parse_mode='Markdown'
            )
            return

        # Check cache first
        cached_result = self.agent_cache.get_query_result(query)
        if cached_result:
            logger.info(f"Agent cache HIT: {query[:50]}")
            await update.message.reply_text(
                f"💾 *(из кэша)*\n\n{cached_result['answer']}",
                parse_mode='Markdown'
            )
            return

        # Send "typing" indicator
        await update.message.reply_chat_action("typing")

        # Send initial status
        status_msg = await update.message.reply_text(
            "🤖 Агент начинает исследование...",
            parse_mode='Markdown'
        )

        try:
            # Run agent (non-streaming for MVP)
            # TODO: Implement streaming in future
            result = await self.orchestrator.answer_question(
                question=query,
                context={'repositories': ['all'], 'scope': 'hybrid'},
                stream=False
            )

            # Delete status message
            await status_msg.delete()

            # Format and send answer
            if result['success']:
                answer_text = self._format_agent_result(result, query)
                await self._send_long_message(update, answer_text)

                # Cache result
                self.agent_cache.set_query_result(query, result)
                self.agent_cache.add_to_semantic_cache(query)

            else:
                await update.message.reply_text(
                    f"❌ Агент не смог найти ответ:\n\n{result['answer']}",
                    parse_mode='Markdown'
                )

        except Exception as e:
            logger.error(f"Agent failed: {e}")
            await status_msg.delete()
            await update.message.reply_text(
                f"❌ Ошибка агента: {str(e)}\n\n"
                "Попробуйте:\n"
                "• Переформулировать вопрос\n"
                "• Использовать /search для обычного поиска\n"
                "• Проверить логи (outputs/pipeline.log)"
            )

    def _format_agent_result(self, result: Dict[str, Any], query: str) -> str:
        """Format agent result for Telegram."""
        question_type = result.get('classification', {}).get('type', 'Unknown')
        agent_used = result.get('agent_used', 'Unknown')

        lines = [
            f"🤖 **Ответ агента**",
            f"Вопрос: {query}",
            f"Тип: {question_type} | Агент: {agent_used}",
            ""
        ]

        # Show tool usage if available
        tool_calls = result.get('tool_calls', [])
        if tool_calls:
            lines.append(f"🔧 Инструменты: {', '.join(tool_calls)}")
            lines.append(f"⏱ Итераций: {result.get('iterations', 0)}")
            lines.append("")

        # Show answer
        lines.append(result['answer'])

        # Show if incomplete
        if not result.get('complete', True):
            lines.append("\n⚠️ Внимание: Ответ может быть неполным (превышен лимит итераций)")

        return '\n'.join(lines)
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /analyze command - analyze Python traceback.
        
        Usage:
            /analyze <paste traceback here>
            
        Or just paste a traceback as a message.
        """
        user = update.effective_user
        start_time = time.time()
        
        # Get traceback text
        traceback_text = ' '.join(context.args) if context.args else ""
        
        if not traceback_text:
            await update.message.reply_text(
                "🐛 **Анализ Traceback**\n\n"
                "Использование: /analyze <traceback>\n\n"
                "Или просто скопируйте и отправьте весь traceback как сообщение.\n\n"
                "Пример:\n"
                "```\n"
                "Traceback (most recent call last):\n"
                '  File "app.py", line 42, in process\n'
                "    result = calculate(data)\n"
                "ValueError: invalid input\n"
                "```",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_chat_action("typing")
        
        try:
            # Analyze traceback
            result = await self.traceback_analyzer.analyze(
                traceback_text,
                include_related=True,
                explain=self.llm_client is not None
            )
            
            # Format and send response
            response = self.traceback_analyzer.format_for_telegram(result)
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Log interaction
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=traceback_text[:500],  # Truncate for storage
                answer=response[:1000],
                question_type="analyze",
                response_time_ms=response_time,
                success=result.get('success', False),
                metadata={
                    'exception_type': result.get('exception', {}).get('type'),
                    'frames_matched': result.get('frames_matched', 0),
                    'frames_total': result.get('frames_total', 0),
                }
            ))
            
            await self._send_long_message(update, response)
            
        except Exception as e:
            logger.error(f"Traceback analysis failed: {e}")
            
            # Log error
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=traceback_text[:500],
                answer="",
                question_type="analyze",
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e)
            ))
            
            await update.message.reply_text(
                f"❌ Ошибка анализа: {str(e)}\n\n"
                "Убедитесь что это корректный Python traceback."
            )
    
    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command - show user's question history."""
        user = update.effective_user
        
        # Get optional limit
        limit = 10
        if context.args:
            try:
                limit = min(int(context.args[0]), 50)
            except ValueError:
                pass
        
        history = self.db.get_user_history(user.id, limit=limit)
        
        if not history:
            await update.message.reply_text(
                "📜 История пуста.\n\nЗадайте вопрос, используя /search или /ask"
            )
            return
        
        lines = [f"📜 **Ваша история** (последние {len(history)} запросов):\n"]
        
        for i, log in enumerate(history, 1):
            # Truncate question
            question = log.question[:50] + "..." if len(log.question) > 50 else log.question
            success = "✅" if log.success else "❌"
            
            lines.append(
                f"{i}. {success} [{log.question_type}] {question}\n"
                f"   ⏱ {log.response_time_ms}ms | {log.created_at}"
            )
        
        await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')
    
    async def role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /role command - view/change user role."""
        user = update.effective_user
        
        # Get current user profile
        profile = self.db.get_or_create_user(user.id, user.username)
        
        if not context.args:
            await update.message.reply_text(
                f"👤 **Ваш профиль**\n\n"
                f"ID: {profile.user_id}\n"
                f"Роль: {profile.role}\n"
                f"Запросов: {profile.total_queries}\n\n"
                f"Для смены роли: /role <developer|business>\n\n"
                f"**Роли:**\n"
                f"• `developer` - технические вопросы, traceback анализ\n"
                f"• `business` - инструкции 'как сделать', скриншоты UI",
                parse_mode='Markdown'
            )
            return
        
        new_role = context.args[0].lower()
        if new_role not in ('developer', 'business', 'admin'):
            await update.message.reply_text(
                "❌ Неверная роль. Доступны: developer, business"
            )
            return
        
        self.db.update_user_role(user.id, new_role)
        await update.message.reply_text(
            f"✅ Роль изменена на: {new_role}"
        )
    
    async def guide_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /guide command - get step-by-step instructions for business tasks.
        
        Usage:
            /guide как добавить клиента
            /guide how to create an order
        """
        user = update.effective_user
        start_time = time.time()
        
        question = ' '.join(context.args) if context.args else ""
        
        if not question:
            await update.message.reply_text(
                "📋 **Инструкции по работе с системой**\n\n"
                "Использование: /guide <ваш вопрос>\n\n"
                "**Примеры:**\n"
                "• /guide как добавить клиента\n"
                "• /guide как найти заказ\n"
                "• /guide как изменить товар\n"
                "• /guide где настройки профиля\n\n"
                "Я сгенерирую пошаговую инструкцию с описанием каждого действия.",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_chat_action("typing")
        
        try:
            # Generate workflow
            workflow = await self.business_agent.answer(
                question,
                include_screenshots=False  # Screenshots disabled for now
            )
            
            # Format response
            response = self.business_agent.format_for_telegram(workflow)
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Log interaction
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=question,
                answer=response[:1000],
                question_type="guide",
                response_time_ms=response_time,
                success=True,
                metadata={
                    'task_type': workflow.task_type.value,
                    'steps_count': len(workflow.steps)
                }
            ))
            
            await self._send_long_message(update, response)
            
        except Exception as e:
            logger.error(f"Guide generation failed: {e}")
            
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=question,
                answer="",
                question_type="guide",
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e)
            ))
            
            await update.message.reply_text(
                f"❌ Ошибка: {str(e)}\n\n"
                "Попробуйте переформулировать вопрос."
            )

    async def handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle plain text questions."""
        query = update.message.text
        
        # Check if it's a traceback (auto-detect)
        traceback_indicators = [
            'Traceback (most recent call last):',
            'File "', 
            'line ', 
            'Error:', 
            'Exception:'
        ]
        
        is_traceback = (
            'Traceback' in query and 
            any(ind in query for ind in traceback_indicators[1:])
        )
        
        if is_traceback:
            # Auto-analyze traceback
            context.args = [query]  # Pass as argument
            await self.analyze_command(update, context)
            return

        # Check if it's a troubleshooting request
        troubleshoot_keywords = [
            'не работает', 'ошибка', 'проблема', 'не могу',
            'not working', 'error', 'problem', 'issue'
        ]

        if any(kw in query.lower() for kw in troubleshoot_keywords):
            await self._handle_troubleshoot(update, query)
        else:
            await self._handle_search(update, query)

    async def _handle_search(self, update: Update, query: str):
        """Handle search query."""
        user = update.effective_user
        start_time = time.time()
        
        # Send "typing" indicator
        await update.message.reply_chat_action("typing")

        try:
            # Detect scope
            scope_hint = self.scope_detector.detect_scope(query)
            logger.info(
                f"Query scope: {scope_hint.scope.value} "
                f"(confidence={scope_hint.confidence:.2f})"
            )

            # Determine strategy
            if "как работает" in query.lower() or "how does" in query.lower():
                strategy = SearchStrategy.UI_TO_DATABASE
            elif "что сломается" in query.lower() or "impact" in query.lower():
                strategy = SearchStrategy.IMPACT_ANALYSIS
            elif "где используется" in query.lower() or "where is used" in query.lower():
                strategy = SearchStrategy.DATABASE_TO_UI
            else:
                strategy = SearchStrategy.SEMANTIC_ONLY

            # Apply scope filter
            config_override = {}
            if scope_hint.confidence > 0.5:
                config_override = self.scope_detector.apply_scope_filter(
                    scope_hint.scope,
                    {}
                )

            # Search
            result = self.retriever.search(
                query=query,
                strategy=strategy,
                config_override=config_override
            )

            # Format and send response
            response = self._format_search_result(result, query)
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Log successful search
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=query,
                answer=response[:1000],  # Truncate for storage
                question_type="search",
                response_time_ms=response_time,
                success=True,
                metadata={
                    'strategy': str(strategy),
                    'scope': scope_hint.scope.value,
                    'results_count': len(result.primary_nodes) if result.primary_nodes else 0,
                }
            ))
            
            await self._send_long_message(update, response)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            
            # Log error
            self.db.log_interaction(ChatLog(
                user_id=user.id,
                username=user.username,
                chat_id=update.effective_chat.id,
                question=query,
                answer="",
                question_type="search",
                response_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message=str(e)
            ))
            
            await update.message.reply_text(
                f"❌ Ошибка при поиске: {str(e)}\n\n"
                "Попробуйте переформулировать вопрос или используйте /troubleshoot"
            )

    async def _handle_troubleshoot(self, update: Update, query: str):
        """Handle troubleshooting request."""
        await update.message.reply_chat_action("typing")

        try:
            # Troubleshoot
            diagnosis = await self.troubleshooter.diagnose(query)

            # Format response
            response = f"🔧 **Диагностика:**\n\n{diagnosis}"
            await self._send_long_message(update, response)

        except Exception as e:
            logger.error(f"Troubleshooting failed: {e}")
            await update.message.reply_text(
                f"❌ Ошибка: {str(e)}"
            )

    def _format_search_result(
        self,
        result: Any,
        query: str
    ) -> str:
        """Format search result for Telegram."""
        if not result.primary_nodes:
            return (
                f"🔍 По запросу *\"{query}\"* ничего не найдено.\n\n"
                "Попробуйте:\n"
                "• Переформулировать вопрос\n"
                "• Использовать другие ключевые слова\n"
                "• Проверить что репозитории проиндексированы (/repos)"
            )

        # Build response
        lines = [
            f"🔍 **Результаты для:** {query}",
            f"Стратегия: {result.strategy_used}",
            f"Найдено: {len(result.primary_nodes)} основных результатов",
            ""
        ]

        # Show primary results
        for i, node in enumerate(result.primary_nodes[:5], 1):
            node_type = node.get('type', 'Unknown')
            node_name = node.get('name', 'Unknown')
            file_path = node.get('file_path', '')

            lines.append(f"{i}. **{node_name}** ({node_type})")
            if file_path:
                lines.append(f"   📁 {file_path}")

            # Show code snippet if available
            code = node.get('code', '')
            if code:
                # Truncate long code
                code_lines = code.split('\n')[:10]
                code_preview = '\n'.join(code_lines)

                lines.append("   ```")
                lines.append(code_preview)
                if len(code.split('\n')) > 10:
                    lines.append("   ... (truncated)")
                lines.append("   ```")

            lines.append("")

        # Add metadata
        lines.append(
            f"⏱ Время поиска: {result.execution_time_ms:.0f}ms | "
            f"Узлов обработано: {result.total_nodes_visited}"
        )

        return '\n'.join(lines)

    async def _send_long_message(self, update: Update, text: str):
        """Send long message, splitting if necessary."""
        # Telegram limit: 4096 characters
        MAX_LENGTH = 4000

        if len(text) <= MAX_LENGTH:
            await update.message.reply_text(text, parse_mode='Markdown')
        else:
            # Split into chunks
            parts = []
            current = ""

            for line in text.split('\n'):
                if len(current) + len(line) + 1 > MAX_LENGTH:
                    parts.append(current)
                    current = line
                else:
                    current += '\n' + line if current else line

            if current:
                parts.append(current)

            # Send parts
            for i, part in enumerate(parts):
                await update.message.reply_text(
                    f"{part}\n\n_(часть {i+1}/{len(parts)})_",
                    parse_mode='Markdown'
                )

    async def troubleshoot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /troubleshoot command."""
        problem = ' '.join(context.args) if context.args else ""

        if not problem:
            # Show common problems
            keyboard = [
                [InlineKeyboardButton("🔴 Neo4j не подключается", callback_data='ts_neo4j')],
                [InlineKeyboardButton("🔴 Weaviate ошибки", callback_data='ts_weaviate')],
                [InlineKeyboardButton("🔴 Парсер не работает", callback_data='ts_parser')],
                [InlineKeyboardButton("🔴 Медленный поиск", callback_data='ts_slow')],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "🔧 **Troubleshooting**\n\n"
                "Выберите проблему или опишите её текстом:",
                reply_markup=reply_markup
            )
        else:
            await self._handle_troubleshoot(update, problem)

    async def visualize_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /visualize command."""
        query = ' '.join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text(
                "Использование: /visualize <что показать>\n\n"
                "Примеры:\n"
                "• /visualize как работает корзина\n"
                "• /visualize связи ProductCard\n"
                "• /visualize поток данных заказа"
            )
            return

        await update.message.reply_chat_action("upload_photo")

        try:
            # Generate diagram
            diagram = await self.visualizer.generate_diagram(query)

            # Send as image
            await update.message.reply_photo(
                photo=diagram,
                caption=f"Диаграмма для: {query}"
            )

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            await update.message.reply_text(
                f"❌ Не удалось создать диаграмму: {str(e)}"
            )

    async def repos_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /repos command."""
        # TODO: Implement repository management
        await update.message.reply_text(
            "📦 **Управление репозиториями** (в разработке)\n\n"
            "Планируется:\n"
            "• Список проиндексированных репозиториев\n"
            "• Добавление нового репозитория\n"
            "• Переиндексация\n"
            "• Удаление"
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        # Get cache stats
        cache_stats = self.retriever.path_cache.stats()
        
        # Get bot usage stats
        bot_stats = self.db.get_stats(days=7)

        stats_text = f"""
📊 **Статистика системы:**

**Использование бота (7 дней):**
• Всего запросов: {bot_stats['total_queries']}
• Успешных: {bot_stats['successful_queries']} ({bot_stats['success_rate']:.1f}%)
• Среднее время: {bot_stats['avg_response_time_ms']}ms
• Активных пользователей: {bot_stats['active_users']}

**По типам запросов:**
"""
        for qtype, count in bot_stats.get('queries_by_type', {}).items():
            stats_text += f"• {qtype}: {count}\n"

        stats_text += f"""
**Path Cache (multi-hop):**
• Размер: {cache_stats['size']}/{cache_stats['max_size']}
• Всего hits: {cache_stats['total_hits']}
• Hit rate: {cache_stats['hit_rate']:.1%}

**Multi-hop Optimizer:**
• Early stopping: включен
• Max hops: {self.retriever.config.max_hops}
• Timeout: {self.retriever.config.timeout_seconds}s

**Scope Detection:**
• Поддержка русского: ✅
• LLM classification: ✅ (agents)
        """

        # Add agent stats if enabled
        if self.enable_agents and hasattr(self, 'agent_cache'):
            agent_stats = self.agent_cache.get_stats()
            if agent_stats.get('enabled'):
                stats_text += f"""

**Agent Cache:**
• Backend: {agent_stats.get('backend', 'memory')}
• Hit rate: {agent_stats.get('hit_rate', 0):.1%}
• Semantic cache: {agent_stats.get('semantic_cache_size', 0)} queries
                """

        await update.message.reply_text(stats_text, parse_mode='Markdown')

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()

        if query.data.startswith('ts_'):
            # Troubleshooting shortcuts
            problem_map = {
                'ts_neo4j': 'Neo4j не подключается',
                'ts_weaviate': 'Weaviate выдает ошибку',
                'ts_parser': 'Парсер не работает',
                'ts_slow': 'Поиск работает медленно',
            }

            problem = problem_map.get(query.data, '')
            if problem:
                await query.message.reply_text(f"🔧 Диагностирую: {problem}...")
                # Create a fake update for _handle_troubleshoot
                # Note: This is a workaround, better to refactor
                await self._handle_troubleshoot(update, problem)

    def run(self):
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        self.app.run_polling()


# Entry point
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Get token from environment
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    # Initialize clients (placeholder - should come from config)
    from ...config import load_config
    config = load_config()

    neo4j_client = Neo4jClient(config['neo4j'])
    weaviate_indexer = WeaviateIndexer(config['weaviate'])

    # Create and run bot
    bot = CodeRAGBot(token, neo4j_client, weaviate_indexer)
    bot.run()
