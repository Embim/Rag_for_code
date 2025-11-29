"""
Business Agent - handles questions from business users.

Provides step-by-step instructions with UI screenshots for common tasks.

Features:
- Understands natural language questions about "how to do X"
- Finds relevant UI flows in the indexed React application
- Generates step-by-step instructions with annotated screenshots
- Supports Russian and English

Usage:
    agent = BusinessAgent(
        screenshot_service=screenshot_service,
        retriever=code_retriever,
        llm_client=openai_client
    )
    
    result = await agent.answer(
        "Как добавить нового клиента в систему?"
    )
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from ..logger import get_logger

logger = get_logger(__name__)


class TaskType(Enum):
    """Types of business tasks."""
    CREATE = "create"      # Add/create something
    VIEW = "view"          # View/show something
    EDIT = "edit"          # Edit/modify something
    DELETE = "delete"      # Delete/remove something
    SEARCH = "search"      # Find/search something
    NAVIGATE = "navigate"  # How to get to X
    UNKNOWN = "unknown"


@dataclass
class UIStep:
    """Single step in a UI workflow."""
    number: int
    action: str  # click, fill, select, navigate, etc.
    description: str  # Human-readable description
    selector: Optional[str] = None  # CSS selector for the element
    route: Optional[str] = None  # Route to navigate to
    value: Optional[str] = None  # Value to enter (for fill actions)
    screenshot: Optional[bytes] = None  # Screenshot of this step


@dataclass
class UIWorkflow:
    """Complete UI workflow with steps and screenshots."""
    task: str
    task_type: TaskType
    steps: List[UIStep] = field(default_factory=list)
    total_time_estimate: str = ""
    prerequisites: List[str] = field(default_factory=list)


class BusinessAgent:
    """
    Agent for answering business user questions about UI operations.
    
    Uses:
    - Code retriever to find relevant React components
    - Screenshot service to capture UI
    - LLM to generate instructions
    """
    
    # Task detection patterns
    TASK_PATTERNS = {
        TaskType.CREATE: [
            r'как (добавить|создать|завести)',
            r'how (to|do I) (add|create|make|register)',
            r'(добавление|создание|регистрация)',
            r'(adding|creating|registration)',
        ],
        TaskType.VIEW: [
            r'как (посмотреть|увидеть|найти|открыть)',
            r'где (находится|посмотреть|найти)',
            r'how (to|do I) (view|see|find|open|show)',
            r'where (is|can I find)',
        ],
        TaskType.EDIT: [
            r'как (изменить|редактировать|обновить|поменять)',
            r'how (to|do I) (edit|change|update|modify)',
        ],
        TaskType.DELETE: [
            r'как (удалить|убрать|отменить)',
            r'how (to|do I) (delete|remove|cancel)',
        ],
        TaskType.SEARCH: [
            r'как (искать|найти)',
            r'how (to|do I) search',
            r'(поиск|фильтр)',
        ],
        TaskType.NAVIGATE: [
            r'как (попасть|перейти|зайти)',
            r'где (находится|расположен)',
            r'how (to|do I) (get to|navigate|go to)',
            r'where is',
        ],
    }
    
    # Common entity mappings (Russian -> English component names)
    ENTITY_MAPPING = {
        'клиент': 'customer',
        'клиента': 'customer',
        'пользователь': 'user',
        'пользователя': 'user',
        'заказ': 'order',
        'заказа': 'order',
        'товар': 'product',
        'товара': 'product',
        'продукт': 'product',
        'продукта': 'product',
        'категория': 'category',
        'категории': 'category',
        'скидка': 'discount',
        'скидку': 'discount',
        'промокод': 'promo',
        'промокода': 'promo',
        'отчет': 'report',
        'отчёт': 'report',
        'настройки': 'settings',
        'профиль': 'profile',
    }
    
    def __init__(
        self,
        screenshot_service=None,
        retriever=None,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize business agent.
        
        Args:
            screenshot_service: ScreenshotService for capturing UI
            retriever: CodeRetriever for finding relevant components
            llm_client: OpenAI-compatible client for LLM calls
            config: Additional configuration
        """
        self.screenshot_service = screenshot_service
        self.retriever = retriever
        self.llm = llm_client
        self.config = config or {}
        
        # Route templates based on common patterns
        self.route_templates = {
            ('create', 'customer'): ['/customers/new', '/customers/create', '/customer/add'],
            ('create', 'order'): ['/orders/new', '/orders/create', '/checkout'],
            ('create', 'product'): ['/products/new', '/admin/products/create'],
            ('view', 'customer'): ['/customers', '/customers/list'],
            ('view', 'order'): ['/orders', '/orders/list'],
            ('view', 'product'): ['/products', '/catalog'],
            ('edit', 'customer'): ['/customers/:id/edit', '/customer/edit'],
            ('edit', 'product'): ['/products/:id/edit', '/admin/products/edit'],
            ('search', 'customer'): ['/customers?search=', '/customers/search'],
            ('search', 'order'): ['/orders?search=', '/orders/search'],
            ('settings', 'profile'): ['/settings', '/profile', '/account'],
        }
    
    def detect_task_type(self, question: str) -> TaskType:
        """Detect the type of task from the question."""
        question_lower = question.lower()
        
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return task_type
        
        return TaskType.UNKNOWN
    
    def extract_entity(self, question: str) -> Optional[str]:
        """Extract the main entity from the question."""
        question_lower = question.lower()
        
        # Check entity mappings
        for ru_word, en_word in self.ENTITY_MAPPING.items():
            if ru_word in question_lower:
                return en_word
        
        # Try to extract noun after "как добавить/создать..."
        patterns = [
            r'как (?:добавить|создать|изменить|удалить|найти)\s+(\w+)',
            r'how to (?:add|create|edit|delete|find)\s+(?:a\s+)?(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)
        
        return None
    
    async def find_relevant_routes(
        self,
        task_type: TaskType,
        entity: str
    ) -> List[str]:
        """Find relevant routes for the task."""
        routes = []
        
        # Check templates first
        key = (task_type.value, entity)
        if key in self.route_templates:
            routes.extend(self.route_templates[key])
        
        # Search in code if retriever available
        if self.retriever:
            try:
                # Search for React components/routes related to entity
                search_queries = [
                    f"{entity} form component",
                    f"{entity} page route",
                    f"create {entity} modal",
                    f"{entity} list table",
                ]
                
                for query in search_queries[:2]:  # Limit to avoid too many searches
                    result = self.retriever.search(query)
                    
                    for node in result.primary_nodes[:3]:
                        # Extract route from component
                        code = node.get('code', '')
                        route_matches = re.findall(
                            r'["\']/([\w/-]+)["\']',
                            code
                        )
                        routes.extend([f'/{r}' for r in route_matches if entity in r.lower()])
                        
            except Exception as e:
                logger.warning(f"Error searching for routes: {e}")
        
        # Remove duplicates and sort by relevance
        unique_routes = list(dict.fromkeys(routes))
        return unique_routes[:5]
    
    async def generate_steps_with_llm(
        self,
        question: str,
        task_type: TaskType,
        entity: str,
        routes: List[str],
        ui_structure: Optional[str] = None
    ) -> List[UIStep]:
        """Use LLM to generate step-by-step instructions."""
        if not self.llm:
            # Return default steps without LLM
            return self._generate_default_steps(task_type, entity, routes)
        
        prompt = f"""Ты помощник для бизнес-пользователей. Сгенерируй пошаговую инструкцию.

ВОПРОС: {question}

ТИП ЗАДАЧИ: {task_type.value}
СУЩНОСТЬ: {entity}
ДОСТУПНЫЕ РОУТЫ: {', '.join(routes) if routes else 'не найдены'}

{f'СТРУКТУРА UI: {ui_structure}' if ui_structure else ''}

ЗАДАЧА: Создай инструкцию из 3-7 шагов. Для каждого шага укажи:
1. Номер шага
2. Действие (click, fill, select, navigate)
3. Описание на русском языке
4. CSS селектор элемента (если применимо)
5. Роут (если нужна навигация)

Формат ответа (JSON):
[
  {{"number": 1, "action": "navigate", "description": "...", "route": "/..."}},
  {{"number": 2, "action": "click", "description": "...", "selector": "button.primary"}},
  ...
]

Отвечай ТОЛЬКО JSON массивом, без пояснений.
"""

        try:
            response = await self.llm.chat.completions.create(
                model=self.config.get('model', 'deepseek/deepseek-r1:free'),
                messages=[
                    {"role": "system", "content": "Ты генерируешь UI инструкции в JSON формате."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            import json
            
            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                steps_data = json.loads(json_match.group())
                
                return [
                    UIStep(
                        number=step.get('number', i + 1),
                        action=step.get('action', 'click'),
                        description=step.get('description', ''),
                        selector=step.get('selector'),
                        route=step.get('route'),
                        value=step.get('value')
                    )
                    for i, step in enumerate(steps_data)
                ]
                
        except Exception as e:
            logger.error(f"LLM step generation failed: {e}")
        
        return self._generate_default_steps(task_type, entity, routes)
    
    def _generate_default_steps(
        self,
        task_type: TaskType,
        entity: str,
        routes: List[str]
    ) -> List[UIStep]:
        """Generate default steps without LLM."""
        steps = []
        
        if task_type == TaskType.CREATE:
            route = routes[0] if routes else f'/{entity}s/new'
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description=f"Перейдите в раздел создания {entity}",
                    route=route
                ),
                UIStep(
                    number=2,
                    action="fill",
                    description="Заполните обязательные поля формы",
                    selector="form input:first-of-type"
                ),
                UIStep(
                    number=3,
                    action="click",
                    description="Нажмите кнопку 'Сохранить' или 'Создать'",
                    selector="button[type=submit], .btn-primary, .save-btn"
                ),
            ]
        elif task_type == TaskType.VIEW:
            route = routes[0] if routes else f'/{entity}s'
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description=f"Перейдите в раздел '{entity}'",
                    route=route
                ),
                UIStep(
                    number=2,
                    action="click",
                    description="Выберите нужную запись из списка",
                    selector="table tr, .list-item, .card"
                ),
            ]
        elif task_type == TaskType.EDIT:
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description=f"Найдите и откройте нужный {entity}",
                    route=routes[0] if routes else f'/{entity}s'
                ),
                UIStep(
                    number=2,
                    action="click",
                    description="Нажмите кнопку 'Редактировать'",
                    selector=".edit-btn, button[aria-label='Edit'], .pencil-icon"
                ),
                UIStep(
                    number=3,
                    action="fill",
                    description="Измените необходимые поля",
                    selector="form input"
                ),
                UIStep(
                    number=4,
                    action="click",
                    description="Сохраните изменения",
                    selector="button[type=submit], .save-btn"
                ),
            ]
        elif task_type == TaskType.DELETE:
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description=f"Найдите и откройте нужный {entity}",
                    route=routes[0] if routes else f'/{entity}s'
                ),
                UIStep(
                    number=2,
                    action="click",
                    description="Нажмите кнопку 'Удалить'",
                    selector=".delete-btn, button[aria-label='Delete'], .trash-icon"
                ),
                UIStep(
                    number=3,
                    action="click",
                    description="Подтвердите удаление в диалоговом окне",
                    selector=".confirm-btn, .modal button.primary"
                ),
            ]
        elif task_type == TaskType.SEARCH:
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description=f"Перейдите в раздел '{entity}'",
                    route=routes[0] if routes else f'/{entity}s'
                ),
                UIStep(
                    number=2,
                    action="fill",
                    description="Введите поисковый запрос",
                    selector="input[type=search], .search-input, input[placeholder*='Поиск']"
                ),
                UIStep(
                    number=3,
                    action="click",
                    description="Нажмите 'Найти' или Enter",
                    selector="button[type=submit], .search-btn"
                ),
            ]
        else:
            steps = [
                UIStep(
                    number=1,
                    action="navigate",
                    description="Откройте главное меню",
                    selector=".menu, nav, .sidebar"
                ),
                UIStep(
                    number=2,
                    action="click",
                    description=f"Найдите и выберите раздел '{entity}'",
                    selector=f"a[href*='{entity}'], .menu-item"
                ),
            ]
        
        return steps
    
    async def capture_step_screenshots(
        self,
        steps: List[UIStep]
    ) -> List[UIStep]:
        """Capture screenshots for each step."""
        if not self.screenshot_service:
            logger.warning("Screenshot service not available")
            return steps
        
        for step in steps:
            try:
                highlight = [step.selector] if step.selector else []
                
                if step.route:
                    screenshot = await self.screenshot_service.capture_route(
                        route=step.route,
                        highlight_selectors=highlight
                    )
                else:
                    # Capture current page with highlighting
                    screenshot = await self.screenshot_service.capture_route(
                        route="",  # Stay on current page
                        highlight_selectors=highlight
                    )
                
                step.screenshot = screenshot
                
            except Exception as e:
                logger.warning(f"Could not capture screenshot for step {step.number}: {e}")
        
        return steps
    
    async def answer(
        self,
        question: str,
        include_screenshots: bool = True
    ) -> UIWorkflow:
        """
        Answer a business user question.
        
        Args:
            question: Natural language question
            include_screenshots: Whether to capture screenshots
            
        Returns:
            UIWorkflow with steps and optionally screenshots
        """
        # Detect task type and entity
        task_type = self.detect_task_type(question)
        entity = self.extract_entity(question) or "item"
        
        logger.info(f"Business question: {question}")
        logger.info(f"Detected task: {task_type.value}, entity: {entity}")
        
        # Find relevant routes
        routes = await self.find_relevant_routes(task_type, entity)
        
        # Generate steps
        steps = await self.generate_steps_with_llm(
            question=question,
            task_type=task_type,
            entity=entity,
            routes=routes
        )
        
        # Capture screenshots if enabled and service available
        if include_screenshots and self.screenshot_service:
            steps = await self.capture_step_screenshots(steps)
        
        # Build workflow
        workflow = UIWorkflow(
            task=question,
            task_type=task_type,
            steps=steps,
            total_time_estimate=f"~{len(steps)} минут",
            prerequisites=[
                "Вы должны быть авторизованы в системе",
                f"У вас должны быть права на работу с '{entity}'"
            ]
        )
        
        return workflow
    
    def format_for_telegram(self, workflow: UIWorkflow) -> str:
        """Format workflow for Telegram message."""
        lines = [
            f"📋 **Инструкция**\n",
            f"Задача: {workflow.task}",
            f"Время: {workflow.total_time_estimate}",
            ""
        ]
        
        if workflow.prerequisites:
            lines.append("**Предусловия:**")
            for prereq in workflow.prerequisites:
                lines.append(f"• {prereq}")
            lines.append("")
        
        lines.append("**Шаги:**\n")
        
        for step in workflow.steps:
            emoji = {
                'navigate': '🔗',
                'click': '👆',
                'fill': '✏️',
                'select': '📝',
            }.get(step.action, '▪️')
            
            lines.append(f"{step.number}. {emoji} {step.description}")
            
            if step.route:
                lines.append(f"   📍 Адрес: `{step.route}`")
        
        lines.append("\n✅ Готово!")
        
        return '\n'.join(lines)

