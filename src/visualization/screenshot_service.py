"""
Screenshot Service - captures UI screenshots using Playwright.

Designed for React applications running locally via `npm run dev`.

Features:
- Start/stop local React dev server
- Navigate to specific routes
- Take screenshots with element highlighting
- Add annotations explaining UI elements
- Support authentication (cookies/session storage)

Usage:
    service = ScreenshotService(
        project_dir="/path/to/react-app",
        base_url="http://localhost:3000"
    )
    
    await service.start()
    screenshot = await service.capture_route("/dashboard", highlight=["#save-btn"])
    await service.stop()
"""

import asyncio
import subprocess
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from io import BytesIO

from ..logger import get_logger

logger = get_logger(__name__)

# Lazy import Playwright to avoid startup cost
_playwright = None
_playwright_async = None


def _get_playwright():
    """Lazy import playwright."""
    global _playwright_async
    if _playwright_async is None:
        try:
            from playwright.async_api import async_playwright
            _playwright_async = async_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
    return _playwright_async


@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture."""
    width: int = 1280
    height: int = 720
    full_page: bool = False
    quality: int = 80  # JPEG quality (0-100)
    timeout_ms: int = 30000
    wait_for_selector: Optional[str] = None
    wait_for_load_state: str = "networkidle"  # load, domcontentloaded, networkidle


@dataclass
class Annotation:
    """UI annotation with position and text."""
    selector: str
    text: str
    color: str = "#FF0000"
    position: str = "right"  # top, right, bottom, left


@dataclass
class AuthConfig:
    """Authentication configuration for the app."""
    login_url: str = "/login"
    username_selector: str = "input[name=username]"
    password_selector: str = "input[name=password]"
    submit_selector: str = "button[type=submit]"
    username: str = ""
    password: str = ""
    # Alternative: use saved cookies/storage
    cookies_file: Optional[str] = None
    storage_state_file: Optional[str] = None


class ScreenshotService:
    """
    Service for capturing React UI screenshots with Playwright.
    
    Handles:
    - Starting local dev server (npm run dev)
    - Browser automation
    - Screenshot capture with highlighting
    - Authentication flow
    """
    
    def __init__(
        self,
        project_dir: str,
        base_url: str = "http://localhost:3000",
        start_command: str = "npm run dev",
        config: Optional[ScreenshotConfig] = None,
        auth_config: Optional[AuthConfig] = None
    ):
        """
        Initialize screenshot service.
        
        Args:
            project_dir: Path to React project directory
            base_url: Base URL of the dev server
            start_command: Command to start dev server
            config: Screenshot configuration
            auth_config: Authentication configuration
        """
        self.project_dir = Path(project_dir)
        self.base_url = base_url.rstrip('/')
        self.start_command = start_command
        self.config = config or ScreenshotConfig()
        self.auth_config = auth_config
        
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._dev_server_process = None
        self._is_started = False
        self._is_authenticated = False
    
    async def start(self, start_server: bool = True):
        """
        Start the service.
        
        Args:
            start_server: Whether to start the dev server (set False if already running)
        """
        if self._is_started:
            logger.warning("Screenshot service already started")
            return
        
        # Start dev server if needed
        if start_server:
            await self._start_dev_server()
        
        # Start Playwright
        async_playwright = _get_playwright()
        self._playwright = await async_playwright().start()
        
        # Launch browser (headless)
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        # Create context with viewport
        context_options = {
            'viewport': {
                'width': self.config.width,
                'height': self.config.height
            }
        }
        
        # Load storage state if available
        if self.auth_config and self.auth_config.storage_state_file:
            storage_path = Path(self.auth_config.storage_state_file)
            if storage_path.exists():
                context_options['storage_state'] = str(storage_path)
                self._is_authenticated = True
                logger.info("Loaded authentication state from file")
        
        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()
        
        self._is_started = True
        logger.info(f"Screenshot service started for {self.base_url}")
    
    async def _start_dev_server(self):
        """Start the React dev server."""
        if not self.project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")
        
        logger.info(f"Starting dev server in {self.project_dir}...")
        
        # Start process
        self._dev_server_process = subprocess.Popen(
            self.start_command,
            shell=True,
            cwd=str(self.project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, 'CI': 'true', 'BROWSER': 'none'}  # Prevent auto-opening browser
        )
        
        # Wait for server to be ready
        await self._wait_for_server()
        logger.info("Dev server is ready")
    
    async def _wait_for_server(self, timeout: int = 60):
        """Wait for dev server to become available."""
        import aiohttp
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, timeout=aiohttp.ClientTimeout(total=2)):
                        return  # Server is ready
            except Exception:
                await asyncio.sleep(1)
        
        raise TimeoutError(f"Dev server did not start within {timeout} seconds")
    
    async def authenticate(self, force: bool = False):
        """
        Authenticate in the application.
        
        Args:
            force: Force re-authentication even if already authenticated
        """
        if self._is_authenticated and not force:
            logger.info("Already authenticated")
            return
        
        if not self.auth_config:
            raise ValueError("AuthConfig not provided")
        
        if not self.auth_config.username or not self.auth_config.password:
            raise ValueError("Username and password required for authentication")
        
        # Navigate to login
        login_url = f"{self.base_url}{self.auth_config.login_url}"
        await self._page.goto(login_url, wait_until="networkidle")
        
        # Fill credentials
        await self._page.fill(self.auth_config.username_selector, self.auth_config.username)
        await self._page.fill(self.auth_config.password_selector, self.auth_config.password)
        
        # Submit
        await self._page.click(self.auth_config.submit_selector)
        
        # Wait for navigation
        await self._page.wait_for_load_state("networkidle")
        
        # Save storage state for future use
        if self.auth_config.storage_state_file:
            await self._context.storage_state(path=self.auth_config.storage_state_file)
            logger.info(f"Saved authentication state to {self.auth_config.storage_state_file}")
        
        self._is_authenticated = True
        logger.info("Authentication successful")
    
    async def capture_route(
        self,
        route: str,
        highlight_selectors: Optional[List[str]] = None,
        annotations: Optional[List[Annotation]] = None,
        wait_for: Optional[str] = None
    ) -> bytes:
        """
        Capture screenshot of a specific route.
        
        Args:
            route: Route to navigate to (e.g., "/dashboard")
            highlight_selectors: CSS selectors to highlight with red border
            annotations: List of annotations to add
            wait_for: Optional selector to wait for before capturing
            
        Returns:
            Screenshot as PNG bytes
        """
        if not self._is_started:
            raise RuntimeError("Service not started. Call start() first.")
        
        url = f"{self.base_url}{route}"
        logger.info(f"Capturing screenshot: {url}")
        
        # Navigate
        await self._page.goto(url, wait_until=self.config.wait_for_load_state)
        
        # Wait for specific element if provided
        if wait_for:
            await self._page.wait_for_selector(wait_for, timeout=self.config.timeout_ms)
        elif self.config.wait_for_selector:
            await self._page.wait_for_selector(
                self.config.wait_for_selector, 
                timeout=self.config.timeout_ms
            )
        
        # Add highlights
        if highlight_selectors:
            await self._add_highlights(highlight_selectors)
        
        # Take screenshot
        screenshot = await self._page.screenshot(
            full_page=self.config.full_page,
            type="png"
        )
        
        # Add annotations if provided (using Pillow)
        if annotations:
            screenshot = await self._add_annotations(screenshot, annotations)
        
        return screenshot
    
    async def _add_highlights(self, selectors: List[str]):
        """Add red border highlights to elements."""
        for selector in selectors:
            try:
                await self._page.evaluate(f"""
                    (selector) => {{
                        const el = document.querySelector(selector);
                        if (el) {{
                            el.style.outline = '3px solid red';
                            el.style.outlineOffset = '2px';
                        }}
                    }}
                """, selector)
            except Exception as e:
                logger.warning(f"Could not highlight {selector}: {e}")
    
    async def _add_annotations(
        self, 
        screenshot: bytes, 
        annotations: List[Annotation]
    ) -> bytes:
        """Add text annotations to screenshot using Pillow."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow not installed, skipping annotations")
            return screenshot
        
        # Load image
        img = Image.open(BytesIO(screenshot))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        
        # Get element positions and add annotations
        for ann in annotations:
            try:
                # Get element bounding box
                element = await self._page.query_selector(ann.selector)
                if element:
                    box = await element.bounding_box()
                    if box:
                        # Calculate annotation position
                        x, y = self._calculate_annotation_position(box, ann.position)
                        
                        # Draw background
                        text_bbox = draw.textbbox((x, y), ann.text, font=font)
                        padding = 4
                        draw.rectangle(
                            [
                                text_bbox[0] - padding,
                                text_bbox[1] - padding,
                                text_bbox[2] + padding,
                                text_bbox[3] + padding
                            ],
                            fill=ann.color
                        )
                        
                        # Draw text
                        draw.text((x, y), ann.text, fill="white", font=font)
                        
                        # Draw line to element
                        element_center = (
                            box['x'] + box['width'] / 2,
                            box['y'] + box['height'] / 2
                        )
                        draw.line(
                            [element_center, (x, y + 8)],
                            fill=ann.color,
                            width=2
                        )
            except Exception as e:
                logger.warning(f"Could not add annotation for {ann.selector}: {e}")
        
        # Save to bytes
        output = BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    
    def _calculate_annotation_position(
        self, 
        box: Dict[str, float], 
        position: str
    ) -> Tuple[int, int]:
        """Calculate annotation position relative to element."""
        margin = 20
        
        if position == "top":
            return int(box['x']), int(box['y'] - margin)
        elif position == "bottom":
            return int(box['x']), int(box['y'] + box['height'] + margin)
        elif position == "left":
            return int(box['x'] - 150), int(box['y'] + box['height'] / 2)
        else:  # right
            return int(box['x'] + box['width'] + margin), int(box['y'] + box['height'] / 2)
    
    async def find_element_by_text(self, text: str) -> Optional[str]:
        """
        Find element containing specific text.
        
        Args:
            text: Text to search for
            
        Returns:
            CSS selector for the element or None
        """
        # Try common selectors
        selectors_to_try = [
            f"button:has-text('{text}')",
            f"a:has-text('{text}')",
            f"[class*='btn']:has-text('{text}')",
            f"*:has-text('{text}')",
        ]
        
        for selector in selectors_to_try:
            try:
                element = await self._page.query_selector(selector)
                if element:
                    return selector
            except Exception:
                continue
        
        return None
    
    async def capture_workflow(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[bytes]:
        """
        Capture a multi-step workflow.
        
        Args:
            steps: List of steps, each with:
                - route: Optional route to navigate to
                - action: Optional action ('click', 'fill', 'wait')
                - selector: Selector for action
                - value: Value for 'fill' action
                - highlight: Optional list of selectors to highlight
                - annotation: Optional annotation text
                
        Returns:
            List of screenshots for each step
        """
        screenshots = []
        
        for i, step in enumerate(steps):
            logger.info(f"Executing step {i + 1}: {step}")
            
            # Navigate if route provided
            if 'route' in step:
                await self._page.goto(
                    f"{self.base_url}{step['route']}", 
                    wait_until="networkidle"
                )
            
            # Execute action
            action = step.get('action')
            selector = step.get('selector')
            
            if action == 'click' and selector:
                await self._page.click(selector)
                await self._page.wait_for_load_state("networkidle")
            elif action == 'fill' and selector:
                await self._page.fill(selector, step.get('value', ''))
            elif action == 'wait' and selector:
                await self._page.wait_for_selector(selector)
            
            # Add small delay for animations
            await asyncio.sleep(0.5)
            
            # Capture screenshot
            highlight = step.get('highlight', [])
            if selector and action == 'click':
                highlight.append(selector)
            
            annotations = []
            if step.get('annotation'):
                annotations.append(Annotation(
                    selector=selector or 'body',
                    text=step['annotation'],
                    position=step.get('annotation_position', 'right')
                ))
            
            screenshot = await self.capture_route(
                route="",  # Already on the page
                highlight_selectors=highlight,
                annotations=annotations if annotations else None
            )
            screenshots.append(screenshot)
        
        return screenshots
    
    async def stop(self):
        """Stop the service and clean up."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        # Stop dev server
        if self._dev_server_process:
            self._dev_server_process.terminate()
            try:
                self._dev_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._dev_server_process.kill()
            logger.info("Dev server stopped")
        
        self._is_started = False
        self._is_authenticated = False
        logger.info("Screenshot service stopped")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Convenience function for one-off screenshots
async def capture_screenshot(
    project_dir: str,
    route: str,
    base_url: str = "http://localhost:3000",
    highlight: Optional[List[str]] = None,
    start_server: bool = True
) -> bytes:
    """
    Convenience function to capture a single screenshot.
    
    Args:
        project_dir: Path to React project
        route: Route to capture
        base_url: Base URL of dev server
        highlight: Selectors to highlight
        start_server: Whether to start dev server
        
    Returns:
        Screenshot as PNG bytes
    """
    async with ScreenshotService(project_dir, base_url) as service:
        if not start_server:
            # Just start Playwright, not the server
            service._dev_server_process = None
        return await service.capture_route(route, highlight_selectors=highlight)

