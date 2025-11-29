"""
Visualization module - screenshots, diagrams, and visual documentation.
"""

from .screenshot_service import (
    ScreenshotService,
    ScreenshotConfig,
    Annotation,
    AuthConfig,
    capture_screenshot,
)

__all__ = [
    'ScreenshotService',
    'ScreenshotConfig',
    'Annotation',
    'AuthConfig',
    'capture_screenshot',
]

