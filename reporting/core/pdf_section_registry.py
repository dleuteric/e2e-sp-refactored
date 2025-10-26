#!/usr/bin/env python3
"""
Section registry pattern for modular report building.

Provides:
- Abstract Section base class
- SectionConfig for section metadata
- Global SECTION_REGISTRY for auto-registration
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging

from reportlab.platypus import Flowable

LOGGER = logging.getLogger(__name__)


# ============================================================================
# SECTION CONFIGURATION
# ============================================================================
@dataclass
class SectionConfig:
    """
    Configuration for a report section.

    Attributes:
        name: Unique section identifier (e.g., "executive_dashboard")
        title: Display title for bookmarks/TOC
        enabled: Whether section should be included
        order: Sort order (lower = earlier in report)
        page_break_before: Insert page break before section
        page_break_after: Insert page break after section
    """
    name: str
    title: str
    enabled: bool = True
    order: int = 100
    page_break_before: bool = False
    page_break_after: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.name:
            raise ValueError("Section name cannot be empty")
        if self.order < 0:
            raise ValueError("Section order must be non-negative")


# ============================================================================
# ABSTRACT SECTION
# ============================================================================
class Section(ABC):
    """
    Abstract base class for report sections.

    Each section is responsible for:
    1. Rendering its content as ReportLab Flowables
    2. Providing bookmark title for TOC
    3. Optionally validating required data

    Subclasses implement render() method.
    """

    def __init__(self, config: SectionConfig):
        """
        Initialize section.

        Args:
            config: Section configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    @abstractmethod
    def render(self, context: 'RenderContext') -> List[Flowable]:
        """
        Render section content.

        Args:
            context: Rendering context with data/metrics/paths

        Returns:
            List of ReportLab Flowables (Paragraphs, Tables, Images, etc.)
        """
        pass

    def validate(self, context: 'RenderContext') -> bool:
        """
        Validate that required data is available.

        Args:
            context: Rendering context

        Returns:
            True if section can be rendered, False otherwise
        """
        return True  # Default: always valid

    def get_bookmark_title(self) -> str:
        """
        Get title for PDF bookmark/TOC.

        Returns:
            Bookmark title string
        """
        return self.config.title

    def __repr__(self) -> str:
        return f"Section(name='{self.config.name}', enabled={self.config.enabled}, order={self.config.order})"


# ============================================================================
# RENDER CONTEXT
# ============================================================================
@dataclass
class RenderContext:
    """
    Context passed to section render() methods.

    Contains all data needed for rendering:
    - Raw data (DataBundle)
    - Computed metrics (MetricsBundle)
    - Extracted KPIs (Dict)
    - Chart export directory (Path)
    - Configuration (Dict)
    """
    data: Any  # DataBundle from pdf_data_loader
    metrics: Any  # MetricsBundle from pdf_metrics
    kpis: Dict[str, float]
    chart_dir: Any  # Path to chart export directory
    config: Dict[str, Any]
    run_id: str
    target_id: str


# ============================================================================
# SECTION REGISTRY
# ============================================================================
class SectionRegistry:
    """
    Global registry for report sections.

    Sections auto-register on module import via:
        SECTION_REGISTRY.register(MySection(...))

    Usage:
        # In section module
        from core.pdf_section_registry import SECTION_REGISTRY, Section

        class MySection(Section):
            ...

        SECTION_REGISTRY.register(MySection(SectionConfig(...)))
    """

    def __init__(self):
        """Initialize empty registry."""
        self._sections: Dict[str, Section] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, section: Section) -> None:
        """
        Register a section.

        Args:
            section: Section instance to register

        Raises:
            ValueError: If section name already registered
        """
        name = section.config.name
        if name in self._sections:
            raise ValueError(f"Section '{name}' already registered")

        self._sections[name] = section
        self.logger.debug(f"Registered section: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a section by name.

        Args:
            name: Section name to unregister
        """
        if name in self._sections:
            del self._sections[name]
            self.logger.debug(f"Unregistered section: {name}")

    def get(self, name: str) -> Optional[Section]:
        """
        Get section by name.

        Args:
            name: Section name

        Returns:
            Section instance or None if not found
        """
        return self._sections.get(name)

    def get_enabled_sections(self) -> List[Section]:
        """
        Get all enabled sections sorted by order.

        Returns:
            List of enabled Section instances, sorted by config.order
        """
        sections = [s for s in self._sections.values() if s.config.enabled]
        return sorted(sections, key=lambda s: s.config.order)

    def list_all(self) -> List[str]:
        """
        List all registered section names.

        Returns:
            List of section names
        """
        return list(self._sections.keys())

    def __len__(self) -> int:
        """Return number of registered sections."""
        return len(self._sections)

    def __contains__(self, name: str) -> bool:
        """Check if section is registered."""
        return name in self._sections


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================
SECTION_REGISTRY = SectionRegistry()