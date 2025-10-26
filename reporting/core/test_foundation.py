# test_foundation.py
"""
Test foundation modules.

Run with: pytest test_foundation.py -v
"""
import pytest
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

from reporting.core.pdf_styles import *
from reporting.core.pdf_section_registry import *
from reporting.core.pdf_kpi_config import *


def test_colors_defined():
    """Test that all required colors are defined."""
    assert 'excellent' in COLORS
    assert 'good' in COLORS
    assert 'marginal' in COLORS
    assert 'poor' in COLORS


def test_paragraph_styles_defined():
    """Test that all paragraph styles exist."""
    assert style_title is not None
    assert style_h1 is not None
    assert style_kpi_value is not None


def test_section_registry():
    """Test section registration."""
    registry = SectionRegistry()

    # Create dummy section
    class DummySection(Section):
        def render(self, context):
            return [Paragraph("Dummy", style_body)]

    config = SectionConfig(name="dummy", title="Dummy", order=10)
    section = DummySection(config)

    # Register
    registry.register(section)
    assert len(registry) == 1
    assert "dummy" in registry

    # Get enabled
    enabled = registry.get_enabled_sections()
    assert len(enabled) == 1
    assert enabled[0].config.name == "dummy"


def test_kpi_config():
    """Test KPI configuration."""
    assert 'cep50_km' in KPI_CONFIGS
    assert 'rmse_km' in KPI_CONFIGS

    cep50_cfg = KPI_CONFIGS['cep50_km']
    assert cep50_cfg.enabled
    assert cep50_cfg.extractor is not None
    assert cep50_cfg.label == 'Tracking Accuracy'


def test_pdf_generation_basic():
    """Test basic PDF generation with styles."""
    output = Path("test_foundation.pdf")

    doc = SimpleDocTemplate(str(output), pagesize=A4)
    story = []

    # Add styled content
    story.append(Paragraph("Test Report", style_title))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Section Heading", style_h1))
    story.append(Paragraph("Body text goes here.", style_body))

    # Build
    doc.build(story)

    assert output.exists()
    assert output.stat().st_size > 1000  # Non-empty PDF

    # Cleanup
    output.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])