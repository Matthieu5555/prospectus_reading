"""HTML template assets for the prospectus visualization dashboard."""

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"

CSS_STYLES = (_TEMPLATES_DIR / "report.css").read_text()
JS_SCRIPT = (_TEMPLATES_DIR / "report.js").read_text()
