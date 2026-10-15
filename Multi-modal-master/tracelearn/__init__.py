"""TraceLearn — Flask application factory."""
from __future__ import annotations

from flask import Flask

from . import db as dbmod
from .config import settings
from .seed import ensure_seeded

# (key, desktop label, href, icon, short mobile label)
NAV = [
    ("dashboard", "Dashboard", "/", "home", "Home"),
    ("upload", "Sources", "/upload", "upload", "Sources"),
    ("chat", "Tutor", "/chat", "chat", "Tutor"),
    ("graph", "Concept Map", "/graph", "graph", "Map"),
    ("assessment", "Practice", "/assessment", "quiz", "Practice"),
    ("mastery", "Mastery", "/mastery", "target", "Mastery"),
    ("eval", "Evaluation", "/eval", "chart", "Eval"),
]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.secret_key = settings.secret_key
    app.teardown_appcontext(dbmod.close_db)

    from .routes.api import bp as api_bp
    from .routes.pages import bp as pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.template_filter("hms")
    def _hms(seconds) -> str:
        try:
            total = int(max(0, float(seconds)))
        except (TypeError, ValueError):
            return "0:00"
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    @app.template_filter("cite_label")
    def _cite_label(citation: dict) -> str:
        citation = citation or {}
        source_type = citation.get("source_type")
        if source_type == "video" and citation.get("timestamp_start") is not None:
            return "Play from " + _hms(citation["timestamp_start"])
        if source_type == "slide" and citation.get("slide_number") is not None:
            return f"Open slide {citation['slide_number']}"
        if citation.get("page_number") is not None:
            return f"Open page {citation['page_number']}"
        if citation.get("figure_reference"):
            return "Open figure"
        return "Open source"

    @app.context_processor
    def inject_globals() -> dict:
        return {
            "app_state": settings.summary(),
            "demo_mode": settings.demo_mode,
            "nav_items": NAV,
            "course_id": "dsa_6006",
        }

    with app.app_context():
        dbmod.init_db()
        ensure_seeded()

    return app
