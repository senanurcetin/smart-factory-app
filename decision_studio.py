"""Independent, non-destructive Decision Studio preview routes.

The original /case-study and /rul-case-study routes remain unchanged.
The studio fetches the existing /api/case-study and /api/rul-case-study
endpoints, preserving source truth and error handling.
"""

from pathlib import Path

from flask import Blueprint, send_from_directory

decision_studio_bp = Blueprint("decision_studio", __name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@decision_studio_bp.get("/studio/maintenance")
@decision_studio_bp.get("/studio/rul")
def decision_studio():
    return send_from_directory(_STATIC_DIR, "decision-studio.html")
