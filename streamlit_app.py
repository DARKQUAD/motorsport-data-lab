"""Entry point for Streamlit Community Cloud and Codespaces: makes ``src`` importable, runs the app.

Locally, ``streamlit run app/streamlit_app.py`` after ``pip install -e .[app]`` does the same.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "app" / "streamlit_app.py"), run_name="__main__")
