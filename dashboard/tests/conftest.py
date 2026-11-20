"""Pytest path setup: expose the dashboard package dir so tests can
`import common` and reach the pages without a real Streamlit session."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
