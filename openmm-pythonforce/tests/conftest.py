import sys
from pathlib import Path

# Expose the local runner modules; torchmdnet comes from the active environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
