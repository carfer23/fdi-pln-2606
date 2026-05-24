"""Entry point for uv run fdi-pln-2606-p5."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# When installed as a wheel the top-level py-modules are in site-packages
# alongside this package, so no path manipulation is needed.
# When running from the source tree (editable), we add the project root so
# that the sibling .py files are importable.
_src = Path(__file__).parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from main import cli  # noqa: E402

__all__ = ["cli"]
