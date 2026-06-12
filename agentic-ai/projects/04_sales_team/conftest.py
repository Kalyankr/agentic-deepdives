"""Ensure this project's own ``solution``/``starter`` modules are imported.

Each project ships a top-level ``solution.py``/``starter.py``. When the whole ``projects`` tree is
collected in one pytest run, the first-imported ``solution`` would otherwise stay cached and shadow
the others. Evicting the cached names here (pytest's prepend import mode puts *this* project's
directory first on ``sys.path``) makes both ``pytest projects`` and per-project runs work.
"""

import sys

for _name in ("solution", "starter"):
    sys.modules.pop(_name, None)
