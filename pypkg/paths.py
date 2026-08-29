"""Repository-root-relative paths, for code that may run from anywhere.

Scripts are launched from the repository root, so a bare "data/..." works for
them. Notebooks are not: papermill executes a notebook with the working
directory set to the notebook's own folder, so moving notebook.ipynb into
notebooks/ silently broke every relative path inside it. Resolving against the
installed package's location instead works from either place, and keeps working
if the notebook moves again.
"""

from __future__ import annotations

import os

#: Repository root, derived from this file's location. pypkg is installed
#: editable into every environment, so this points at the working tree.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts) -> str:
    """A repository-relative path, resolved against the root."""
    return os.path.join(ROOT, *parts)
