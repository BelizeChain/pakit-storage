"""Pakit package namespace — redirects imports to the flat repo layout."""
import os as _os

# Point pakit's __path__ to the repo root so that
# `from pakit.core import ...` resolves to `core/...` at the project root.
__path__ = [_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))]
