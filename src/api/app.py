"""
EXPLAINIUM - Main API (Redirects to Simplified API)

This file now redirects to the new simplified API for backward compatibility.
For new development, use simplified_app directly.
"""

# Import the new simplified app
from src.api.simplified_app import app

# Re-export for backward compatibility
__all__ = ["app"]
