"""
API Module for Unified AI Platform

This module provides REST API endpoints for all platform capabilities.
"""

from .analytics_routes import analytics_bp

__all__ = ['analytics_bp']
