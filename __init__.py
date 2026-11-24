"""Top-level package for comfychat_filters."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """1iminal"""
__email__ = "1im03189@gmail.com"
__version__ = "0.0.1"

from .src.comfychat_filters.nodes import NODE_CLASS_MAPPINGS
from .src.comfychat_filters.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
