import os
import sys

from core.types import AppContext


def detect_app_context() -> AppContext:
	return 'cli'
