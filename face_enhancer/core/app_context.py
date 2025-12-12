import os
import sys

from face_enhancer.core.types import AppContext


def detect_app_context() -> AppContext:
	return 'cli'
