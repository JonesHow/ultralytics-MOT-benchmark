import os
import sys


def _ensure_src_on_path():
    # Add the project's 'src' directory to sys.path for test imports
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    src_dir = os.path.join(project_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_src_on_path()

