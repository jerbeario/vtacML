""" Utils functions for vtacML"""

from pathlib import Path


# Define the root directory
ROOTDIR = Path(__file__).parent


def get_path(subpath):
    """Utility function to get the path to a subpath relative to ROOTDIR."""
    print(f"rootdir: {ROOTDIR}/{subpath} FIX ME I HAVE AN EXTRA '/' ")
    return f"{ROOTDIR}/{subpath}"
