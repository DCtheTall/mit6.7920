"""
Utility functions for displaying results

"""

from typing import Any, Dict, Tuple


def print_grid(X: Dict[Tuple[int, int], Any],
               size: int = 4):
    for y in range(size - 1, -1, -1):
        print(*(str(X[(x, y)]) + '\t' for x in range(size)))
