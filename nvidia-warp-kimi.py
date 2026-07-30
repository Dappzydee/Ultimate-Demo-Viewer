"""Compatibility entry point for the refactored CS2 visibility analyzer.

The reusable implementation now lives in ``cs2_visibility``. Existing command
lines can keep using this file unchanged.
"""

from cs2_visibility.cli import main


if __name__ == "__main__":
    main()
