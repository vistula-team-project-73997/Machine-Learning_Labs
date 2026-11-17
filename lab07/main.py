"""Run in PyCharm. Continues complete, configuration-matching experiments."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from run_lab import main
    main()
