"""Run this file in PyCharm with the existing course Python 3.11.8 environment."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    check = subprocess.run([sys.executable, str(HERE / "src" / "test_autodiff.py")],
                           capture_output=True, text=True)
    evidence = check.stdout + check.stderr
    (HERE / "results" / "verification.txt").write_text(evidence, encoding="utf-8")
    print(evidence, flush=True)
    check.check_returncode()
    from experiments import run
    run()
    saved = subprocess.run([sys.executable, str(HERE / "src" / "verify_results.py")],
                           capture_output=True, text=True)
    evidence = saved.stdout + saved.stderr
    (HERE / "results" / "artifact_verification.txt").write_text(evidence, encoding="utf-8")
    print(evidence, flush=True)
    saved.check_returncode()
