import subprocess
import sys
from pathlib import Path


def test_original_feedforward_script_runs(tmp_path):
    weights = tmp_path / "weights.txt"
    weights.write_text("1 1\n1\n")

    script = Path("src/mathematical_ml/mnist/feedforward.py")

    result = subprocess.run(
        [sys.executable, str(script), str(weights), "T1", "1", "2"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "[3.0]" in result.stdout
