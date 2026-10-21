"""
TactAI Phase 3 Benchmark & Validation Suite Runner.

This script delegates directly to `scripts/run_phase3_validation.py` to ensure a single
authoritative source of truth for all benchmarks, ablations, and figures.
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "run_phase3_validation.py")
    sys.exit(subprocess.call([sys.executable, script_path] + sys.argv[1:]))
