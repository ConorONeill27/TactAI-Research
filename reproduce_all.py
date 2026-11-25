"""Regenerate logged MuJoCo benchmarks, manuscript figures and the IEEEtran PDF."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


def compile_paper(engine, tectonic):
    build = ROOT / "build" / "latex"
    build.mkdir(parents=True, exist_ok=True)
    if tectonic:
        command = [engine, "--keep-logs", "--outdir", str(build), "main.tex"]
    else:
        command = [engine, "-pdf", "-halt-on-error", "-interaction=nonstopmode",
                   f"-outdir={build}", "main.tex"]
    subprocess.run(command, cwd=DOCS, check=True)
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"undefined references|undefined citations|(?:Reference|Citation).*undefined",
                 log, flags=re.IGNORECASE):
        raise RuntimeError(f"Unresolved references in {build / 'main.log'}")
    shutil.copyfile(build / "main.pdf", DOCS / "research_paper.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true",
                        help="Use existing benchmark logs and figures.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes for the contact audit.")
    compilers = parser.add_mutually_exclusive_group()
    compilers.add_argument("--tectonic", metavar="PATH", help="Path to a Tectonic executable.")
    compilers.add_argument("--latexmk", metavar="PATH", help="Path to a latexmk executable.")
    args = parser.parse_args()
    latexmk = args.latexmk or (shutil.which("latexmk") if not args.tectonic else None)
    tectonic = args.tectonic or (shutil.which("tectonic") if not latexmk else None)
    if not latexmk and not tectonic:
        parser.error("Install latexmk with pdfLaTeX, or Tectonic, and add it to PATH.")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    steps = []
    if not args.compile_only:
        steps.extend([
            ("scripts.run_phase3_validation", "--only", "contour-slip"),
            ("scripts.plot_unified_grasp",),
            ("scripts.run_phase2_validation",),
            ("scripts.render_visual_snapshots",),
            ("scripts.draw_controller_architecture",),
        ])
    steps.append(("scripts.generate_manuscript_results",))
    for step in steps:
        print(f"\n== {' '.join(step)} ==", flush=True)
        subprocess.run([sys.executable, "-m", *step], cwd=ROOT, env=env, check=True)
    compile_paper(latexmk or tectonic, tectonic is not None)
    subprocess.run([sys.executable, "-m", "scripts.check_manuscript_consistency"],
                   cwd=ROOT, env=env, check=True)
    print(f"Built {DOCS / 'research_paper.pdf'}")


if __name__ == "__main__":
    main()
