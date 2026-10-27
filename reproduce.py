"""Run the original experiment configurations from any working directory."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    print("\n> python " + " ".join(map(str, arguments)), flush=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=env, check=True)


def tests():
    run("-m", "unittest", "experiments.test_benchmark", "experiments.posthoc.test_audit",
        "experiments.threshold_fairness.test_experiment", "-v")


def pilot(output, smoke=False):
    destination = output / ("smoke_pilot" if smoke else "pilot")
    run("-m", "experiments.run", "--config", ROOT / "experiments/configs" / ("smoke.json" if smoke else "pilot.json"),
        "--out", destination, "--resume")
    run("-m", "experiments.verify", destination)
    if not smoke:
        run("-m", "experiments.analyze", destination)
        run("-m", "experiments.posthoc.audit", destination)


def threshold(output, smoke=False):
    destination = output / ("smoke_threshold" if smoke else "threshold_fairness")
    run("-m", "experiments.threshold_fairness.oracle", "--out", destination / "oracle")
    run("-m", "experiments.threshold_fairness.run", "--config",
        ROOT / "experiments/threshold_fairness" / ("smoke.json" if smoke else "config.json"), "--out", destination)
    run("-m", "experiments.threshold_fairness.evaluate", destination)
    run("-m", "experiments.threshold_fairness.verify", destination)
    if not smoke:
        run("-m", "experiments.threshold_fairness.analyze", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["tests", "smoke", "pilot", "threshold", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="Output directory; use a new path after changing source/config")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.experiment in ("tests", "smoke", "all"):
        tests()
    if args.experiment in ("smoke", "all"):
        pilot(output, smoke=True)
        threshold(output, smoke=True)
    if args.experiment in ("pilot", "all"):
        pilot(output)
    if args.experiment in ("threshold", "all"):
        threshold(output)
    print("\nCompleted successfully.", flush=True)


if __name__ == "__main__":
    main()
