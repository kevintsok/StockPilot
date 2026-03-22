"""
Scheduler setup: generate cron.d file and installation guide.

Usage:
    python -m auto_select_stock.notify.scheduler --install
"""

import argparse
import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def generate_cron_line(script_path: str, log_path: str) -> str:
    """A股收盘后 15:05 (周一至周五)"""
    return f"5 15 * * 1-5 {script_path} >> {log_path} 2>&1\n"


def generate_cron_file() -> str:
    root = get_repo_root()
    script = root / "scripts" / "notify_daily.sh"
    log = root / "logs" / "notify.log"
    return generate_cron_line(str(script), str(log))


def ensure_scripts_dir() -> None:
    root = get_repo_root()
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)


def run_install() -> None:
    root = get_repo_root()
    cron_dir = root / "cron.d"
    cron_dir.mkdir(exist_ok=True)

    cron_file = cron_dir / "stockpilot_notify"
    cron_content = generate_cron_file()
    cron_file.write_text(cron_content)

    # Ensure log dir exists
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Ensure shell script exists
    shell_script = root / "scripts" / "notify_daily.sh"
    if not shell_script.exists():
        ensure_scripts_dir()
        shell_script.write_text(
            f"""#!/bin/bash
# StockPilot daily notification wrapper
# Called by cron at 15:05 Mon-Fri

source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin

cd "{root}"
PYTHONPATH=./src python -m auto_select_stock.notify.runner
"""
        )
        os.chmod(shell_script, 0o755)

    print(f"Cron file written to: {cron_file}")
    print(f"Cron content:\n{cron_content}")
    print("\nTo install, run:")
    print(f"  sudo cp {cron_file} /etc/cron.d/stockpilot_notify")
    print("Or add this line to your crontab (crontab -e):")
    print(f"  {cron_content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPilot notification scheduler")
    parser.add_argument("--install", action="store_true", help="Generate and install cron config")
    args = parser.parse_args()

    if args.install:
        run_install()
    else:
        print("Scheduler setup for StockPilot daily notification")
        print("\nUsage:")
        print("  python -m auto_select_stock.notify.scheduler --install")
        print("\nThis will:")
        print("  1. Create cron.d file at project root")
        print("  2. Create scripts/notify_daily.sh wrapper")
        print("  3. Print installation instructions")


if __name__ == "__main__":
    main()
