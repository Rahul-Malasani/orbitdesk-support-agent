"""Print the hardware + runtime the assignment asks us to disclose.

Run:  python -m scripts.detect_hardware
"""
from __future__ import annotations

import platform
import subprocess


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("== Host ==")
    print(f"platform     : {platform.platform()}")
    print(f"machine/arch : {platform.machine()}")
    print(f"processor    : {_sysctl('machdep.cpu.brand_string')}")
    print(f"logical cpus : {_sysctl('hw.ncpu')}")
    try:
        ram_gb = int(_sysctl("hw.memsize")) / 1024 ** 3
        print(f"memory (RAM) : {ram_gb:.0f} GB")
    except Exception:
        print("memory (RAM) : unknown")
    print(f"python       : {platform.python_version()}")

    print("\n== Accelerator ==")
    try:
        import torch

        mps = torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        device = "mps" if mps else ("cuda" if cuda else "cpu")
        print(f"torch        : {torch.__version__}")
        print(f"mps available: {mps}")
        print(f"cuda avail.  : {cuda}")
        print(f"embedding dev: {device}")
    except Exception as exc:
        print(f"torch        : not importable ({exc})")


if __name__ == "__main__":
    main()
