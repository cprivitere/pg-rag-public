"""VRAM helpers for embedder / model VRAM measurement.

Utility module (no standalone probe): the embedder bake-off (embed_eval.py)
and the service-flag sweeper (vram_sweep.py) import _url_args / _total_vram_mb
/ get_per_pid_vram from here. The old standalone probe that wrote
data/embed_vram.json was superseded by the bake-off's per-PID measurement and
removed.
"""

import re
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PORT = 8084
HOST = "Scamper"


def get_base_url():
    import socket

    try:
        addrs = socket.getaddrinfo(HOST, PORT, socket.AF_INET6, socket.SOCK_STREAM)
        for res in addrs:
            af, socktype, proto, canonname, sa = res
            ip, port_num, flowinfo, scope_id = sa
            if scope_id:
                return f"http://[{ip}%{scope_id}]:{port_num}"
            else:
                return f"http://[{ip}]:{port_num}"
    except Exception:
        pass
    return f"http://{HOST}:{PORT}"


def _total_vram_mb():
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            (
                "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
                ".CounterSamples | Measure-Object -Property CookedValue -Sum"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Output contains ANSI color codes; strip them
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout)
    m = re.search(r"Sum\s*:\s*([\d.]+)", cleaned)
    return (float(m.group(1)) / 1e6) if m else None  # MB


def get_per_pid_vram(pid):
    """Dedicated GPU bytes for one PID (summed across LUIDs/segments).

    Process-scoped \\GPU Process Memory(*)\\Dedicated Usage — immune to driver
    page retention across back-to-back same-shape processes, unlike the summed
    adapter total. Reliable for attributing one llama-server's VRAM.
    """
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            (
                f"(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
                f"-ErrorAction SilentlyContinue).CounterSamples | "
                f"Where-Object {{ $_.InstanceName -match '^pid_{pid}_' }} | "
                f"Measure-Object -Property CookedValue -Sum"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout)
    m = re.search(r"Sum\s*:\s*([\d.]+)", cleaned)
    return (float(m.group(1)) / 1e6) if m else None  # MB


def _url_args(url):
    """llama-server args for a model given as a repo:quant ref (always -hf)."""
    repo, _, ref = url.partition(":")
    return ["-hf", f"{repo}:{ref}"]
