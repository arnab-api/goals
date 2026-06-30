"""Shared datatypes for the baulab dispatcher.

Plain, dependency-free dataclasses used across the module — and, importantly, by the
per-client `CONFIG.py`. Kept out of `CONFIG.py` so that file holds *only* settings, and out
of `dispatch.py` so the config template can build its machine lists without importing the
scheduler. (Config-aware types like `Worker`, which read settings from `CONFIG`, stay in
`dispatch.py` to avoid a CONFIG <-> datatypes import cycle.)
"""

from dataclasses import dataclass, field


@dataclass
class MachineConfig:
    """One worker slot: a host plus the GPU indices it may use.

    Repeat a host with different `cuda` sets to give it several independent, concurrent slots
    (e.g. `cuda=[0]` and `cuda=[1]` on the same host).
    """

    name: str
    cuda: list[int] = field(default_factory=lambda: [0])
