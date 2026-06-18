"""Machine inventory template — copy to `MACHINES.py` (gitignored) and edit per client.

    cp run_jobs/baulab/MACHINES.example.py run_jobs/baulab/MACHINES.py

`dispatch.py` imports `MACHINES` and `SAFE_MACHINES` from `MACHINES.py`. Different clients
have access to (and "own") different boxes, so this lives outside version control.

A *worker* is one `MachineConfig(name=<host>, cuda=[<gpu indices>])` entry; it runs at most
one of our jobs at a time. Repeat a host with different `cuda` sets to give it several
independent, concurrent slots (e.g. `cuda=[0]` and `cuda=[1]` on the same host).
"""

from dataclasses import dataclass, field


@dataclass
class MachineConfig:
    """One worker: a host plus the GPU indices it may use."""

    name: str
    cuda: list[int] = field(default_factory=lambda: [0])


# Fallback tier — used only when no SAFE_MACHINES entry currently fits a job.
MACHINES = [
    MachineConfig(name="local", cuda=[0]),
    # MachineConfig(name="my-cluster-box-a", cuda=[0]),
    # MachineConfig(name="my-cluster-box-b", cuda=[0, 1]),   # one job spanning GPUs 0,1
    # MachineConfig(name="my-cluster-box-b", cuda=[2, 3]),   # same host, separate concurrent slot
    MachineConfig(name="umibozu", cuda=[0]),
    MachineConfig(name="tokyo", cuda=[0]),
    # MachineConfig(name="hawaii", cuda=[0]),
    # MachineConfig(name="karasuno", cuda=[0]),
    MachineConfig(name="kobe", cuda=[0]),
    MachineConfig(name="macondo", cuda=[0]),
    MachineConfig(name="ei", cuda=[0]),
    # MachineConfig(name="andromeda", cuda=[0]), #! 20.04
    MachineConfig(name="kawasaki", cuda=[0]),
    # MachineConfig(name="sendai", cuda=[0]),   #! 20.04
    # MachineConfig(name="kumamoto", cuda=[0]),
    # MachineConfig(name="hamada", cuda=[0]),
    MachineConfig(name="fukuyama", cuda=[0]),
    # MachineConfig(name="osaka", cuda=[0]), #! 20.04
    MachineConfig(name="cancun", cuda=[0]),
    MachineConfig(name="kameoka", cuda=[0]),
    # MachineConfig(name="bippu", cuda=[0]), #! 20.04
    # MachineConfig(name="hokkaido", cuda=[0]), #! 20.04
    MachineConfig(name="kyoto", cuda=[0, 1]),
    MachineConfig(name="tottori", cuda=[0]),
    MachineConfig(name="saitama", cuda=[0, 1]),
    # MachineConfig(name="nagoya", cuda=[3, 4, 5, 6, 7]),
    # MachineConfig(name="hakone", cuda=list(range(8)))
]

# SAFE_MACHINES are the *preferred* tier — tried FIRST for every job (e.g. machines you own,
# where your jobs won't be pre-empted by someone reclaiming the box). Only when **no** safe
# machine currently satisfies a job does the scheduler fall back to MACHINES. You may list a
# host here even if it's also in MACHINES — duplicates are de-duped into the safe tier. An
# empty SAFE_MACHINES just means "treat every machine equally".
SAFE_MACHINES = [
    # MachineConfig(name="my-own-box", cuda=[0]),
]
