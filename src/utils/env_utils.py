"""Tools for reading and controlling the runtime environment."""

import logging
import os
import pathlib
from typing import Union

import yaml

ENV_RESULTS_DIR = "RELATIONS_RESULTS_DIR"
CLAUDE_CACHE_DIR = "CLAUDE_CACHE_DIR"
HF_CACHE_DIR = None
EM_ADAPTER = None  # env.yml's EM_ADAPTER: the LoRA checkpoint dir of scripts/em/ (optional)

logger = logging.getLogger(__name__)

try:
    PROJECT_ROOT = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
    with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
        config = yaml.safe_load(f)
        DEFAULT_MODELS_DIR = config["MODEL_DIR"]
        DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, config["DATA_DIR"])
        DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, config["RESULTS_DIR"])
        CLAUDE_CACHE_DIR = os.path.join(PROJECT_ROOT, config["CLAUDE_CACHE_DIR"])
        HF_CACHE_DIR = config.get("HF_CACHE", None)
        if HF_CACHE_DIR is not None:
            HF_CACHE_DIR = HF_CACHE_DIR.strip()
            if len(HF_CACHE_DIR) == 0:
                HF_CACHE_DIR = None
        EM_ADAPTER = (config.get("EM_ADAPTER") or "").strip() or None

        for dir in [
            DEFAULT_MODELS_DIR,
            DEFAULT_DATA_DIR,
            DEFAULT_RESULTS_DIR,
            CLAUDE_CACHE_DIR,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)

except FileNotFoundError:
    logger.error(
        f'''env.yml not found in {PROJECT_ROOT}!
Setting MODEL_ROOT="". Models will now be downloaded to conda env cache, if not already there
Other defaults are set to:
    DATA_DIR = "data"
    RESULTS_DIR = "results"'''
    )
    DEFAULT_MODELS_DIR = ""
    DEFAULT_DATA_DIR = "data"
    DEFAULT_RESULTS_DIR = "results"
    CLAUDE_CACHE_DIR = "claude_cache"

PathLike = Union[str, pathlib.Path]

logger = logging.getLogger(__name__)


def maybe_relative_to_repo(path: PathLike) -> pathlib.Path:
    """Resolve the (potentially relative) path.

    Args:
        path: The path to resolve. If the path is relative, it is assumed to be
            relative to the repository root. If this is already an absolute path,
            it is returned unchanged.

    Returns:
        The resolved path.

    """
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return pathlib.Path(__file__).parents[2] / path


def read_path(name: str, default: PathLike) -> pathlib.Path:
    """Try to read a path from the env.

    Args:
        name: Name of the env variable to read.
        default: The default path in case the environment variable
            does not exist. If relative, assumed to be relative from the repo root.

    Returns:
        The path, if the variable could be read and/or if default was provided.

    """
    read = os.environ.get(name)
    path: PathLike = maybe_relative_to_repo(default) if read is None else read
    return pathlib.Path(path)


def determine_results_dir(default: PathLike = DEFAULT_RESULTS_DIR) -> pathlib.Path:
    """Return directory containing results from any scripts.

    Args:
        default: Default to use if RELATIONS_RESULTS_DIR env
            variable is not set. Defaults to './results'.

    Returns:
        Directory results are stored in.

    """
    return read_path(ENV_RESULTS_DIR, default)


def load_env_var(var: str) -> Union[str, None]:
    try:
        PROJECT_ROOT = "/".join(
            os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]
        )
        with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"""env.yml not found in {PROJECT_ROOT}!""")

    if var not in config or config[var] is None or config[var] == "":
        logger.error(f"{var} not set in env.yml!")
        return None

    else:
        return config[var]
