import os
from getpass import getuser
from pathlib import Path

from dotenv import dotenv_values, find_dotenv
from packg.constclass import Const


class OvqaEnvKeys(Const):
    ENV_DATA_DIR = "ENV_DATA_DIR"
    ENV_CACHE_DIR = "ENV_CACHE_DIR"
    OVQA_REPO_ROOT = "OVQA_REPO_ROOT"
    OVQA_OUTPUT_DIR = "OVQA_OUTPUT_DIR"


uname = getuser()
home = Path.home()
base_dir = str(Path(__file__).parent.parent)
OVQA_ENV_DEFAULTS = {
    OvqaEnvKeys.ENV_DATA_DIR: "data",
    OvqaEnvKeys.ENV_CACHE_DIR: (home / ".cache").as_posix(),
    OvqaEnvKeys.OVQA_REPO_ROOT: base_dir,
    OvqaEnvKeys.OVQA_OUTPUT_DIR: f"output",
}

_setup_environ_done = False


def setup_ovqa_environ(
    verbose=False,
    load_dotenv=True,
    override_from_dotenv=True,
    dotenv_path=None,
):
    global _setup_environ_done
    if _setup_environ_done and not verbose:
        return
    _setup_environ_done = True

    if load_dotenv:
        if dotenv_path is not None:
            dotenv_path = Path(dotenv_path).as_posix()
            if not Path(dotenv_path).is_file():
                raise FileNotFoundError(f"dotenv_path provided but not found: {dotenv_path}")
        else:
            dotenv_path = ""
            if verbose:
                print(f"Searching dotenv file...")

        if dotenv_path == "":
            # try to find it relative to the current dir
            dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path == "":
            # try to find in home
            proposal = Path.home() / ".env"
            if proposal.is_file():
                dotenv_path = proposal.as_posix()

        if dotenv_path != "":
            values = dotenv_values(dotenv_path, verbose=verbose)
            if verbose:
                print(
                    f"Got {len(values)} from .env file, "
                    f"found it as {find_dotenv()} from {os.getcwd()}"
                )
        else:
            values = {}
            if verbose:
                print(f"Dotenv file not found.")
        for k, v in values.items():
            if override_from_dotenv or k not in os.environ:
                if verbose:
                    print(f"    From .env write: {k}={type(v).__name__} length {len(v)}")
                os.environ[k] = v

    for env_k, v in OVQA_ENV_DEFAULTS.items():
        if env_k not in os.environ:
            if verbose:
                print(f"from ovqa.paths defaults write: {env_k}={v}")
            os.environ[env_k] = v
    if verbose:
        print(f"# Done setting up ovqa environment.")


def get_from_environ(env_k: str):
    setup_ovqa_environ()
    value = os.environ[env_k]
    if value == "" or value is None:
        raise ValueError(f"Environment variable {env_k} is undefined: '{value}'")
    return value


def get_data_dir() -> Path:
    setup_ovqa_environ()
    return Path(os.environ[OvqaEnvKeys.ENV_DATA_DIR])


def get_cache_dir() -> Path:
    setup_ovqa_environ()
    return Path(os.environ[OvqaEnvKeys.ENV_CACHE_DIR])


def get_ovqa_repo_root():
    setup_ovqa_environ()
    return Path(os.environ[OvqaEnvKeys.OVQA_REPO_ROOT])


def get_ovqa_annotations_dir():
    return get_ovqa_repo_root() / "ovqa/annotations"


def get_ovqa_output_dir():
    setup_ovqa_environ()
    return Path(os.environ[OvqaEnvKeys.OVQA_OUTPUT_DIR])


def get_ovqa_cache_dir():
    setup_ovqa_environ()
    return get_cache_dir() / "ovqa"


def print_all_environment_variables(print_fn=print, prefix="    ", verbose=True):
    setup_ovqa_environ(verbose=verbose)
    print_fn(f"# path definitions:")
    for env_k in list(OvqaEnvKeys.values()) + list(OvqaEnvKeys.values()):
        print_fn(f"{prefix}{env_k}={os.environ[env_k]}")
