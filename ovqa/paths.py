import os
from getpass import getuser
from pathlib import Path

from packg import Const
from packg.paths import setup_environ, EnvKeys, get_cache_dir


class OvqaEnvKeys(Const):
    OVQA_REPO_ROOT = "OVQA_REPO_ROOT"
    OVQA_OUTPUT_DIR = "OVQA_OUTPUT_DIR"


uname = getuser()
base_dir = str(Path(__file__).parent.parent)
OVQA_ENV_DEFAULTS = {
    EnvKeys.ENV_DATA_DIR: "data",
    EnvKeys.ENV_CACHE_DIR: f"/home/{uname}/.cache",
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

    setup_environ(
        load_dotenv=load_dotenv,
        override_from_dotenv=override_from_dotenv,
        dotenv_path=dotenv_path,
        use_defaults=False,
    )

    for env_k, v in OVQA_ENV_DEFAULTS.items():
        if env_k not in os.environ:
            if verbose:
                print(f"from ovqa.paths defaults write: {env_k}={v}")
            os.environ[env_k] = v
    if verbose:
        print(f"# Done setting up ovqa environment.")


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
    for env_k in list(EnvKeys.values()) + list(OvqaEnvKeys.values()):
        print_fn(f"{prefix}{env_k}={os.environ[env_k]}")
