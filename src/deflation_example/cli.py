"""Reader-facing command; validation errors never silently change the requested device."""

from dataclasses import dataclass
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from .runner import run_demo


@dataclass
class DemoConfig:
    problem: str = "diffusion"
    grid: int | None = None
    rank: int = 20
    alpha: float = 1e-3
    device: str = "cpu"
    plot: bool = False
    threads: int = 1
    maxiter: int = 20000
    output: str = "???"


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig):
    try:
        checked = OmegaConf.merge(OmegaConf.structured(DemoConfig), config)
        args = OmegaConf.to_container(checked, resolve=True, throw_on_missing=True)
        report = run_demo(**args)
    except (ValueError, RuntimeError, OSError, OmegaConfBaseException) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"Results: {args['output']}; all checks passed: {report['success']}")
    if not report["success"]:
        raise SystemExit(1)
