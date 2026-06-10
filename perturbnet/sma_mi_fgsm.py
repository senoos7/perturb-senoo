"""Backward-compatible re-export. Prefer perturbnet.attacks."""
from perturbnet.attacks.sma_mi_fgsm import ALGORITHM_NAME, run_sma_mi_fgsm, runtime_profile_summary

__all__ = ["ALGORITHM_NAME", "run_sma_mi_fgsm", "runtime_profile_summary"]
