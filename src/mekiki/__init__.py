"""mekiki: find the bad demonstrations before you train on them.

mekiki is a pre-training data quality layer for robot learning demonstration
datasets. It reads episodes, runs physically-grounded integrity checks, and
reports which episodes are corrupted, low-value, or under-represented — with
magnitudes and tolerances, never bare pass/fail verdicts.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
