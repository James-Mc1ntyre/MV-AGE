"""
MV_AGE

Simple active learning for food labeling with ingredient text and nutrient data.
"""

__version__ = "0.1.0"

__all__ = [
    "FoodLabelingProject",
    "__version__",
    "prepare_project",
]


def __getattr__(name: str):
    if name in {"FoodLabelingProject", "prepare_project"}:
        from .ActiveLearning import FoodLabelingProject, prepare_project

        exports = {
            "FoodLabelingProject": FoodLabelingProject,
            "prepare_project": prepare_project,
        }
        return exports[name]
    raise AttributeError(f"module 'mv_age' has no attribute {name!r}")
