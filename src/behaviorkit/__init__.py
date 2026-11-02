"""A local, replayable behaviour engine. Importing this module loads no ML framework."""
from .core import BehaviorEngine, Detection, Observation, SceneConfig, Evaluation

__version__ = "0.3.0"
__all__ = ["BehaviorEngine", "Detection", "Observation", "SceneConfig", "Evaluation"]
