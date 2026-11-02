"""Run from the checkout after `python -m pip install -e .`."""
import json
from pathlib import Path
from behaviorkit import BehaviorEngine, SceneConfig, Observation, Detection

config = SceneConfig.from_dict(json.loads(Path("configs/people.json").read_text()))
engine = BehaviorEngine(config)
for index in range(21):
    result = engine.process(Observation(index / 5, (Detection("person-1", "person", 0.9, (0.3, 0.2, 0.5, 0.8)),)))
    for event in result.events:
        print(json.dumps(event))
