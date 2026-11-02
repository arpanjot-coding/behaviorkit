from importlib.resources import files
import json
import pytest


def test_packaged_structural_schemas_and_generated_recording(tmp_path):
    jsonschema=pytest.importorskip('jsonschema')
    from behaviorkit.contracts import Capabilities
    from behaviorkit.recording import record_candidates
    from test_system_foundations import frame
    schemas={name:json.loads(files('behaviorkit').joinpath('schemas',name+'.schema.json').read_text())
             for name in ('frame','recording','evaluation')}
    for value in schemas.values(): jsonschema.Draft202012Validator.check_schema(value)
    f=frame(0); manifest=record_candidates([f],tmp_path/'c',Capabilities(('person',),.1),{})
    jsonschema.validate(f.to_dict(),schemas['frame'])
    jsonschema.validate(manifest,schemas['recording'])
    annotations={'schema_version':'evaluation/1','coverage':[{'source':'cam','epoch':'0','start':0,'end':10}],
                 'events':[{'source':'cam','epoch':'0','rule':'r','start':0,'end':3}]}
    jsonschema.validate(annotations,schemas['evaluation'])


def test_custom_class_capability_rejected_before_model_loading(tmp_path):
    from behaviorkit.adapters import PresetAdapter
    with pytest.raises(ValueError,match='COCO'): PresetAdapter('d-fine-n',tmp_path,classes=['custom_pallet'])
