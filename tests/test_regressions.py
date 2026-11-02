"""Regression tests for failures found during the feasibility audit."""
import json
from dataclasses import replace
import pytest

from behaviorkit import BehaviorEngine, Detection, Observation, SceneConfig
from behaviorkit.core import Rule
from behaviorkit.cli import demo, main, replay
from behaviorkit.tracking import IoUTracker


def configuration():
    return {"schema_version":"0.1","zones":{"all":[[0,0],[1,0],[1,1],[0,1]]},"rules":[{"id":"n","kind":"count","zone":"all"}]}


@pytest.mark.parametrize('change',[
    {'zones':[]}, {'rules':{}}, {'zones':{'all':None}}, {'rules':[None]},
    {'rules':[{'id':'n','kind':'count'}]}, {'rules':[{'id':'n','kind':'count','zone':[]}]},
])
def test_bad_config_has_actionable_validation_error(change):
    data=configuration();data.update(change)
    with pytest.raises(ValueError):SceneConfig.from_dict(data)


def test_direct_rule_constructor_cannot_bypass_validation():
    with pytest.raises(ValueError):Rule('x','unsupported','all')


def test_configuration_owns_immutable_geometry():
    config=SceneConfig.from_dict(configuration())
    with pytest.raises(TypeError):config.zones['all']=((0,0),(1,0),(1,1))


def test_tracker_capacity_error_is_atomic():
    t=IoUTracker(max_tracks=1,min_hits=1,capacity_policy='error')
    first=Detection('a','person',.9,(.1,.1,.2,.2))
    separate=Detection('b','person',.9,(.7,.7,.8,.8))
    t.update((first,),0)
    with pytest.raises(ValueError,match='capacity'):t.update((separate,),.2)
    assert t.last_timestamp==0
    assert t.update((first,),.2)[0].track_id=='1'


def test_failed_replay_replaces_stale_success_summary(tmp_path):
    run=tmp_path/'run';demo(run)
    broken=tmp_path/'broken.jsonl';broken.write_text('{broken\n')
    with pytest.raises(ValueError):replay(broken,run/'config.json',run)
    summary=json.loads((run/'summary.json').read_text())
    assert summary['status']=='error'


def test_replay_never_truncates_its_input(tmp_path):
    run=tmp_path/'run';demo(run)
    source=run/'events.jsonl';original=source.read_bytes()
    with pytest.raises(ValueError,match='overlap'):replay(source,run/'config.json',run)
    assert source.read_bytes()==original


def test_replay_hashes_stream_without_reading_entire_file(tmp_path,monkeypatch):
    from pathlib import Path
    run=tmp_path/'run';demo(run)
    source=run/'observations.jsonl'
    original=Path.read_bytes
    def guard(path):
        if path==source:raise AssertionError('whole observation file was read into RAM')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guard)
    assert replay(source,run/'config.json',tmp_path/'other')['events']==1


def test_cli_bad_container_returns_error_without_traceback(tmp_path,capsys):
    path=tmp_path/'bad.json';data=configuration();data['zones']=[];path.write_text(json.dumps(data))
    assert main(['validate','--config',str(path)])==2
    assert 'behaviorkit:' in capsys.readouterr().err
