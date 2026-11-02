import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from behaviorkit.providers import select_provider
from behaviorkit.vision import OnnxDetector,runtime


@pytest.mark.parametrize('machine',['AMD64','x86_64','arm64','aarch64'])
def test_cpu_is_architecture_independent(machine):
    selection=select_provider('cpu',['CPUExecutionProvider'],machine=machine)
    assert selection.chain==('CPUExecutionProvider',)


def test_apple_auto_uses_portable_cpu_coreml_is_opt_in():
    available=['CoreMLExecutionProvider','CPUExecutionProvider']
    assert select_provider('auto',available,system='Darwin',machine='arm64').selected=='cpu'
    chosen=select_provider('coreml',available,system='Darwin',machine='arm64')
    assert chosen.chain[0][1]['ModelFormat']=='MLProgram'
    assert chosen.chain[0][1]['MLComputeUnits']=='ALL'


def test_explicit_provider_never_claims_unsupported_hardware():
    with pytest.raises(ValueError,match='native arm64'):select_provider('coreml',['CoreMLExecutionProvider'],system='Windows',machine='AMD64')
    with pytest.raises(ValueError,match='native arm64'):select_provider('coreml',['CoreMLExecutionProvider'],system='Darwin',machine='x86_64')
    with pytest.raises(RuntimeError,match='unavailable'):select_provider('cuda',['CPUExecutionProvider'])
    assert select_provider('auto',['CPUExecutionProvider']).selected=='cpu'


def test_cpu_runtime_does_not_preload_cuda(monkeypatch):
    fake=SimpleNamespace(preload_dlls=lambda:pytest.fail('CPU path loaded CUDA'))
    monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    assert runtime() is fake


def test_real_cpu_session_and_profile(signal_model,tmp_path,monkeypatch):
    ort=pytest.importorskip('onnxruntime')
    if hasattr(ort,'preload_dlls'):monkeypatch.setattr(ort,'preload_dlls',lambda:pytest.fail('CPU session loaded CUDA'))
    detector=OnnxDetector('d-fine-n',signal_model,'cpu',threads=1,profile=True,profile_dir=tmp_path/'profiles')
    rgb=np.zeros((100,200,3),dtype=np.uint8);rgb[:,:,0]=255
    detections,_=detector.predict(rgb)
    assert len(detections)==1 and detections[0].box==pytest.approx((.1,.1,.4,.9))
    profile=detector.end_profile()
    assert profile['requested_provider_observed']
    assert not profile['gpu_execution_observed']
    assert detector.session.get_providers()==['CPUExecutionProvider']


def test_bad_input_rejected_before_runtime(signal_model):
    pytest.importorskip('onnxruntime')
    detector=OnnxDetector('d-fine-n',signal_model,'cpu',threads=1)
    for bad in [np.zeros((10,10),dtype=np.uint8),np.zeros((10,10,3),dtype=np.float32),np.zeros((0,10,3),dtype=np.uint8)]:
        with pytest.raises(ValueError,match='HWC uint8'):detector.predict(bad)


def test_manifest_tamper_fails_before_session(signal_model):
    path=signal_model/'d-fine-n/model.onnx';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256'):OnnxDetector('d-fine-n',signal_model,'cpu')


def test_auto_reports_failed_cuda_initialization_but_explicit_cuda_fails(signal_model,monkeypatch):
    from behaviorkit import vision
    calls=[]
    class Session:
        def __init__(self,path,sess_options,providers):calls.append(providers)
        def get_providers(self):return ['CPUExecutionProvider']
        def get_inputs(self):return [SimpleNamespace(name='images',type='tensor(float)',shape=[1,3,640,640]),SimpleNamespace(name='orig_target_sizes',type='tensor(int64)',shape=[1,2])]
        def get_outputs(self):return [SimpleNamespace(name=n,type='tensor(int64)' if n=='labels' else 'tensor(float)',shape=[1,2,4] if n=='boxes' else [1,2]) for n in ['labels','boxes','scores']]
    fake=SimpleNamespace(__version__='unit-test',SessionOptions=SimpleNamespace,InferenceSession=Session,get_available_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider'])
    monkeypatch.setattr(vision,'runtime',lambda preload_cuda=False:fake)
    d=OnnxDetector('d-fine-n',signal_model,'auto')
    assert d.selected_provider=='cpu' and 'failed CUDA' in d.selection_note
    assert calls[-1]==['CPUExecutionProvider']
    with pytest.raises(RuntimeError,match='refusing silent'):OnnxDetector('d-fine-n',signal_model,'cuda')


def test_coreml_profile_does_not_claim_a_specific_gpu(tmp_path):
    path=tmp_path/'profile.json';path.write_text(json.dumps([{'args':{'provider':'CoreMLExecutionProvider'}}]))
    d=object.__new__(OnnxDetector);d.provider='CoreMLExecutionProvider';d.session=SimpleNamespace(end_profiling=lambda:str(path))
    result=d.end_profile()
    assert result['requested_provider_observed'] and result['coreml_delegation_observed']
    assert not result['gpu_execution_observed']


@pytest.mark.skipif(sys.platform!='darwin',reason='requires actual Apple Silicon hardware')
def test_coreml_real_delegation(signal_model,tmp_path):
    import platform
    if platform.machine().lower()!='arm64':pytest.skip('native Apple arm64 Python required')
    ort=pytest.importorskip('onnxruntime')
    if 'CoreMLExecutionProvider' not in ort.get_available_providers():pytest.skip('wheel has no CoreML EP')
    d=OnnxDetector('d-fine-n',signal_model,'coreml',profile=True,profile_dir=tmp_path/'coreml')
    rgb=np.full((100,100,3),255,dtype=np.uint8)
    detections,_=d.predict(rgb);assert len(detections)==2
    assert d.end_profile()['coreml_delegation_observed']
