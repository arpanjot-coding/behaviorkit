import pytest


@pytest.mark.parametrize('precision',['fp16','int8'])
def test_actual_precision_export_keeps_float_io_and_remains_unqualified(tmp_path,precision):
    np=pytest.importorskip('numpy'); onnx=pytest.importorskip('onnx'); ort=pytest.importorskip('onnxruntime')
    from onnx import TensorProto as T,helper as h,numpy_helper as nh
    from behaviorkit.profiles import export_precision_candidate
    source=tmp_path/'reference.onnx'; output=tmp_path/(precision+'.onnx')
    graph=h.make_graph([h.make_node('MatMul',['input','weights'],['output'])],'matmul',
                      [h.make_tensor_value_info('input',T.FLOAT,[1,4])],
                      [h.make_tensor_value_info('output',T.FLOAT,[1,4])],
                      [nh.from_array(np.eye(4,dtype=np.float32),'weights')])
    onnx.save(h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=10),source)
    from onnxruntime.quantization import CalibrationDataReader
    class Reader(CalibrationDataReader):
        def __init__(self): self.items=iter([{'input':np.ones((1,4),np.float32)},{'input':np.zeros((1,4),np.float32)}])
        def get_next(self): return next(self.items,None)
    report=export_precision_candidate(source,output,precision=precision,calibration_reader=Reader() if precision=='int8' else None)
    assert report['status']=='unqualified'
    session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].type=='tensor(float)' and session.get_outputs()[0].type=='tensor(float)'
    expected=np.ones((1,4),np.float32)
    np.testing.assert_allclose(session.run(None,{'input':expected})[0],expected,atol=.02)
    with pytest.raises(ValueError,match='already exists'): export_precision_candidate(source,output,precision=precision)


def test_int8_rejects_uncalibrated_export_before_loading(tmp_path):
    from behaviorkit.profiles import export_precision_candidate
    with pytest.raises(ValueError,match='calibration'):
        export_precision_candidate(tmp_path/'not-read.onnx',tmp_path/'q.onnx',precision='int8')
