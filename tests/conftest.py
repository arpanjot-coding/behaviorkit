"""An ONNX signal fixture, not a learned detector or an accuracy dataset."""
import json
import pytest


@pytest.fixture
def signal_model(tmp_path):
    onnx=pytest.importorskip('onnx')
    from onnx import helper,TensorProto as T
    from behaviorkit.storage import sha256
    folder=tmp_path/'models/d-fine-n';folder.mkdir(parents=True)
    def tensor(name,kind,shape,values):return helper.make_tensor(name,kind,shape,values)
    initializers=[tensor('channels',T.INT64,[2],[0,1]),tensor('axes',T.INT64,[1],[1]),tensor('normalized_boxes',T.FLOAT,[1,2,4],[.1,.1,.4,.9,.6,.1,.9,.9]),tensor('person_labels',T.INT64,[1,2],[0,0])]
    nodes=[helper.make_node('ReduceMean',['images'],['means'],axes=[2,3],keepdims=0),helper.make_node('Gather',['means','channels'],['scores'],axis=1),helper.make_node('Cast',['orig_target_sizes'],['sizes'],to=T.FLOAT),helper.make_node('Concat',['sizes','sizes'],['xyxy_scale'],axis=1),helper.make_node('Unsqueeze',['xyxy_scale','axes'],['scale']),helper.make_node('Mul',['normalized_boxes','scale'],['boxes']),helper.make_node('Identity',['person_labels'],['labels'])]
    graph=helper.make_graph(nodes,'colour-coded-transport-fixture',[helper.make_tensor_value_info('images',T.FLOAT,[1,3,640,640]),helper.make_tensor_value_info('orig_target_sizes',T.INT64,[1,2])],[helper.make_tensor_value_info('labels',T.INT64,[1,2]),helper.make_tensor_value_info('boxes',T.FLOAT,[1,2,4]),helper.make_tensor_value_info('scores',T.FLOAT,[1,2])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9
    onnx.checker.check_model(model);onnx.save(model,folder/'model.onnx')
    (folder/'manifest.json').write_text(json.dumps({'name':'d-fine-n','contract':'dfine_deploy','sha256':sha256(folder/'model.onnx'),'test_fixture':True,'note':'RGB signal transport fixture; no learned computer vision'}))
    return folder.parent
