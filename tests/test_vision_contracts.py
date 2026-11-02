import numpy as np
import pytest
from behaviorkit.vision import OnnxDetector,PRESETS,COCO_IDS,COCO_NAMES,_nms


def detector(name):
    d=object.__new__(OnnxDetector);d.name=name;d.spec=PRESETS[name];d.labels={'person'};d.confidence=.3
    d.id_to_label=dict(zip(COCO_IDS,COCO_NAMES)) if name=='rf-detr-nano' else dict(enumerate(COCO_NAMES))
    return d


def test_rfdetr_sparse_class_global_topk():
    d=detector('rf-detr-nano')
    logits=np.full((1,2,91),-10,dtype=np.float32);logits[0,0,1]=5;logits[0,0,3]=6;logits[0,1,1]=4
    boxes=np.array([[[.5,.5,.2,.4],[.2,.3,.1,.1]]],dtype=np.float32)
    result=d.decode({'dets':boxes,'labels':logits},100,100,(1,1))
    assert len(result)==1 # top2 includes two classes for query0, not query1
    assert result[0].label=='person';assert result[0].box==pytest.approx((.4,.3,.6,.7))


def test_dfine_no_second_scaling():
    d=detector('d-fine-n')
    r=d.decode({'labels':np.array([[0]]),'boxes':np.array([[[20,10,80,90]]]),'scores':np.array([[.9]])},100,100,(6.4,6.4))
    assert r[0].box==(.2,.1,.8,.9)


def test_rtmdet_classwise_nms_and_letterbox_inverse():
    d=detector('rtmdet-tiny');scores=np.zeros((1,2,80));scores[0,:,0]=[.9,.8]
    r=d.decode({'boxes':np.array([[[100,100,200,200],[100,100,200,200]]]),'scores':scores},100,100,(2,2))
    assert len(r)==1;assert r[0].box==(.5,.5,1,1)


def test_preprocess_contracts():
    for name in PRESETS:
        d=detector(name);feed,_=d.preprocess(np.zeros((100,200,3),dtype=np.uint8))
        assert feed['images'].shape==(1,3,d.spec['size'],d.spec['size'])
        assert feed['images'].dtype==np.float32
        if name=='d-fine-n':assert feed['orig_target_sizes'].tolist()==[[200,100]]
