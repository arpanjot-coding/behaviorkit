"""Optional explicit ONNX box adapter. Config is data, never executable Python."""
from pathlib import Path
import time
from .contracts import Capabilities, fingerprint
from .core import Detection, keys, number, text
from .storage import sha256


class PresetAdapter:
    def __init__(self, name, models_dir='models', provider='auto', *, classes=('person',), confidence_floor=.1, threads=0, profile=False, profile_dir=None):
        from .vision import COCO_NAMES, OnnxDetector
        self.capabilities = Capabilities(tuple(sorted(classes)), confidence_floor)
        if set(self.capabilities.classes)-set(COCO_NAMES): raise ValueError('preset adapter cannot provide classes outside its COCO contract')
        self.detector = OnnxDetector(name, models_dir, provider, labels=classes, confidence=confidence_floor, threads=threads,profile=profile,profile_dir=profile_dir)
        self.provenance = {'adapter':'preset/1','name':name,'model':self.detector.manifest,
                           'provider':self.detector.selected_provider,'provider_note':self.detector.selection_note,
                           'runtime':self.detector.ort_version,'threads':self.detector.threads,
                           'global_candidate_limit':self.capabilities.max_detections,
                           'capture_stage':'postprocessed candidates, not raw tensors',
                           'postprocessing':({'nms':'classwise IoU','iou':.5,'limit_per_class':300} if name=='rtmdet-tiny' else
                                             {'nms':None,'selection':'exported D-FINE deploy top-K; then capture class/score filter'} if name=='d-fine-n' else
                                             {'nms':None,'selection':'global sigmoid query/class top-K equal to query count; then capture class/score filter'})}

    def predict(self, rgb):
        detections,timing=self.detector.predict(rgb)
        if len(detections)>self.capabilities.max_detections:
            detections=tuple(sorted(detections,key=lambda d:(-d.confidence,d.track_id))[:self.capabilities.max_detections])
        return detections,timing


class BoxOnnxAdapter:
    """Fixed NCHW float RGB resize; outputs boxes [1,N,4], scores [1,N,C].

    Scores must already be probabilities. Box units are normalized or input_pixels.
    No implicit sigmoid, class offsets, letterboxing, objectness or layout guesses.
    """
    def __init__(self, model, contract, *, provider='auto', confidence_floor=.1, threads=0):
        import numpy as np
        import copy
        contract=copy.deepcopy(contract)
        from .vision import runtime
        from .providers import select_provider
        keys(contract, {'schema_version','sha256','classes','width','height','input','boxes','scores','box_units','scale','mean','std','max_detections','nms_iou','license'},
             'box adapter', {'schema_version','sha256','classes','width','height','input','boxes','scores','box_units','license'})
        if contract['schema_version'] != 'box-onnx/1': raise ValueError('expected box-onnx/1')
        if sha256(model) != contract['sha256']: raise ValueError('custom model digest mismatch')
        import onnx
        graph=onnx.load(str(model),load_external_data=False)
        def verify_tensors(message):
            if isinstance(message,onnx.TensorProto) and (message.external_data or message.data_location==onnx.TensorProto.EXTERNAL):
                raise ValueError('custom adapter requires a self-contained ONNX file; external data is unsupported')
            for field,value in message.ListFields():
                if field.message_type is not None:
                    if field.is_repeated if hasattr(field,'is_repeated') else field.label==field.LABEL_REPEATED:
                        for child in value: verify_tensors(child)
                    else: verify_tensors(value)
        verify_tensors(graph)
        text(contract['license'],'model license provenance')
        classes = tuple(contract['classes'])
        self.capabilities = Capabilities(classes, confidence_floor, max_detections=contract.get('max_detections',300))
        if len(classes) > 1024: raise ValueError('custom model class capacity exceeded')
        self.width, self.height = contract['width'], contract['height']
        for v in (self.width,self.height):
            if type(v) is not int or not 1 <= v <= 4096: raise ValueError('input dimensions must be in 1..4096')
        if contract['box_units'] not in {'normalized','input_pixels'}: raise ValueError('unsupported box units')
        self.scale = number(contract.get('scale',1/255),'scale',.000001,1000)
        self.nms_iou = number(contract.get('nms_iou',.5),'nms_iou',0,1)
        self.mean, self.std = tuple(contract.get('mean',[0,0,0])), tuple(contract.get('std',[1,1,1]))
        if len(self.mean)!=3 or len(self.std)!=3: raise ValueError('mean/std must contain 3 channels')
        for v in self.mean: number(v,'mean',-10000,10000)
        for v in self.std: number(v,'std',.000001,10000)
        if type(threads) is not int or not 0<=threads<=256: raise ValueError('invalid threads')
        ort = runtime(); selection = select_provider(provider,ort.get_available_providers())
        options = ort.SessionOptions()
        if threads: options.intra_op_num_threads = threads
        if selection.selected in {'cuda','tensorrt'}: ort=runtime(preload_cuda=True)
        self.session = ort.InferenceSession(str(model),sess_options=options,providers=list(selection.chain))
        if selection.ort_name not in self.session.get_providers():
            raise RuntimeError('requested provider did not initialize; refusing silent fallback')
        inputs, outputs = self.session.get_inputs(), {x.name:x for x in self.session.get_outputs()}
        if len(inputs)!=1 or inputs[0].name!=contract['input'] or inputs[0].type!='tensor(float)' or inputs[0].shape!=[1,3,self.height,self.width]:
            raise ValueError('custom model input contract mismatch')
        if any(name not in outputs or outputs[name].type!='tensor(float)' for name in (contract['boxes'],contract['scores'])):
            raise ValueError('custom model output contract mismatch')
        self.contract = dict(contract)
        self.provenance = {'adapter':'box-onnx/1','contract':dict(contract),'contract_sha256':fingerprint(contract),
                           'provider_requested':provider,'provider_chain':self.session.get_providers(),
                           'runtime':ort.__version__,'delegation':'Profile separately to establish node execution.'}

    def predict(self, rgb):
        import numpy as np
        import cv2
        begin = time.perf_counter()
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError('adapter requires uint8 HWC RGB')
        image = cv2.resize(rgb,(self.width,self.height)).astype(np.float32)*self.scale
        image = (image-np.array(self.mean,dtype=np.float32))/np.array(self.std,dtype=np.float32)
        tensor = np.ascontiguousarray(image.transpose(2,0,1)[None])
        prepared = time.perf_counter()
        boxes,scores = self.session.run([self.contract['boxes'],self.contract['scores']],{self.contract['input']:tensor})
        inferred = time.perf_counter()
        if boxes.ndim!=3 or boxes.shape[0]!=1 or boxes.shape[2]!=4 or scores.ndim!=3 or scores.shape[:2]!=boxes.shape[:2] or scores.shape[2]!=len(self.capabilities.classes):
            raise ValueError('custom output shape mismatch')
        if boxes.shape[1]>100000 or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
            raise ValueError('invalid custom output values or capacity')
        boxes = boxes[0].astype(np.float64)
        if self.contract['box_units']=='input_pixels': boxes /= [self.width,self.height,self.width,self.height]
        if (boxes<0).any() or (boxes>1).any() or (boxes[:,0]>=boxes[:,2]).any() or (boxes[:,1]>=boxes[:,3]).any():
            raise ValueError('custom boxes violate normalized xyxy contract')
        class_ids = scores[0].argmax(axis=1)
        best = scores[0,np.arange(len(class_ids)),class_ids]
        from .vision import _nms
        keep = []
        for cls in range(len(self.capabilities.classes)):
            indices = np.flatnonzero((best>=self.capabilities.confidence_floor)&(class_ids==cls))
            keep.extend(int(indices[i]) for i in _nms(boxes[indices],best[indices],self.nms_iou,self.capabilities.max_detections))
        keep = sorted(keep,key=lambda i:(-float(best[i]),int(i)))[:self.capabilities.max_detections]
        detections = tuple(Detection(f'candidate-{i}',self.capabilities.classes[int(class_ids[i])],float(best[i]),tuple(boxes[i])) for i in keep)
        return detections, {'preprocess_ms':(prepared-begin)*1000,'inference_ms':(inferred-prepared)*1000,
                            'postprocess_ms':(time.perf_counter()-inferred)*1000}
