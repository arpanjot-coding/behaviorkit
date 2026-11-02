"""Explicit ONNX contracts for the three shortlisted COCO models.

No model code is downloaded or executed at runtime. Model artifacts are supplied
locally after export. NumPy, OpenCV and ONNX Runtime are imported only here.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import time

from .core import Detection, number
from .providers import select_provider
from .storage import sha256

COCO_NAMES = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",")
COCO_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
PRESETS = {
    "rtmdet-tiny": {"size":640, "contract":"rtmdet_decoded", "license":"Apache-2.0 code; verify checkpoint provenance", "source":"https://github.com/open-mmlab/mmdetection", "person_class_id":0},
    "d-fine-n": {"size":640, "contract":"dfine_deploy", "license":"Apache-2.0; use COCO-only checkpoint, review upstream terms", "source":"https://github.com/Peterande/D-FINE", "person_class_id":0},
    "rf-detr-nano": {"size":384, "contract":"rfdetr_raw", "license":"Apache-2.0 Nano weights; other family sizes can differ", "source":"https://github.com/roboflow/rf-detr", "person_class_id":1},
}


def runtime(preload_cuda=False):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError('Install a runtime extra: python -m pip install -e ".[vision,cpu]" (Intel/AMD), .[vision,apple] (Apple Silicon), or .[vision,cuda] (NVIDIA)') from e
    if preload_cuda and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    return ort


def _nms(boxes, scores, threshold=0.5, limit=300):
    import numpy as np
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < limit:
        index = int(order[0]); keep.append(index); rest = order[1:]
        if not rest.size:
            break
        left = np.maximum(boxes[index,:2], boxes[rest,:2])
        right = np.minimum(boxes[index,2:], boxes[rest,2:])
        intersection = np.maximum(0,right-left).prod(axis=1)
        area = np.maximum(0,boxes[rest,2:]-boxes[rest,:2]).prod(axis=1)
        own = np.maximum(0,boxes[index,2:]-boxes[index,:2]).prod()
        overlap = intersection / np.maximum(1e-12, area+own-intersection)
        order = rest[overlap <= threshold]
    return keep


class OnnxDetector:
    def __init__(self, name, models_dir="models", provider="auto", labels=("person",), confidence=0.3, profile=False, threads=0, profile_dir=None):
        import numpy as np
        if name not in PRESETS:
            raise ValueError(f"unknown model {name}; choose from {', '.join(PRESETS)}")
        confidence=number(confidence,'confidence',0,1)
        self.name, self.spec = name, PRESETS[name]
        self.labels, self.confidence = set(labels), confidence
        if not self.labels <= set(COCO_NAMES):
            raise ValueError(f"unsupported COCO labels: {sorted(self.labels-set(COCO_NAMES))}")
        self.path = Path(models_dir) / name / "model.onnx"
        manifest_path = self.path.with_name("manifest.json")
        if not self.path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Missing {self.path} or manifest.json. See scripts/prepare_models.py and docs/model-contracts.md.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("name") != name or self.manifest.get("contract") != self.spec["contract"]:
            raise ValueError("model manifest name/contract does not match the selected preset")
        if self.manifest.get("sha256") != sha256(self.path):
            raise ValueError("ONNX model SHA256 does not match manifest.json")
        for external in self.manifest.get("external_data", []):
            artifact = self.path.parent / external["file"]
            if artifact.resolve().parent != self.path.parent.resolve() or sha256(artifact) != external["sha256"]:
                raise ValueError("ONNX external data path/hash mismatch")
        if type(threads) is not int or not 0<=threads<=1024:raise ValueError("threads must be an integer in 0..1024; zero uses runtime defaults")
        ort = runtime()
        selection=select_provider(provider,ort.get_available_providers())
        if selection.selected in {"cuda","tensorrt"}:ort=runtime(preload_cuda=True)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads=threads or (0 if selection.selected=="cpu" else 1)
        options.enable_profiling = profile
        if profile:
            profile_folder=Path(profile_dir) if profile_dir else Path(tempfile.mkdtemp(prefix="behaviorkit-profile-"))
            profile_folder.mkdir(parents=True,exist_ok=True)
            options.profile_file_prefix = str(profile_folder / "ort-profile")
        providers=list(selection.chain)
        if selection.selected=="tensorrt":
            providers[0]=(selection.ort_name,{"trt_fp16_enable":True,"trt_engine_cache_enable":True,"trt_engine_cache_path":str(self.path.parent/"trt-cache")})
        try:
            self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=providers)
            initialized=selection.ort_name in self.session.get_providers()
        except Exception as error:
            if provider!="auto" or selection.selected!="cuda":
                raise RuntimeError(f"ONNX session initialization failed for {selection.ort_name}: {error}") from error
            initialized=False
        if not initialized:
            if provider!="auto":raise RuntimeError(f"{selection.ort_name} initialization failed; refusing silent CPU fallback")
            selection=select_provider("cpu",ort.get_available_providers())
            self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=list(selection.chain))
            self.selection_note="Auto detected failed CUDA initialization and selected CPU; explicit cuda would fail."
        else:self.selection_note=selection.reason
        self.requested_provider=provider
        self.selected_provider=selection.selected
        self.provider=selection.ort_name
        self.threads=options.intra_op_num_threads
        self.ort_version = ort.__version__
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [i.name for i in self.session.get_outputs()]
        expected_inputs = {"images", "orig_target_sizes"} if self.spec["contract"] == "dfine_deploy" else {"images"}
        if self.spec["contract"] == "rfdetr_raw" and set(self.input_names) in ({"input"}, {"images"}):
            self.image_input_name = self.input_names[0]
        elif set(self.input_names) == expected_inputs:
            self.image_input_name = "images"
        else:
            raise ValueError(f"Unexpected ONNX inputs {self.input_names}; expected {sorted(expected_inputs)}")
        expected_outputs = {"boxes","scores"} if self.spec["contract"] == "rtmdet_decoded" else ({"labels","boxes","scores"} if self.spec["contract"] == "dfine_deploy" else {"dets","labels"})
        if set(self.output_names) != expected_outputs:
            raise ValueError(f"Unexpected ONNX outputs {self.output_names}; expected {sorted(expected_outputs)}")
        size=self.spec['size']
        contracts={self.image_input_name:('tensor(float)',(1,3,size,size))}
        if self.spec['contract']=='dfine_deploy':contracts['orig_target_sizes']=('tensor(int64)',(1,2))
        for meta in self.session.get_inputs():self._validate_metadata(meta,*contracts[meta.name])
        classes=91 if self.spec['contract']=='rfdetr_raw' else 80
        output_contracts=({'boxes':('tensor(float)',(1,None,4)),'scores':('tensor(float)',(1,None,classes))} if self.spec['contract']=='rtmdet_decoded' else
            {'labels':('tensor(int64)',(1,None)),'boxes':('tensor(float)',(1,None,4)),'scores':('tensor(float)',(1,None))} if self.spec['contract']=='dfine_deploy' else
            {'dets':('tensor(float)',(1,None,4)),'labels':('tensor(float)',(1,None,classes))})
        for meta in self.session.get_outputs():self._validate_metadata(meta,*output_contracts[meta.name])
        self.id_to_label = dict(zip(COCO_IDS, COCO_NAMES)) if self.spec["contract"] == "rfdetr_raw" else dict(enumerate(COCO_NAMES))

    @staticmethod
    def _validate_metadata(meta,dtype,shape):
        if meta.type!=dtype:raise ValueError(f'ONNX {meta.name} dtype {meta.type} does not match required {dtype}')
        if len(meta.shape)!=len(shape) or any(expected is not None and isinstance(actual,int) and actual!=expected for actual,expected in zip(meta.shape,shape)):
            raise ValueError(f'ONNX {meta.name} shape {meta.shape} does not match required {shape}')

    def _validate_outputs(self,outputs):
        """Reject malformed results instead of silently truncating via zip()."""
        import numpy as np
        contract=self.spec['contract']
        boxes=outputs['dets' if contract=='rfdetr_raw' else 'boxes']
        if boxes.ndim!=3 or boxes.shape[0]!=1 or boxes.shape[2]!=4:raise RuntimeError('ONNX output boxes must have shape [1,N,4]')
        n=boxes.shape[1]
        expected=({'boxes':(1,n,4),'scores':(1,n,80)} if contract=='rtmdet_decoded' else
            {'labels':(1,n),'boxes':(1,n,4),'scores':(1,n)} if contract=='dfine_deploy' else
            {'dets':(1,n,4),'labels':(1,n,91)})
        for name,shape in expected.items():
            value=outputs[name]
            if value.shape!=shape or not np.isfinite(value).all():raise RuntimeError(f'ONNX output {name} has invalid shape or non-finite values; required {shape}')
        if contract!='rfdetr_raw' and ((outputs['scores']<0).any() or (outputs['scores']>1).any()):raise RuntimeError('ONNX output scores must be probabilities in [0,1]')

    def preprocess(self, rgb):
        import cv2
        import numpy as np
        height, width = rgb.shape[:2]
        size = self.spec["size"]
        if self.spec["contract"] == "rtmdet_decoded":
            scale = min(size/width, size/height)
            rw, rh = min(size, int(width*scale+0.5)), min(size, int(height*scale+0.5))
            canvas = np.full((size,size,3),114,dtype=np.uint8)
            canvas[:rh,:rw] = cv2.resize(rgb[...,::-1], (rw,rh), interpolation=cv2.INTER_LINEAR)
            image = (canvas.astype(np.float32)-np.array([103.53,116.28,123.675],dtype=np.float32))/np.array([57.375,57.12,58.395],dtype=np.float32)
            # Independent factors preserve exact inverse coordinates after rounding.
            transform = (rw/width, rh/height)
        else:
            image = cv2.resize(rgb,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
            transform = (size/width,size/height)
            if self.spec["contract"] == "rfdetr_raw":
                image = (image-np.array([0.485,0.456,0.406],dtype=np.float32))/np.array([0.229,0.224,0.225],dtype=np.float32)
        feed = {getattr(self,"image_input_name","images"):np.ascontiguousarray(image.transpose(2,0,1)[None])}
        if self.spec["contract"] == "dfine_deploy":
            feed["orig_target_sizes"] = np.array([[width,height]],dtype=np.int64)
        return feed, transform

    def decode(self, outputs, width, height, transform):
        import numpy as np
        contract = self.spec["contract"]
        if contract == "rtmdet_decoded":
            raw_boxes, raw_scores = outputs["boxes"][0], outputs["scores"][0]
            boxes, scores, labels = [], [], []
            for cid, label in self.id_to_label.items():
                if label not in self.labels:
                    continue
                selected = raw_scores[:,cid] >= self.confidence
                b, s = raw_boxes[selected], raw_scores[selected,cid]
                for i in _nms(b,s):
                    boxes.append(b[i]);scores.append(s[i]);labels.append(cid)
            boxes = np.asarray(boxes,dtype=np.float32).reshape(-1,4)
            boxes /= np.array([transform[0],transform[1],transform[0],transform[1]],dtype=np.float32)
        elif contract == "dfine_deploy":
            boxes, scores, labels = outputs["boxes"][0], outputs["scores"][0], outputs["labels"][0]
        else:
            logits = outputs["labels"][0]
            probabilities = 1/(1+np.exp(-np.clip(logits,-80,80)))
            k = min(probabilities.shape[0],probabilities.size)
            indices = np.argsort(-probabilities.ravel(),kind="stable")[:k]
            labels = indices % probabilities.shape[1]
            scores = probabilities.ravel()[indices]
            cxcywh = outputs["dets"][0][indices // probabilities.shape[1]]
            boxes = np.concatenate((cxcywh[:,:2]-cxcywh[:,2:]/2,cxcywh[:,:2]+cxcywh[:,2:]/2),axis=1)
            boxes *= np.array([width,height,width,height])
        result = []
        for index, (box, score, cid) in enumerate(zip(boxes,scores,labels)):
            label = self.id_to_label.get(int(cid))
            if label not in self.labels or not np.isfinite(score) or score < self.confidence:
                continue
            normalized = np.asarray(box,dtype=np.float64)/[width,height,width,height]
            if not np.isfinite(normalized).all():
                continue
            normalized = np.clip(normalized,0,1)
            if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                continue
            result.append(Detection(f"detection-{index}",label,float(score),tuple(normalized)))
        return tuple(result)

    def predict(self, rgb):
        import numpy as np
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError("predict expects a nonempty HWC uint8 RGB array")
        start = time.perf_counter()
        feed, transform = self.preprocess(rgb)
        prepared = time.perf_counter()
        values = self.session.run(self.output_names,feed)
        inferred = time.perf_counter()
        outputs=dict(zip(self.output_names,values))
        self._validate_outputs(outputs)
        detections = self.decode(outputs,rgb.shape[1],rgb.shape[0],transform)
        end = time.perf_counter()
        return detections, {"preprocess_ms":(prepared-start)*1000, "inference_transfer_ms":(inferred-prepared)*1000, "postprocess_ms":(end-inferred)*1000, "total_ms":(end-start)*1000}

    def end_profile(self):
        path = self.session.end_profiling()
        if not path:
            return {}
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = {}
        for record in records:
            provider = record.get("args",{}).get("provider")
            if provider:
                counts[provider] = counts.get(provider,0)+1
        return {"file":str(path), "kernel_events_by_provider":counts, "requested_provider":self.provider,"requested_provider_observed":counts.get(self.provider,0)>0,"gpu_execution_observed":any(counts.get(p,0)>0 for p in ["CUDAExecutionProvider","TensorrtExecutionProvider"]),"coreml_delegation_observed":counts.get("CoreMLExecutionProvider",0)>0,"note":"Provider kernel events do not identify which Apple compute unit CoreML used."}
