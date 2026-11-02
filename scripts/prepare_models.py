"""Explicit, offline-runtime model preparation using trusted publisher sources.

This is a build-time script, not imported by BehaviorKit. Run exports in a separate
environment. Never pass untrusted checkpoints to the upstream pickle loaders.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import urllib.request

URLS = {
 "rtmdet-tiny":"https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
 "d-fine-n":"https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
 "rf-detr-nano":"https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
}


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def download(url,path):
 if path.exists():return
 path.parent.mkdir(parents=True,exist_ok=True)
 request=urllib.request.Request(url,headers={'User-Agent':'BehaviorKit-feasibility/0.1'})
 temporary=path.with_suffix('.download')
 with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as target:
  shutil.copyfileobj(response,target)
 temporary.replace(path)


def prepare_rtmdet(folder,checkpoint):
 import torch
 # Import-only placeholder: this network export uses no compiled MMCV operators.
 # Any accidentally invoked extension fails, never returns fabricated output.
 ext=types.ModuleType('mmcv._ext');ext.__file__='unavailable_mmcv_extension';ext.__path__=[]
 ext.__spec__=importlib.machinery.ModuleSpec('mmcv._ext',loader=None)
 def missing(name):
  if name.startswith('__'):raise AttributeError(name)
  def fail(*a,**kw):raise RuntimeError(f'MMCV operator {name} is unavailable in this export-only environment')
  return fail
 ext.__getattr__=missing;sys.modules['mmcv._ext']=ext
 import mmdet
 from mmengine.config import Config
 from mmdet.apis import init_detector
 cfg_path=Path(mmdet.__file__).parent/'.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
 cfg=Config.fromfile(str(cfg_path));cfg.model.backbone.init_cfg=None
 model=init_detector(cfg,checkpoint=None,device='cpu')
 # Only the explicit publisher checkpoint above is loaded with pickle enabled.
 checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 state=checkpoint_data['state_dict']
 # These old persistent preprocessing buffers are outside the exported network;
 # current MMEngine registers them non-persistently. Keep strict learned weights.
 state={key:value for key,value in state.items() if key not in ('data_preprocessor.mean','data_preprocessor.std')}
 model.load_state_dict(state,strict=True);model.eval()
 class Export(torch.nn.Module):
  def __init__(self,m):super().__init__();self.m=m
  def forward(self,images):
   cls_scores,bbox_preds=self.m.bbox_head(self.m.extract_feat(images))
   priors=self.m.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls_scores],dtype=images.dtype,device=images.device)
   scores=torch.cat([x.permute(0,2,3,1).reshape(1,-1,80) for x in cls_scores],1).sigmoid()
   distances=torch.cat([x.permute(0,2,3,1).reshape(1,-1,4) for x in bbox_preds],1)
   points=torch.cat(priors,0)[None,:,:2]
   boxes=torch.cat([points-distances[:,:,:2],points+distances[:,:,2:]],-1)
   return boxes,scores
 wrapper=Export(model).eval();sample=torch.randn(1,3,640,640)
 torch.onnx.export(wrapper,(sample,),str(folder/'model.onnx'),input_names=['images'],output_names=['boxes','scores'],opset_version=17,dynamo=False)
 return wrapper,(sample,),['boxes','scores'],{'mmdet':mmdet.__version__,'prior_offset':float(model.bbox_head.prior_generator.offset)}


def prepare_dfine(folder,checkpoint,source):
 import torch
 if source is None or not (source/'src/core').exists():
  raise ValueError('--dfine-source must point to the official D-FINE checkout')
 sys.path.insert(0,str(source.resolve()))
 from src.core import YAMLConfig
 cfg=YAMLConfig(str(source/'configs/dfine/dfine_hgnetv2_n_coco.yml'),resume=str(checkpoint))
 if 'HGNetv2' in cfg.yaml_cfg:cfg.yaml_cfg['HGNetv2']['pretrained']=False
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 state=state['ema']['module'] if 'ema' in state else state['model']
 cfg.model.load_state_dict(state)
 class Export(torch.nn.Module):
  def __init__(self):super().__init__();self.model=cfg.model.deploy();self.post=cfg.postprocessor.deploy()
  def forward(self,images,orig_target_sizes):return self.post(self.model(images),orig_target_sizes)
 model=Export().eval();sample=(torch.rand(1,3,640,640),torch.tensor([[1280,720]],dtype=torch.int64))
 torch.onnx.export(model,sample,str(folder/'model.onnx'),input_names=['images','orig_target_sizes'],output_names=['labels','boxes','scores'],opset_version=17,dynamo=False)
 try:revision=subprocess.check_output(['git','-c',f'safe.directory={source.resolve().as_posix()}','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 except subprocess.SubprocessError:revision='not recorded'
 return model,sample,['labels','boxes','scores'],{'upstream_commit':revision}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--models',nargs='+',choices=list(URLS),default=list(URLS));p.add_argument('--models-dir',type=Path,default=Path('models'));p.add_argument('--dfine-source',type=Path);p.add_argument('--skip-parity',action='store_true',help='Record unverified export explicitly; not recommended')
 a=p.parse_args()
 import numpy as np
 import onnx
 import onnxruntime as ort
 import torch
 torch.set_num_threads(min(8,os.cpu_count() or 1))
 torch.manual_seed(2026)
 for name in a.models:
  folder=(a.models_dir/name).resolve();folder.mkdir(parents=True,exist_ok=True)
  checkpoint=folder/'publisher-checkpoint.pth';download(URLS[name],checkpoint)
  if name=='rf-detr-nano':
   from rfdetr import RFDETRNano
   import rfdetr
   model=RFDETRNano(device='cpu',pretrain_weights=str(checkpoint))
   original_export=torch.onnx.export
   def quiet_export(*args,**kwargs):
    kwargs['verbose']=False
    return original_export(*args,**kwargs)
   torch.onnx.export=quiet_export
   try:model.export(format='onnx',output_dir=str(folder/'export'))
   finally:torch.onnx.export=original_export
   paths=list((folder/'export').rglob('*.onnx'))
   if len(paths)!=1:raise RuntimeError(f'Expected one RF-DETR ONNX graph, found {paths}')
   # Load external weights and save one self-contained graph for reproducible hashing.
   graph=onnx.load(str(paths[0]),load_external_data=True)
   onnx.save_model(graph,str(folder/'model.onnx'),save_as_external_data=False)
   metadata={'rfdetr':importlib.metadata.version('rfdetr')}
   parity={'status':'upstream_export','note':'Upstream RF-DETR exporter; no independent PyTorch/ONNX parity check performed by this script.'}
  else:
   if name=='rtmdet-tiny':wrapper,inputs,outputs,metadata=prepare_rtmdet(folder,checkpoint)
   else:wrapper,inputs,outputs,metadata=prepare_dfine(folder,checkpoint,a.dfine_source)
   if not a.skip_parity:
    session=ort.InferenceSession(str(folder/'model.onnx'),providers=['CPUExecutionProvider'])
    with torch.no_grad():expected=wrapper(*inputs)
    actual=session.run(outputs,{i.name:t.detach().numpy() for i,t in zip(session.get_inputs(),inputs)})
    errors={}
    if name=='d-fine-n':
     # Floating-point ties can permute top-K rows. Match the same labels/boxes
     # before asserting tensor values; do not hide missing or changed detections.
     from scipy.optimize import linear_sum_assignment
     ref=[t.detach().numpy() for t in expected]
     cost=(ref[0][0,:,None]!=actual[0][0,None,:])*1e6+np.max(np.abs(ref[1][0,:,None,:]-actual[1][0,None,:,:]),axis=-1)+np.abs(ref[2][0,:,None]-actual[2][0,None,:])*1000
     _,permutation=linear_sum_assignment(cost)
     actual=[value[:,permutation] for value in actual]
    for key,ref,value in zip(outputs,expected,actual):
     reference=ref.detach().numpy();errors[key]=float(np.max(np.abs(reference-value)))
     np.testing.assert_allclose(value,reference,rtol=1e-3,atol=0.05 if key=='boxes' else 1e-3)
    parity={'status':'passed','sample':'one seeded random tensor; tensor parity, not accuracy','alignment':'same-label box/score bipartite alignment for D-FINE top-K tie permutations' if name=='d-fine-n' else 'tensor order','max_absolute_error':errors}
   else:parity={'status':'not_run'}
  graph=onnx.load(str(folder/'model.onnx'));onnx.checker.check_model(graph)
  from behaviorkit.vision import PRESETS
  manifest={'schema_version':'0.1','name':name,'contract':PRESETS[name]['contract'],'sha256':digest(folder/'model.onnx'),'checkpoint_sha256':digest(checkpoint),'checkpoint_url':URLS[name],'license_basis':PRESETS[name]['license'],'source':PRESETS[name]['source'],'export':metadata,'parity':parity,'inputs':[i.name for i in graph.graph.input],'outputs':[i.name for i in graph.graph.output],'precision':'FP32','batch':1,'external_data':[]}
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
  print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
