"""Select only measured deployment profiles; export variants remain candidates."""
from pathlib import Path
from .core import number, text
from .storage import sha256, write_json


def select_profile(profiles, *, min_precision, min_recall, max_false_alerts_per_hour,
                   max_delay_seconds, max_p95_ms, max_rss_bytes, hardware_id, dataset_id):
    for name,value in [('min_precision',min_precision),('min_recall',min_recall)]: number(value,name,0,1)
    for name,value in [('max_false_alerts_per_hour',max_false_alerts_per_hour),('max_delay_seconds',max_delay_seconds),
                       ('max_p95_ms',max_p95_ms),('max_rss_bytes',max_rss_bytes)]: number(value,name,0)
    text(hardware_id,'hardware_id'); text(dataset_id,'dataset_id')
    eligible,rejected=[],[]
    for profile in profiles:
        identity=text(profile['id'],'profile id')
        reasons=[]
        if profile.get('qualification')!='learned_model_and_events': reasons.append('unqualified')
        if profile.get('hardware_id')!=hardware_id or profile.get('dataset_id')!=dataset_id: reasons.append('different_hardware_or_dataset')
        for key,budget,minimum in [('precision',min_precision,True),('recall',min_recall,True),
                                   ('false_alerts_per_hour',max_false_alerts_per_hour,False),('delay_seconds',max_delay_seconds,False),
                                   ('p95_ms',max_p95_ms,False),('rss_bytes',max_rss_bytes,False)]:
            value=profile.get(key)
            if value is None: reasons.append('missing_'+key); continue
            number(value,key,0)
            if value<budget if minimum else value>budget: reasons.append('budget_'+key)
        import re
        if any(not isinstance(profile.get(k),str) or not re.fullmatch('[0-9a-f]{64}',profile[k]) for k in ('evidence_sha256','model_sha256')):
            reasons.append('missing_artifact_provenance')
        if reasons: rejected.append({'id':identity,'reasons':reasons})
        else: eligible.append(profile)
    eligible.sort(key=lambda p:(p['p95_ms'],p['rss_bytes'],p['id']))
    return {'selected':eligible[0] if eligible else None,'eligible':[p['id'] for p in eligible],
            'rejected':rejected,'note':'Selection uses supplied measured reports. It does not extrapolate camera capacity or assert new qualification.'}


def export_precision_candidate(model, output, *, precision, calibration_reader=None):
    """Explicit offline export, no inference downloads and no automatic promotion.

    INT8 readers must follow ORT CalibrationDataReader and supply representative
    preprocessed tensors. Both variants preserve float32 public I/O contracts.
    """
    output=Path(output)
    if output.exists() or output.with_suffix('.candidate.json').exists():
        raise ValueError('precision candidate output already exists')
    if Path(model).resolve()==output.resolve(): raise ValueError('cannot overwrite reference model')
    if precision not in {'fp16','int8'}: raise ValueError('precision must be fp16 or int8')
    if precision=='int8' and calibration_reader is None: raise ValueError('representative calibration reader is required for INT8')
    import onnx
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        if precision=='fp16':
            from onnxruntime.transformers.float16 import convert_float_to_float16
            from onnxruntime.transformers.onnx_model import OnnxModel
            converted=convert_float_to_float16(onnx.load(str(model)),keep_io_types=True)
            OnnxModel(converted).topological_sort()
            onnx.save(converted,str(output))
        else:
            from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
            quantize_static(str(model),str(output),calibration_reader,quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8)
        onnx.checker.check_model(str(output))
        report={'schema_version':'precision-candidate/1','status':'unqualified','precision':precision,
                'reference_sha256':sha256(model),'candidate_sha256':sha256(output),
                'requirements':['provider execution profile','positive and negative model fixtures','held-out event comparison','full pipeline resource budgets'],
                'note':'This candidate is not approved for runtime presets. Export success is not accuracy or speed evidence.'}
        write_json(output.with_suffix('.candidate.json'),report)
    except BaseException:
        output.unlink(missing_ok=True); raise
    return report
