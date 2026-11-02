"""Offline event evaluation with explicit coverage and denominators.

Annotation format evaluation/1 uses source/epoch/rule, [start,end] truth
intervals, confirmation `timestamp` for predictions, and disjoint scored windows.
Intervals touching an unknown boundary must be annotated censored and are ignored.
This is an offline dataset-size operation; runtime engines retain no event history.
"""
from __future__ import annotations
from collections import defaultdict, deque
from math import sqrt
from .core import keys, number, text


def _key(row):
    return tuple(text(row[k], k) for k in ('source', 'epoch', 'rule'))


def _interval(row):
    start = number(row['start'], 'start')
    end = number(row.get('end', row.get('timestamp')), 'end')
    if end < start:
        raise ValueError('interval ends before it starts')
    return start, end


def _matching(edges):
    """Maximum cardinality, deterministic augmenting paths. No greedy lost TP."""
    assigned, owner = {}, {}
    for root in sorted(edges):
        parents, queue, found = {root: None}, deque([root]), False
        while queue and not found:
            node = queue.popleft()
            for target in edges[node]:
                if target not in owner:
                    while node is not None:
                        old = assigned.get(node)
                        assigned[node] = target
                        owner[target] = node
                        target, node = old, parents[node]
                    found = True
                    break
                child = owner[target]
                if child not in parents:
                    parents[child] = node
                    queue.append(child)
    return assigned


def evaluate(annotations, predictions, *, start_tolerance=0.5, min_interval_iou=0.0):
    """Score one frozen camera/day split. Counts align by exact source-time key.

    A truth interval matches by start tolerance and optionally temporal IoU.
    A confirmed prediction may end earlier/later than the truth. Delay is measured
    from truth `eligible_at` (default truth end), and can be negative.
    Predictions intersecting censored truth are ignored rather than called FP.
    """
    if annotations.get('schema_version') != 'evaluation/1':
        raise ValueError('expected evaluation/1 annotation document')
    keys(annotations, {'schema_version','split','groups','coverage','events','counts'}, 'annotations', {'schema_version','coverage'})
    keys(predictions, {'events','counts'}, 'predictions')
    number(start_tolerance, 'start_tolerance', 0, 86400)
    number(min_interval_iou, 'min_interval_iou', 0, 1)
    windows = defaultdict(list)
    for w in annotations.get('coverage', []):
        keys(w, {'source','epoch','start','end','covered'}, 'coverage window', {'source','epoch','start','end'})
        k = (text(w['source'], 'source'), text(w['epoch'], 'epoch'))
        a, b = _interval(w)
        if b <= a:
            raise ValueError('coverage windows must have positive duration')
        if type(w.get('covered', True)) is not bool:
            raise ValueError('covered must be boolean')
        windows[k].append((a, b, w.get('covered', True)))
    for values in windows.values():
        values.sort()
        if any(x[1] > y[0] for x, y in zip(values, values[1:])):
            raise ValueError('coverage windows overlap')
    if not windows:
        raise ValueError('explicit scored/unknown coverage windows are required')

    def covered(row):
        a, b = _interval(row)
        return any(ok and x <= a and b <= y for x, y, ok in windows[(row['source'], row['epoch'])])

    truths, ignored_truth = [], []
    for row in annotations.get('events', []):
        keys(row, {'source','epoch','rule','start','end','eligible_at','censored'}, 'annotated event', {'source','epoch','rule','start','end'})
        _key(row); _interval(row)
        if type(row.get('censored', False)) is not bool:
            raise ValueError('censored must be boolean')
        (ignored_truth if row.get('censored') or not covered(row) else truths).append(row)
    preds, ignored_pred = [], []
    ids = set()
    for row in predictions.get('events', []):
        _key(row); _interval(row); number(row['timestamp'], 'timestamp')
        identity = text(row['id'], 'event id')
        # Repeated delivery of the same ID is reported separately from episodes.
        if identity in ids:
            raise ValueError('duplicate prediction ID; deduplicate transport before event scoring')
        ids.add(identity)
        a, b = _interval(row)
        uncertain = any(_key(row) == _key(t) and max(a, _interval(t)[0]) <= min(b, _interval(t)[1]) for t in ignored_truth)
        (ignored_pred if uncertain or not covered(row) else preds).append(row)

    edges = {}
    for i, pred in enumerate(preds):
        a, b = _interval(pred)
        candidates = []
        for j, truth in enumerate(truths):
            c, d = _interval(truth)
            intersection = max(0, min(b, d) - max(a, c))
            union = max(b, d) - min(a, c)
            overlap = intersection / union if union else 1.0
            if _key(pred) == _key(truth) and abs(a - c) <= start_tolerance + 1e-9 and overlap + 1e-9 >= min_interval_iou:
                candidates.append((abs(a-c), j))
        edges[i] = [j for _, j in sorted(candidates)]
    matched = _matching(edges)
    tp, fp, fn = len(matched), len(preds) - len(matched), len(truths) - len(matched)
    duplicates = sum(bool(edges[i]) for i in range(len(preds)) if i not in matched)
    delays = [preds[i]['timestamp'] - number(truths[j].get('eligible_at', _interval(truths[j])[1]), 'eligible_at') for i, j in matched.items()]

    expected = {}
    for row in annotations.get('counts', []):
        keys(row, {'source','epoch','rule','timestamp','count'}, 'annotated count', {'source','epoch','rule','timestamp','count'})
        key = (*_key(row), number(row['timestamp'], 'timestamp'))
        if key in expected:
            raise ValueError('duplicate count annotation key')
        if row.get('count') is not None and (type(row['count']) is not int or row['count'] < 0):
            raise ValueError('count must be null or a nonnegative integer')
        expected[key] = row
    actual = {}
    for row in predictions.get('counts', []):
        key = (*_key(row), number(row['timestamp'], 'timestamp'))
        if key in actual:
            raise ValueError('duplicate prediction count key')
        if row.get('count') is not None and (type(row['count']) is not int or row['count'] < 0):
            raise ValueError('count must be null or a nonnegative integer')
        actual[key] = row
    errors, unknown_counts, missing_counts, unscored_counts = [], 0, 0, 0
    for key, row in expected.items():
        point = {**row, 'start': row['timestamp'], 'end': row['timestamp']}
        if row.get('count') is None or not covered(point):
            unscored_counts += 1
        elif key not in actual:
            missing_counts += 1
        elif actual[key].get('count') is None:
            unknown_counts += 1
        else:
            errors.append(actual[key]['count'] - row['count'])
    scored_seconds = sum(b-a for v in windows.values() for a,b,ok in v if ok)
    explicit_unknown = sum(b-a for v in windows.values() for a,b,ok in v if not ok)
    unannotated_gaps = sum(max(0,y[0]-x[1]) for v in windows.values() for x,y in zip(v,v[1:]))
    unknown_seconds = explicit_unknown + unannotated_gaps
    mean = lambda values: sum(values)/len(values) if values else None
    return {
        'schema_version': 'evaluation-report/1',
        'split': annotations.get('split', 'unspecified'),
        'matching': {'start_tolerance': start_tolerance, 'min_interval_iou': min_interval_iou},
        'events': {'tp': tp, 'fp': fp, 'fn': fn, 'duplicates': duplicates,
                   'precision': tp/(tp+fp) if tp+fp else None,
                   'recall': tp/(tp+fn) if tp+fn else None,
                   'false_alerts_per_camera_hour': fp/(scored_seconds/3600) if scored_seconds else None,
                   'delay_seconds': {'samples': len(delays), 'mean': mean(delays), 'min': min(delays) if delays else None, 'max': max(delays) if delays else None},
                   'ignored_truth': len(ignored_truth), 'ignored_predictions': len(ignored_pred),
                   'matches': [{'prediction_id': preds[i]['id'], 'truth_index': j} for i,j in sorted(matched.items())]},
        'counts': {'scored_samples': len(errors), 'mae': mean([abs(x) for x in errors]),
                   'rmse': sqrt(mean([x*x for x in errors])) if errors else None,
                   'bias': mean(errors), 'missing_samples': missing_counts,
                   'unknown_prediction_samples': unknown_counts, 'unscored_annotations': unscored_counts,
                   'extra_prediction_samples': len(set(actual)-set(expected))},
        'coverage': {'scored_camera_seconds': scored_seconds, 'unknown_camera_seconds': unknown_seconds,
                     'explicit_unknown_camera_seconds':explicit_unknown,'unannotated_gap_camera_seconds':unannotated_gaps,
                     'unknown_fraction': unknown_seconds/(scored_seconds+unknown_seconds)},
    }


def validate_split_groups(datasets):
    """Prevent camera/day leakage when callers supply their explicit group IDs."""
    seen = {}
    for dataset in datasets:
        split = text(dataset['split'], 'split')
        for group in dataset.get('groups', []):
            group = text(group, 'camera/day group')
            if group in seen and seen[group] != split:
                raise ValueError(f'group {group!r} occurs in both {seen[group]} and {split}')
            seen[group] = split
    return seen


def evaluate_tracks(annotated_frames, predicted_frames, *, iou_threshold=.5):
    """Detection matching, ID switches and fragments on annotated sampled frames.

    This deliberately does not call these metrics HOTA or IDF1. Unknown coverage
    breaks continuity without a switch/fragment penalty. A fragment is a match
    after a known, annotated missed observation of the same truth identity.
    """
    from .core import Observation
    from .tracking import iou
    number(iou_threshold,'iou_threshold',0,1)
    predicted={}
    for row in predicted_frames:
        o=Observation.from_dict(row)
        key=(o.source,o.epoch,o.timestamp)
        if key in predicted: raise ValueError('duplicate predicted frame')
        predicted[key]=o
    total=matched=false_positive=switches=fragments=unknown=0
    history,clocks={},{}
    for row in annotated_frames:
        truth=Observation.from_dict(row); key=(truth.source,truth.epoch,truth.timestamp)
        clock=(truth.source,truth.epoch)
        if clock in clocks and truth.timestamp<=clocks[clock]: raise ValueError('annotation frames must be ordered')
        clocks[clock]=truth.timestamp
        prediction=predicted.pop(key,None)
        if not truth.covered or prediction is not None and not prediction.covered:
            unknown+=1
            history={k:v for k,v in history.items() if k[:2]!=clock}
            continue
        detections=prediction.detections if prediction else ()
        edges={i:[j for _,j in sorted((-iou(t.box,p.box),j) for j,p in enumerate(detections)
                                     if t.label==p.label and iou(t.box,p.box)>=iou_threshold)]
               for i,t in enumerate(truth.detections)}
        assignment=_matching(edges)
        total+=len(truth.detections); matched+=len(assignment); false_positive+=len(detections)-len(assignment)
        present=set()
        for i,t in enumerate(truth.detections):
            hkey=(*clock,t.track_id); present.add(hkey); old=history.get(hkey)
            if i in assignment:
                pid=detections[assignment[i]].track_id
                if old:
                    if old['id']!=pid: switches+=1
                    if old['missed']: fragments+=1
                history[hkey]={'id':pid,'missed':False}
            elif old:
                old['missed']=True
        # Explicit disappearance ends this annotated object episode.
        history={k:v for k,v in history.items() if k[:2]!=clock or k in present}
    return {'schema_version':'tracking-report/1','truth_detections':total,'matched':matched,
            'missed':total-matched,'false_positive':false_positive,'id_switches':switches,
            'fragments':fragments,'unknown_frames':unknown,'unannotated_prediction_frames':len(predicted),
            'iou_threshold':iou_threshold}
