"""Bounded PyAV ingest in killable decoder processes.

Evidence age is time since local decode receipt, not camera capture latency.
Lossless means backpressure within this adapter, not guaranteed camera/network
delivery. Inference remains synchronous in the application process.
"""
from __future__ import annotations
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time
from urllib.parse import urlsplit
import uuid

from .core import number, text


@dataclass(frozen=True)
class SourceSpec:
    id: str
    uri: str
    policy: str = 'latest'
    queue_size: int = 2
    sample_fps: float = 5
    open_timeout: float = 5
    read_timeout: float = 5
    stale_seconds: float = 3
    max_retries: int = 3
    retry_seconds: float = .5
    reconnect_at_eof: bool | None = None

    def __post_init__(self):
        text(self.id,'source id'); text(self.uri,'source URI')
        if self.policy not in {'latest','lossless'}: raise ValueError('queue policy must be latest or lossless')
        if type(self.queue_size) is not int or not 1<=self.queue_size<=64: raise ValueError('queue_size must be in 1..64')
        if type(self.max_retries) is not int or not 0<=self.max_retries<=100: raise ValueError('max_retries must be in 0..100')
        for k in ('open_timeout','read_timeout','stale_seconds','retry_seconds'):
            number(getattr(self,k),k,.01,300)
        number(self.sample_fps,'sample_fps',.001,120)
        if self.reconnect_at_eof is None:
            object.__setattr__(self,'reconnect_at_eof',not Path(self.uri).is_file())
        if type(self.reconnect_at_eof) is not bool: raise ValueError('reconnect_at_eof must be boolean')
        scheme=urlsplit(self.uri).scheme.lower()
        # A Windows drive prefix is not a network scheme.
        if not Path(self.uri).is_file() and scheme not in {'rtsp','rtsps','http','https'}:
            raise ValueError('source must be a local file or explicit RTSP/HTTP URL')


def _put(queue, packet, policy, stop, drops, heartbeat=None):
    while not stop.is_set():
        if heartbeat is not None: heartbeat.value=time.monotonic()
        try:
            if policy=='latest': queue.put_nowait(packet)
            else: queue.put(packet,timeout=.05)
            return True
        except Full:
            if policy=='latest':
                try:
                    old=queue.get_nowait()
                    if old.get('kind')=='frame':
                        with drops.get_lock(): drops.value+=1
                except Empty: stop.wait(.001) # multiprocessing feeder may still be publishing
    return False


def _decode_worker(spec, queue, stop, drops, heartbeat):
    import av
    from .cli import next_sample_time
    retries=0
    while not stop.is_set():
        epoch=uuid.uuid4().hex
        try:
            heartbeat.value=time.monotonic()
            with av.open(spec.uri,timeout=(spec.open_timeout,spec.read_timeout)) as container:
                origin=next_due=last=None
                for frame in container.decode(video=0):
                    if stop.is_set(): return
                    received=time.monotonic(); heartbeat.value=received
                    if frame.pts is None or frame.time_base is None:
                        timestamp=received; clock='monotonic'
                    else:
                        timestamp=float(frame.pts*frame.time_base); clock='video_pts'
                    number(timestamp,'decoded timestamp')
                    if last is not None and timestamp<=last: raise ValueError('source timestamp discontinuity')
                    last=timestamp
                    if next_due is not None and timestamp+1e-9<next_due: continue
                    if origin is None: origin=timestamp
                    next_due=next_sample_time(timestamp,origin,spec.sample_fps)
                    rgb=frame.to_ndarray(format='rgb24')
                    if rgb.nbytes>64*1024*1024: raise ValueError('decoded frame exceeds 64 MiB; resize source')
                    packet={'kind':'frame','source':spec.id,'epoch':epoch,'timestamp':timestamp,'clock':clock,
                            'received_monotonic':received,'rgb':rgb}
                    if not _put(queue,packet,spec.policy,stop,drops,heartbeat): return
                if not spec.reconnect_at_eof:
                    _put(queue,{'kind':'health','source':spec.id,'state':'stopped','reason':'end_of_stream'},'lossless',stop,drops,heartbeat)
                    return
                raise EOFError('live source ended')
        except Exception as error:
            retries+=1
            state='failed' if retries>spec.max_retries else 'recovering'
            # No URI/error text: credentials and camera addresses stay out of logs.
            packet={'kind':'health','source':spec.id,'state':state,'reason':type(error).__name__,
                    'retry':retries,'epoch':epoch}
            _put(queue,packet,'latest',stop,drops,heartbeat)
            if state=='failed': return
            if stop.wait(min(30,spec.retry_seconds*2**min(retries-1,6))): return


class LiveSources:
    def __init__(self, specs, *, context='spawn'):
        specs=tuple(specs)
        if not 1<=len(specs)<=16 or not all(isinstance(s,SourceSpec) for s in specs) or len({s.id for s in specs})!=len(specs):
            raise ValueError('require 1..16 uniquely identified SourceSpec objects')
        self.specs=specs; self.ctx=mp.get_context(context); self._workers={}; self._cursor=0

    def __enter__(self):
        try:
            for spec in self.specs:
                queue=self.ctx.Queue(maxsize=spec.queue_size); stop=self.ctx.Event()
                drops=self.ctx.Value('Q',0); heartbeat=self.ctx.Value('d',time.monotonic())
                worker=self.ctx.Process(target=_decode_worker,args=(spec,queue,stop,drops,heartbeat),daemon=True)
                worker.start()
                self._workers[spec.id]={'spec':spec,'queue':queue,'stop':stop,'drops':drops,'heartbeat':heartbeat,
                                        'process':worker,'state':'connecting','last_frame':None,'stale_drops':0,'terminal':False}
        except BaseException:
            self.close(); raise
        return self

    def poll(self, timeout=.1):
        number(timeout,'poll timeout',0,1)
        deadline=time.monotonic()+timeout
        while True:
            for offset in range(len(self.specs)):
                i=(self._cursor+offset)%len(self.specs); state=self._workers[self.specs[i].id]
                if state['terminal']: continue
                now=time.monotonic(); spec=state['spec']
                try: packet=state['queue'].get_nowait()
                except Empty: packet=None
                if packet is not None:
                    self._cursor=(i+1)%len(self.specs)
                    if packet['kind']=='frame':
                        age=now-packet['received_monotonic']
                        if age>spec.stale_seconds:
                            state['stale_drops']+=1; state['state']='stale'
                            return {'kind':'health','source':spec.id,'state':'stale','reason':'queued_frame_too_old','queue_age_seconds':age}
                        state['state']='healthy'; state['last_frame']=now
                        packet['queue_age_seconds']=age; packet['dropped_frames']=state['drops'].value
                    else:
                        state['state']=packet['state']
                        if state['state'] in {'stopped','failed'}: state['terminal']=True
                    return packet
                # A stuck native decoder is terminated independently of other sources.
                budget=spec.open_timeout+spec.read_timeout+max(2,spec.stale_seconds)
                if now-state['heartbeat'].value>budget and state['process'].is_alive():
                    state['stop'].set(); state['process'].terminate(); state['process'].join(timeout=.1)
                    state['terminal']=True; state['state']='failed'
                    return {'kind':'health','source':spec.id,'state':'failed','reason':'decoder_watchdog_timeout'}
                if not state['process'].is_alive():
                    # Queue feeder delivery may trail process exit briefly.
                    state['terminal']=True; state['state']='stopped' if state['process'].exitcode==0 else 'failed'
                    return {'kind':'health','source':spec.id,'state':state['state'],'reason':'decoder_exited'}
                if state['last_frame'] is not None and now-state['last_frame']>spec.stale_seconds and state['state']=='healthy':
                    state['state']='stale'
                    return {'kind':'health','source':spec.id,'state':'stale','reason':'no_fresh_decoded_frame'}
            if time.monotonic()>=deadline: return None
            time.sleep(min(.01,max(0,deadline-time.monotonic())))

    @property
    def finished(self): return bool(self._workers) and all(s['terminal'] for s in self._workers.values())

    def status(self):
        return {k:{'state':v['state'],'dropped_frames':v['drops'].value,'stale_drops':v['stale_drops'],
                   'policy':v['spec'].policy,'queue_capacity':v['spec'].queue_size} for k,v in self._workers.items()}

    def close(self):
        for state in self._workers.values(): state['stop'].set()
        for state in self._workers.values():
            state['process'].join(timeout=.5)
            if state['process'].is_alive():
                state['process'].terminate(); state['process'].join(timeout=.5)
            state['queue'].cancel_join_thread(); state['queue'].close()

    def __exit__(self,*exc): self.close()
