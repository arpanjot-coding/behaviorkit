import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import time

import pytest


def test_real_pyav_http_eof_reconnect_timeout_and_epoch_isolation(tmp_path):
    av=pytest.importorskip('av'); np=pytest.importorskip('numpy')
    from behaviorkit.live import LiveSources,SourceSpec
    path=tmp_path/'stream.ts'
    with av.open(str(path),'w',format='mpegts') as output:
        stream=output.add_stream('mpeg2video',rate=10); stream.width=160; stream.height=120; stream.pix_fmt='yuv420p'
        for _ in range(25):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((120,160,3),np.uint8),format='rgb24')): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    body=path.read_bytes(); connections=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a): pass
        def do_GET(self):
            connections.append(self.path)
            self.send_response(200); self.send_header('Content-Type','video/mp2t'); self.end_headers()
            try:
                if self.path.startswith('/stall'):
                    self.wfile.flush(); time.sleep(.8)
                else:
                    self.wfile.write(body); self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    packets=[]
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        specs=[SourceSpec('reconnect',base+'/stream?token=PRIVATE',policy='lossless',queue_size=8,stale_seconds=10,
                          max_retries=1,retry_seconds=.01,open_timeout=1,read_timeout=1),
               SourceSpec('timeout',base+'/stall?token=PRIVATE',max_retries=0,open_timeout=.1,read_timeout=.1)]
        with LiveSources(specs) as live:
            until=time.monotonic()+10
            while not live.finished and time.monotonic()<until:
                packet=live.poll()
                if packet:
                    packet.pop('rgb',None); packets.append(packet)
            assert live.finished
            assert live.status()['timeout']['state']=='failed'
        frames=[p for p in packets if p['kind']=='frame' and p['source']=='reconnect']
        assert len({p['epoch'] for p in frames})==2
        assert any(p.get('state')=='recovering' for p in packets)
        assert sum(x.startswith('/stream') for x in connections)==2
        assert 'PRIVATE' not in json.dumps(packets)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
