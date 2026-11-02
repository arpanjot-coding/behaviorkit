"""Loopback-only zone editor. Media stays in the browser, no cloud service."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from .advanced import parse_scene
from .core import number
from .setup import Geometry


def validate_setup(data):
    scene=parse_scene(data['scene']); geometry=Geometry(**data['geometry'])
    cadence=number(data['sample_fps'],'sample_fps',.001,120)
    warnings=[]
    for rule in scene.rules:
        if rule.kind!='count' and rule.max_gap_seconds+1e-9<1/cadence:
            raise ValueError(f'{rule.id}: maximum gap is shorter than sampling period')
        if rule.kind=='dwell' and rule.seconds<2/cadence:
            warnings.append(f'{rule.id}: dwell threshold spans fewer than two sample intervals')
    return {'valid':True,'scene':scene.to_dict(),'geometry':data['geometry'],'sample_fps':cadence,
            'geometry_fingerprint':geometry.fingerprint,'warnings':warnings,
            'next':'Run doctor, then capture a positive/negative clip and evaluate held-out events.'}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def _respond(self,status,value,kind='application/json'):
        raw=value.encode() if isinstance(value,str) else json.dumps(value,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type',kind+'; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; media-src blob:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
        self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/': self._respond(200,files('behaviorkit').joinpath('wizard.html').read_text(encoding='utf-8'),'text/html')
        else: self._respond(404,{'error':'not found'})
    def do_POST(self):
        if self.path!='/validate': self._respond(404,{'error':'not found'}); return
        origin=self.headers.get('Origin')
        expected=f'http://127.0.0.1:{self.server.server_port}'
        if origin and origin!=expected:
            self._respond(403,{'error':'origin rejected'}); return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024*1024: raise ValueError('setup request must be in 1..1048576 bytes')
            result=validate_setup(json.loads(self.rfile.read(length)))
            self._respond(200,result)
        except (ValueError,KeyError,TypeError) as error:
            self._respond(400,{'error':str(error)})


def make_server(port=8765):
    if type(port) is not int or not 0<=port<=65535: raise ValueError('invalid port')
    return ThreadingHTTPServer(('127.0.0.1',port),_Handler)


def serve(port=8765):
    with make_server(port) as server:
        print(f'BehaviorKit setup: http://127.0.0.1:{server.server_port}/',flush=True)
        try: server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt: pass
