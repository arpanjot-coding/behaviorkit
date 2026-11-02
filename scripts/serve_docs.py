"""Preview the static docs with byte-range support for reliable video seeking."""
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,os,re

class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining=None
        path=self.translate_path(self.path)
        requested=self.headers.get('Range','')
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
        if not match or not os.path.isfile(path):return super().send_head()
        f=open(path,'rb');size=os.fstat(f.fileno()).st_size
        first,last=match.groups()
        if not first and not last:
            f.close();return super().send_head()
        start=int(first) if first else max(0,size-int(last))
        end=min(int(last),size-1) if first and last else size-1
        if start>=size or start>end or (not first and int(last)==0):
            f.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
        self.send_response(206);self.send_header('Content-Type',self.guess_type(path));self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Content-Length',str(end-start+1));self.send_header('Last-Modified',self.date_time_string(os.fstat(f.fileno()).st_mtime));self.end_headers()
        f.seek(start);self.remaining=end-start+1;return f

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes');super().end_headers()

    def copyfile(self,source,outputfile):
        if self.remaining is None:return super().copyfile(source,outputfile)
        remaining=self.remaining
        while remaining:
            data=source.read(min(64*1024,remaining))
            if not data:break
            outputfile.write(data);remaining-=len(data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8080);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'website/dist');a=p.parse_args()
    with ThreadingHTTPServer(('127.0.0.1',a.port),partial(RangeHandler,directory=str(a.directory))) as server:
        print(f'Local docs: http://127.0.0.1:{a.port}/demo.html',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
if __name__=='__main__':main()
