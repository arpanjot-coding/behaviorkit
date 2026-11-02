"""Prepare silent H.264 excerpts from explicitly supplied local demo sources.

No network access. Source URLs and separate media licenses are in sources.json.
"""
from pathlib import Path
from fractions import Fraction
import argparse,json
import av
from behaviorkit.storage import sha256,write_json,guard_outputs

ROOT=Path(__file__).resolve().parents[1]

def prepare(source,target,start,seconds,max_width=960):
    guard_outputs(target.parent,[source],[target.name])
    with av.open(str(source)) as incoming:
        stream=incoming.streams.video[0]
        width=min(max_width,stream.width);height=round(stream.height*width/stream.width/2)*2;width=round(width/2)*2
        rate=Fraction(15,1)
        incoming.seek(max(0,int((start-1)*1e6)))
        with av.open(str(target),'w',options={'movflags':'+faststart'}) as outgoing:
            encoded=outgoing.add_stream('libx264',rate=rate)
            encoded.width=width;encoded.height=height;encoded.pix_fmt='yuv420p';encoded.options={'crf':'23','preset':'fast'}
            count=0;due=start
            for frame in incoming.decode(video=0):
                t=float(frame.time)
                if t+1e-7<due:continue
                if t>=start+seconds-1e-7:break
                prepared=frame.reformat(width=width,height=height,format='yuv420p');prepared.pts=count;prepared.time_base=Fraction(1,15)
                for packet in encoded.encode(prepared):outgoing.mux(packet)
                count+=1;due=start+count/15
            for packet in encoded.encode():outgoing.mux(packet)
    if not count:raise ValueError('source excerpt had no frames')
    return {'frames':count,'fps':15,'width':width,'height':height,'seconds':count/15,'sha256':sha256(target),'bytes':target.stat().st_size,'audio':False,'transform':'Silent excerpt; resized to <=960 pixels wide; sampled at 15 fps; source offset mapped to clip time zero.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'examples/video-lab/media');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'examples/video-lab/sources.json').read_text(encoding='utf-8'))
    for key,source in manifest['sources'].items():source['sha256']=sha256(a.sources_dir/source['file'])
    for case in manifest['cases']:
        target=a.output/(case['id']+'.mp4')
        case['prepared']=prepare(a.sources_dir/manifest['sources'][case['source']]['file'],target,case['start'],case['seconds'])
        print(case['id'],case['prepared']['frames'],case['prepared']['bytes'],flush=True)
    write_json(a.output/'manifest.json',manifest)
if __name__=='__main__':main()
